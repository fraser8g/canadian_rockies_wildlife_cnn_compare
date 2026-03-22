import os
import argparse

# CNN functions (PyTorch)
# https://docs.pytorch.org/docs/stable/index.html
import torch

# Reporting analytics (pandas)
# https://pandas.pydata.org/docs/user_guide/index.html
import pandas as pd

# Data calculations (numpy)
# https://numpy.org/doc/stable/user/index.html
import numpy as np

# Chart modules (Matplotlib)
# https://matplotlib.org/stable/api/index.html
import matplotlib.pyplot as plt

# Heatmap presentation (seaborn)
# https://seaborn.pydata.org/api.html
import seaborn as sns

from torchvision import datasets
from torch.utils.data import DataLoader

from config import DEVICE, OUTPUT_DIR, MODELS, TEST_DIR, BATCH_SIZE
from dataset_utils import get_transforms
from model_utils import load_checkpoint

# Output metrics (scikit)
# https://scikit-learn.org/stable/api/sklearn.metrics.html
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

def evaluate(model_path):
    print(f"\nLoading model: {model_path}")

    model, class_names, model_name = load_checkpoint(model_path, DEVICE)
    model.eval()

    _, val_transform = get_transforms()

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=val_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda")
    )

    all_preds = []
    all_labels = []
    all_top3 = []

    '''
    The below is a modified code sample from (pytorch-tutorial)
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py
    '''
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            preds = outputs.argmax(dim=1)
            _, top3_indices = torch.topk(outputs, k=3, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_top3.extend(top3_indices.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_top3 = np.array(all_top3)

    # Base classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    # Confusion matrix for per-class accuracy
    cm = confusion_matrix(all_labels, all_preds)
    row_sums = cm.sum(axis=1)
    class_accuracy = np.divide(
        cm.diagonal(),
        row_sums,
        out=np.zeros_like(cm.diagonal(), dtype=float),
        where=row_sums != 0
    )

    # Overall Top-1 accuracy
    top1_accuracy = accuracy_score(all_labels, all_preds)
    
    # Overall Top-3 accuracy
    top3_correct = sum(label in top3 for label, top3 in zip(all_labels, all_top3))
    top3_accuracy = top3_correct / len(all_labels)

    print(f"\nOverall Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Overall Top-3 Accuracy: {top3_accuracy:.4f}\n")

    # Per-class Top-3 accuracy
    class_top3_accuracy = []
    for class_idx in range(len(class_names)):
        mask = (all_labels == class_idx)
        n_class = mask.sum()

        if n_class == 0:
            class_top3_accuracy.append(0.0)
        else:
            correct_top3 = sum(
                class_idx in top3
                for top3 in all_top3[mask]
            )
            class_top3_accuracy.append(correct_top3 / n_class)

    # Add per-animal accuracy + top3 accuracy columns
    for i, class_name in enumerate(class_names):
        report_df.loc[class_name, "accuracy"] = class_accuracy[i]
        report_df.loc[class_name, "top3_accuracy"] = class_top3_accuracy[i]

    # Add overall summary values
    if "accuracy" in report_df.index:
        report_df.loc["accuracy", "accuracy"] = top1_accuracy
        report_df.loc["accuracy", "top3_accuracy"] = top3_accuracy

    report_df.loc["macro avg", "top3_accuracy"] = np.mean(class_top3_accuracy)
    report_df.loc["weighted avg", "top3_accuracy"] = np.average(
        class_top3_accuracy,
        weights=row_sums
    )

    print("\nPer-class metrics:")
    print(report_df.round(4))

    save_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save report CSV
    report_path = os.path.join(save_dir, "classification_report.csv")
    report_df.to_csv(report_path)

    # Normalized confusion matrix
    cm_norm = confusion_matrix(all_labels, all_preds, normalize="true")
    n_samples = len(all_labels)

    '''
    Chart code samples were modified from (Matplotlib)
    # https://matplotlib.org/stable/api/index.html
    '''
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.85}
    )

    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    # Added Top-1 and Top-3 accuracy metrics to the plot chart title.
    plt.title(
        f"Normalized Confusion Matrix\nTop-1 Accuracy = {top1_accuracy:.4f} | Top-3 Accuracy = {top3_accuracy:.4f} | Samples = {n_samples}",
        fontsize=14
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nClassification report saved to: {report_path}")
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=MODELS)
    args = parser.parse_args()

    model_path = os.path.join(OUTPUT_DIR, args.model, "model.pth")
    evaluate(model_path)