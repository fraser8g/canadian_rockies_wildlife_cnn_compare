import os
import argparse
import copy
from datetime import datetime
from collections import Counter

# CNN functions (PyTorch)
# https://docs.pytorch.org/docs/stable/index.html
import torch
import torch.nn as nn
import torch.optim as optim

# Reporting analytics (pandas)
# https://pandas.pydata.org/docs/user_guide/index.html
import pandas as pd

# Chart modules (Matplotlib)
# https://matplotlib.org/stable/api/index.html
import matplotlib.pyplot as plt

from config import DEVICE, OUTPUT_DIR, MODELS, NUM_EPOCHS, PHASE1_EPOCHS, PHASE1_LR, PHASE2_EPOCHS, PHASE2_LR, PATIENCE, LABEL_SMOOTHING, INCLUDED_PRETRAINING
from dataset_utils import get_dataloaders
from model_utils import build_model, save_checkpoint


def get_class_weights(train_loader, num_classes, device):
    # Count training samples per class (directories in data/train)
    targets = train_loader.dataset.targets
    class_counts = Counter(targets)
    weights = []

    # Inverse frequency weighting
    for class_idx in range(num_classes):
        count = class_counts.get(class_idx, 1)
        weights.append(1.0 / count)

    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # Normalize weights
    weights = weights / weights.sum() * num_classes
    return weights


def train_one_epoch(model, loader, criterion, optimizer, device):
    # Training model
    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    '''
    The below is a modified code sample from (pytorch-tutorial)
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py
    '''
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Class predictions
        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Average loss and accuracy 
    return running_loss / total, running_correct / total


def validate(model, loader, criterion, device):
    # Validation model
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    '''
    The below is a modified code sample from (pytorch-tutorial)
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py
    '''
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Class predictions
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Average loss and accuracy 
    return running_loss / total, running_correct / total


# Transfer performance improvement to freeze all pretrained layers 
# and only allow the classifier layers to update for Phase 1
# to enable efficient transfer learning on the new image dataset.
def freeze_all_except_classifier(model, model_name):
    for param in model.parameters():
        param.requires_grad = False

    if model_name in MODELS:
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=MODELS)
    parser.add_argument("--phase1-lr", type=float, default=PHASE1_LR)
    parser.add_argument("--phase2-lr", type=float, default=PHASE2_LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    # Only training and validation sets used
    train_loader, val_loader, _, class_names = get_dataloaders()
    num_classes = len(class_names)

    print("--- Started Training ---")
    start_time = datetime.now()
    print(f"Start time: {start_time}")
    print(f"\nStarted training model: {args.model} on device: {DEVICE} | Batch size: {train_loader.batch_size} | Workers: {train_loader.num_workers}")
    print(f"Hyper parameters: Phase 1 epochs: {PHASE1_EPOCHS} | Phase 1 lr: {PHASE1_LR} | Phase 2 epochs: {PHASE2_EPOCHS} | Phase 1 lr: {PHASE2_LR} | Smoothing: {LABEL_SMOOTHING}")

    class_weights = get_class_weights(train_loader, num_classes, DEVICE)
    weightlist = list(zip(class_names, class_weights.tolist()))
    print(f"Class weights:")
    for name, weight in weightlist:
        print(f"\t{name}: {weight:.4f}")

    model = build_model(args.model, num_classes=num_classes, freeze_backbone=False)
    model.to(DEVICE)

    print(model)

    '''
    Criterion and Optimizer utilized from samples in (pytorch-tutorial)
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py
    '''
    # Loss function
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    # PyTorch optimizers were added to improve training performance time
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase1_lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=PATIENCE
    )

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = []
    epochs_without_improvement = 0

    # Only freeze the backbone for pre-trained models for Phase 1
    # and when pre-training weights are included.
    if args.model != "wildlife_cnn" and INCLUDED_PRETRAINING == 1: 
        # Phase 1: train head only to adapt the model to the new dataset
        print("\n--- Phase 1: Training Classifier Head (Linear Layers) ---")
        freeze_all_except_classifier(model, args.model)
    else:
        print("\n--- Phase 1: Non Pre-trained Training ---")

    for epoch in range(PHASE1_EPOCHS):
        current_epoch = epoch + 1
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        # Print Epoch results
        print(
            f"Epoch {current_epoch}/{NUM_EPOCHS} | "
            f"Phase: 1 | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Create Epoch history row
        history.append({
            "epoch": current_epoch,
            "phase": 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # When no improvements are made after patience param met, trigger stop
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping triggered during phase 1 after epoch {current_epoch}.")
            break

    # Phase 2: fine-tuning
    if epochs_without_improvement < args.patience:
        print("\n--- Phase 2: Fine-Tuning Feature Extractor ---")

        # Full backbone allowing all layers to be updated
        unfreeze_all_layers(model)

        # PyTorch optimizers were added to improve training performance times
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.phase2_lr,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=PATIENCE
        )

        for epoch in range(PHASE2_EPOCHS):
            current_epoch = PHASE1_EPOCHS + epoch + 1
            current_lr = optimizer.param_groups[0]["lr"]

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

            scheduler.step(val_loss)

            # Print Epoch results
            print(
                f"Epoch {current_epoch}/{NUM_EPOCHS} | "
                f"Phase: 2 | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Create Epoch history row
            history.append({
                "epoch": current_epoch,
                "phase": 2,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            # When no improvements are made after patience param met, trigger stop
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered during phase 2 after epoch {current_epoch}.")
                break

    model.load_state_dict(best_model_wts)

    save_path = save_checkpoint(model, class_names, args.model)

    # Export history to a csv
    os.makedirs(os.path.join(OUTPUT_DIR, args.model), exist_ok=True)
    history_path = os.path.join(OUTPUT_DIR, args.model, "history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)

    # Extract full metric history from all epochs
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]
    train_accs = [row["train_acc"] for row in history]
    val_accs = [row["val_acc"] for row in history]

    '''
    Chart code samples were modified from (Matplotlib)
    https://matplotlib.org/stable/api/index.html
    '''
    # Create figure with two rows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ---- Loss plot ----
    ax1.plot(epochs, train_losses, marker="o", linewidth=2, label="Train Loss", color="#66cc6d")
    ax1.plot(epochs, val_losses, marker="s", linewidth=2, label="Validation Loss", color="#1f3d21")
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"Training History - {args.model}", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ---- Accuracy plot ----
    ax2.plot(epochs, train_accs, marker="o", linewidth=2, label="Train Accuracy", color="#66cc6d")
    ax2.plot(epochs, val_accs, marker="s", linewidth=2, label="Validation Accuracy", color="#1f3d21")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(epochs)

    # Save figure
    save_dir = os.path.join(OUTPUT_DIR, args.model)
    os.makedirs(save_dir, exist_ok=True)
    chart_path = os.path.join(save_dir, "training_curve.png")

    fig.tight_layout()
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    end_time = datetime.now()

    print(f"\n--- Results ---")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {save_path}")
    print(f"History saved to: {history_path}")
    print(f"End time: {end_time}")
    print(f"Duration: {end_time - start_time}")


if __name__ == "__main__":
    main()