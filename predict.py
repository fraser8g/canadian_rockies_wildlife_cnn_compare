import argparse

# CNN functions (PyTorch)
# https://docs.pytorch.org/docs/stable/index.html
import torch

# Image manipulation (Pillow)
# https://pillow.readthedocs.io/en/stable/
from PIL import Image

from config import DEVICE
from dataset_utils import get_transforms
from model_utils import load_checkpoint


def predict_image(model, image_path, class_names, device, top_k=3):
    # Only use the validation transform
    _, val_transform = get_transforms()

    image = Image.open(image_path).convert("RGB")
    x = val_transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        # Softmax converts logits into normalized class probabilities.
        probs = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        results.append((class_names[idx.item()], prob.item()))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    model, class_names, model_name = load_checkpoint(args.checkpoint, DEVICE)
    predictions = predict_image(model, args.image, class_names, DEVICE)

    print(f"\nModel: {model_name}")
    print(f"Image: {args.image}")
    print("Top predictions:")
    for label, confidence in predictions:
        print(f"{label}: {confidence:.4f}")


if __name__ == "__main__":
    main()
