"""
The following code was based off of samples from (PyTorch CNN Visualizations)
https://github.com/utkuozbulak/pytorch-cnn-visualizations

The generated code was reviewed and modified to provide
additional visualization of the target layer heatmap.
"""

import os
import argparse

# Chart modules (Matplotlib)
# https://matplotlib.org/stable/api/index.html
import matplotlib.pyplot as plt

# CNN functions (PyTorch)
# https://docs.pytorch.org/docs/stable/index.html
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

# Image manipulation (Pillow)
# https://pillow.readthedocs.io/en/stable/
from PIL import Image

from config import DEVICE, IMAGE_SIZE, OUTPUT_DIR
from model_utils import get_last_conv_layer, load_checkpoint


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        # ImageNet standard normalization mean/sd
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

# Reverse ImageNet normalization
def unnormalize_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def replace_inplace_relu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            replace_inplace_relu(child)


def generate_gradcam(model, model_name, image_tensor, class_idx=None):
    activations = []
    gradients = []

    target_layer = get_last_conv_layer(model, model_name)

    def save_activations(_, __, output):
        activations.append(output.detach())

    def save_gradients(_, __, grad_output):
        gradients.append(grad_output[0].detach())

    # Captures layer outputs (feature maps)
    forward_hook = target_layer.register_forward_hook(save_activations)

    # Captures gradients during backpropagation
    backward_hook = target_layer.register_full_backward_hook(save_gradients)

    try:
        model.zero_grad()
        output = model(image_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        output[:, class_idx].backward()

        acts = activations[0]
        grads = gradients[0]

        print("\nTarget layer:", target_layer)

        # Ensure activations are in channels-first format [B, C, H, W].
        # ConvNeXt may output channels-last [B, H, W, C].
        if acts.ndim == 4 and acts.shape[-1] > acts.shape[1]:
            acts = acts.permute(0, 3, 1, 2)
            grads = grads.permute(0, 3, 1, 2)

        # Grad-CAM uses the average gradient of each feature map as its importance weight.
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # Weighted feature maps show which regions most influenced the predicted class.
        cam = (weights * acts).sum(dim=1, keepdim=True)

        # Keep only positive evidence for the chosen class.
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, class_idx
    finally:
        forward_hook.remove()
        backward_hook.remove()


def save_overlay(image_tensor, cam, output_path):
    image = unnormalize_image(image_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()

    '''
    Chart code samples were modified from (Matplotlib)
    https://matplotlib.org/stable/api/index.html
    '''
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.imshow(cam, cmap="jet", alpha=0.4)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model, class_names, model_name = load_checkpoint(args.checkpoint, DEVICE)
    replace_inplace_relu(model)

    image = Image.open(args.image).convert("RGB")
    image_tensor = get_transform()(image).unsqueeze(0).to(DEVICE)

    cam, class_idx = generate_gradcam(model, model_name, image_tensor)

    out_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

    output_path = args.output or os.path.join(out_dir, "gradcam_overlay.png")
    save_overlay(image_tensor, cam, output_path)

    print(f"Predicted class: {class_names[class_idx]}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
