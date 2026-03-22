import os

# CNN functions (PyTorch)
# https://docs.pytorch.org/docs/stable/index.html
import torch
# https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
import torch.nn as nn

#Import CNN model for PyTorch and pre-trained ImageNet weights.
from torchvision.models import (
    AlexNet_Weights,
    ConvNeXt_Tiny_Weights,
    EfficientNet_V2_S_Weights,
    VGG16_Weights,
    alexnet,
    convnext_tiny,
    efficientnet_v2_s,
    vgg16,
)

from config import OUTPUT_DIR


class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        #[1x1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False

  
def build_model(model_name, num_classes, freeze_backbone=True):
    if model_name == "baseline_cnn":
        model = BaselineCNN(num_classes)

    elif model_name == "convnext_tiny":
        # Best available pre-trained weights from ImageNet
        weights = ConvNeXt_Tiny_Weights.DEFAULT 
        model = convnext_tiny(weights=weights)

        # Initially prevent gradient descent due to pre-training
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # print(model) -- [0]: LayerNorm, [1]: Flatten, [2]: Linear 
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_v2_s":
        # Best available pre-trained weights from ImageNet
        weights = EfficientNet_V2_S_Weights.DEFAULT
        model = efficientnet_v2_s(weights=weights)

        # Initially prevent gradient descent due to pre-training
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # print(model) -- [0]: Dropout, [1]: Linear
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "alexnet":
        # Best available pre-trained weights from ImageNet
        weights = AlexNet_Weights.DEFAULT
        model = alexnet(weights=weights)

        # Initially prevent gradient descent due to pre-training
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # print(model) -- [0]: Dropout, [1]: Linear, [2]: ReLU, [3]: Dropout, [4]: Linear, [5]: ReLU, [6]: Linear
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    elif model_name == "vgg16":
        # Best available pre-trained weights from ImageNet
        weights = VGG16_Weights.DEFAULT
        model = vgg16(weights=weights)

        # Initially prevent gradient descent due to pre-training
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # print(model) -- [0]: Linear, [1]: ReLU, [2]: Dropout, [3]: Linear, [4]: ReLU, [5]: Dropout, [6]: Linear
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def get_last_conv_layer(model, model_name):
    if model_name == "baseline_cnn":
        # last Conv2d layer
        return model.features[12]

    if model_name == "convnext_tiny":
        # last block of the last stage
        return model.features[-1][-1]

    if model_name == "efficientnet_v2_s":
        # last major feature stage before classifier
        return model.features[-2]

    if model_name == "alexnet":
        # last Conv2d layer
        return model.features[10]

    if model_name == "vgg16":
        # last Conv2d layer
        return model.features[28]

    raise ValueError(f"Unsupported model: {model_name}")


def save_checkpoint(model, class_names, model_name, filename="model.pth"):
    save_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, filename)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "model_name": model_name
    }, save_path)

    return save_path


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    model_name = checkpoint["model_name"]

    model = build_model(model_name, num_classes=len(class_names), freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, model_name
