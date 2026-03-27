import os

# CNN functions (PyTorch)
# https://docs.pytorch.org/docs/stable/index.html
import torch
# https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
import torch.nn as nn

# Import CNN model for PyTorch and pre-trained ImageNet weights.
# https://docs.pytorch.org/vision/main/models
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

from config import OUTPUT_DIR, INCLUDED_PRETRAINING


class WildlifeCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        '''    
        # V1 - (Hybrid VGG and AlexNet)
        # 6 Layers, 3x3 Kernel, Normalization, Linear Activation, Pooling, Incremental Channel Increase from 3 to 512.

        self.features = nn.Sequential(

            # Block 1 - Basic Edges (Input: 224x224x3)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False), # (Output: 224x224x32)
            nn.BatchNorm2d(num_features=32), # Normalization stability
            nn.ReLU(inplace=True), # Non-linearity
            nn.MaxPool2d(kernel_size=2), # Reduce spatial size (Output: 112x112x32)

            # Block 2 - Texture Features (Input: 112x112x32)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False), # (Output: 112x112x64)
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # (Output: 56x56x64)

            # Block 3 - Shapes (Input: 56x56x64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False), # (Output: 56x56x128)
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            # Block 4 - Feature Refinement (Input: 56x56x128)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False), # (Output: 56x56x128)
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # (Output: 28x28x128)

            # Block 5 - Abstract Objects (Input: 28x28x128)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False), # (Output: 14x14x256)
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            # Block 6 - High-level Features (Input: 14x14x256)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False), # (Output: 14x14x512)
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # Reduce feature map (Output: 1x1x512)

        self.classifier = nn.Sequential(
            nn.Flatten(), # Class scoring
            nn.Linear(in_features=512, out_features=128), # Fully Connected Layer
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3), # Prevent overfitting
            nn.Linear(in_features=128, out_features=num_classes), # Final Fully Connected Layer (Scoring logits)
        )

        '''
        
        
        '''
        # V3 - A modified lightweight MobileNet example with additional inverted bottleneck expansion
        #      and bottleneck compression.
        # 8 layers, using LeakyReLU for continued gradient learning when 0 or negative activation.

        self.stem = nn.Sequential(
            # Input: 224x224x3
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False), # Output: 112x112x32
            nn.BatchNorm2d(num_features=32),
            nn.ReLU6(inplace=True),
        )

        self.features = nn.Sequential(

            # Block 1 - Inverted bottleneck expansion (Input: 112x112x32)
            nn.Conv2d(in_channels=32, out_channels=192, kernel_size=1, bias=False), # Output: 112x112x192
            nn.BatchNorm2d(num_features=192),
            nn.LeakyReLU(negative_slope=0.1, inplace=True), # Continued gradient learning for smaller datasets

            # Block 2 - Depthwise Convolution (Input: 112x112x192)
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, groups=192, bias=False), # Output: 112x112x192
            nn.BatchNorm2d(num_features=192),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            # Block 3 - Inverted bottleneck expansion (Input: 112x112x192)
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=1, bias=False), # Output: 112x112x256
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            # Block 4 - Depthwise Convolution (Input: 112x112x256)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, groups=256, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            
            # Block 5 - Bottleneck Compression (Input: 112x112x256)
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, bias=False), # Output: 112x112x128
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            
            # Block 6 - Bottleneck Compression (Input: 112x112x128) 
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False), # Output: 112x112x64
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            # Block 7 - Feature Refinement (Input: 112x112x64) 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), # Output: 112x112x64
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # Output: 1x1x64

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3), #0.1
            nn.Linear(in_features=64, out_features=num_classes), # Scoring logits
        )
        '''

        # V2 - A lightweight MobileNet example with reduced parameters
        # 4 Layers, Varied Kernel for Depthwise convolution and bottleneck compression.

        self.stem = nn.Sequential(
            # Input: 224x224x3
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False), # Output: 112x112x32
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.features = nn.Sequential(

            # Block 1 - Expand 1x1 (Input: 112x112x32)
            nn.Conv2d(in_channels=32, out_channels=192, kernel_size=1, bias=False), # Output: 112x112x192
            nn.BatchNorm2d(num_features=192),
            nn.ReLU6(inplace=True), # lightweight stabilizer that caps at 6

            # Block 2 - Depthwise Convolution 3x3 (Input: 112x112x192)
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, groups=192, bias=False), # Each channel gets its own filter (Output: 112x112x192)
            nn.BatchNorm2d(num_features=192),
            nn.Conv2d(in_channels=192, out_channels=224, kernel_size=1, bias=False), # Output: 112x112x192
            nn.ReLU6(inplace=True),

            # Block 2 - Depthwise Convolution 3x3 (Input: 112x112x192)
            nn.Conv2d(in_channels=224, out_channels=224, kernel_size=3, padding=1, groups=224, bias=False), # Each channel gets its own filter (Output: 112x112x224)
            nn.BatchNorm2d(num_features=224),
            nn.ReLU6(inplace=True),

            # Block 3 - Bottleneck Compression (Input: 112x112x224)
            nn.Conv2d(in_channels=224, out_channels=192, kernel_size=1, bias=False),  # Output: 112x112x192
            nn.BatchNorm2d(num_features=192),
            nn.ReLU6(inplace=True),

            # Block 3 - Bottleneck Compression (Input: 112x112x192)
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, bias=False),  # Output: 112x112x64
            nn.BatchNorm2d(num_features=64),
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # Output: 1x1x64

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.1), #0.3
            nn.Linear(in_features=64, out_features=num_classes), # Scoring logits
        )

    def forward(self, x):
        x = self.stem(x) # V2 and V3 only
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
        

def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False

  
def build_model(model_name, num_classes, freeze_backbone=True):
    if model_name == "wildlife_cnn":
        model = WildlifeCNN(num_classes)

    # Documentation: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html
    # Samples: https://github.com/pytorch/vision/tree/main/references/classification#convnext
    elif model_name == "convnext_tiny":
        if(INCLUDED_PRETRAINING == 1):
            # Best available pre-trained weights from ImageNet
            weights = ConvNeXt_Tiny_Weights.DEFAULT 
            model = convnext_tiny(weights=weights)

            # Initially prevent gradient descent due to pre-training
            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False
        else:
           model = convnext_tiny() 

        # print(model) -- [0]: LayerNorm, [1]: Flatten, [2]: Linear 
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    # Documentation: https://docs.pytorch.org/vision/main/models/efficientnetv2.html
    # Samples: https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v2
    elif model_name == "efficientnet_v2_s":
        if(INCLUDED_PRETRAINING == 1):
            # Best available pre-trained weights from ImageNet
            weights = EfficientNet_V2_S_Weights.DEFAULT
            model = efficientnet_v2_s(weights=weights)

            # Initially prevent gradient descent due to pre-training
            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False
        
        else:
           model = efficientnet_v2_s()

        # print(model) -- [0]: Dropout, [1]: Linear
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    # Documentation: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html
    # Samples: https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg
    elif model_name == "alexnet":
        if(INCLUDED_PRETRAINING == 1):
            # Best available pre-trained weights from ImageNet
            weights = AlexNet_Weights.DEFAULT
            model = alexnet(weights=weights)

            # Initially prevent gradient descent due to pre-training
            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False
            
        else:
           model = alexnet()

        # print(model) -- [0]: Dropout, [1]: Linear, [2]: ReLU, [3]: Dropout, [4]: Linear, [5]: ReLU, [6]: Linear
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    # Documentation: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html
    # Samples: https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg
    elif model_name == "vgg16":
        if(INCLUDED_PRETRAINING == 1):
            # Best available pre-trained weights from ImageNet
            weights = VGG16_Weights.DEFAULT
            model = vgg16(weights=weights)

            # Initially prevent gradient descent due to pre-training
            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False

        else:
           model = vgg16()

        # print(model) -- [0]: Linear, [1]: ReLU, [2]: Dropout, [3]: Linear, [4]: ReLU, [5]: Dropout, [6]: Linear
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def get_last_conv_layer(model, model_name):
    if model_name == "wildlife_cnn":
        # last Conv2d layer
        return model.features[-1]

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
