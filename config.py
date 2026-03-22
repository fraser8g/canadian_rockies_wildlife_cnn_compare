# CNN functions (PyTorch)
# https://docs.pytorch.org/docs/stable/index.html
import torch

# CNN Models
MODELS = [
    "baseline_cnn",
    "convnext_tiny",
    "efficientnet_v2_s",
    "alexnet",
    "vgg16",
]

# Datasets
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

OUTPUT_DIR = "outputs"
IMAGE_SIZE = 224


# Train off GPU instead of CPU when Geforce cuda is available (NVIDIA Developer)
# https://developer.nvidia.com/cuda
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Pin CPU memory to speed up data transfer from RAM to GPU during training
PIN_MEMORY = torch.cuda.is_available()


# ---Hyper parameters---
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 50

# Phase 1: classifier head only
PHASE1_EPOCHS = 40
PHASE1_LR = 5e-4 #0.001

# Phase 2: fine-tuning
PHASE2_EPOCHS = 10
PHASE2_LR = 1e-4 #0.0001

# Early stopping when no improvements
PATIENCE = 6

# Loss settings
LABEL_SMOOTHING = 0.05 #0.1