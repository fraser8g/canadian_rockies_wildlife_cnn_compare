# CNN functions (PyTorch)
# https://docs.pytorch.org/docs/stable/index.html
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS


def get_transforms():
    train_transform = transforms.Compose([
        # Data augmentation to create different training versions
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),

        # Random adjustments to handle trailcam and snowy backgrounds
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

        # Convert (H, W, C) to (C, H, W) in order to normalize  
        transforms.ToTensor(),

        # ImageNet RGB mean and standard deviation
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),

        # Convert (H, W, C) to (C, H, W) in order to normalize  
        transforms.ToTensor(),

        # ImageNet RGB mean and standard deviation
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def get_dataloaders():
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, #Shuffle for training
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names
