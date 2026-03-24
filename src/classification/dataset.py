from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def build_train_transforms(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transforms(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_imagefolder_dataset(root_dir: str | Path, train: bool, image_size: int = 224):
    transform = build_train_transforms(image_size) if train else build_eval_transforms(image_size)
    return datasets.ImageFolder(root=str(root_dir), transform=transform)


class SingleImageDataset(Dataset):
    def __init__(self, image_path: str | Path, image_size: int = 224):
        self.image_path = Path(image_path)
        self.transform = build_eval_transforms(image_size)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        image = Image.open(self.image_path).convert("RGB")
        return self.transform(image), 0

