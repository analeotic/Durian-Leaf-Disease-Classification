import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class DurianLeafDataset(Dataset):
    """Dataset class for Durian Leaf Disease Classification"""

    def __init__(
        self,
        image_dir: str,
        csv_file: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        transform: Optional[A.Compose] = None,
        is_test: bool = False
    ):
        """
        Args:
            image_dir: Directory containing the images
            csv_file: Path to CSV file with annotations (optional if df provided)
            df: DataFrame with annotations (optional if csv_file provided)
            transform: Albumentations transform to apply
            is_test: Whether this is test data (no labels)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

        if df is not None:
            self.df = df
        elif csv_file is not None:
            self.df = pd.read_csv(csv_file)
        else:
            raise ValueError("Either csv_file or df must be provided")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_name = row['id']
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        if self.is_test:
            return image, img_name

        label = int(row['predict']) if pd.notna(row['predict']) else -1
        return image, label


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Get training augmentation transforms"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=30,
            p=0.5
        ),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1
            ),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MedianBlur(blur_limit=5, p=1),
        ], p=0.3),
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            p=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_valid_transforms(image_size: int = 224) -> A.Compose:
    """Get validation/test transforms"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def create_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    image_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""

    train_dataset = DurianLeafDataset(
        image_dir=image_dir,
        df=train_df,
        transform=get_train_transforms(image_size),
        is_test=False
    )

    valid_dataset = DurianLeafDataset(
        image_dir=image_dir,
        df=valid_df,
        transform=get_valid_transforms(image_size),
        is_test=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader
