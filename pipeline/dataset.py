# pipeline/dataset.py
# ============================================================
# Everything related to loading images:
#   - GPU setup + reproducibility helpers
#   - train / val transforms
#   - RetinopathyDataset      (CSV-backed, used for test-time)
#   - RetinopathyDatasetFromDF (DF-backed,  used for train/val split)
# ============================================================

import logging
import random
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

logger = logging.getLogger(__name__)


# ^ ------------------------------- setting gpu ----------------------------------

def setting_gpu() -> torch.device:
    """Detect GPU, log name, and return a torch.device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device selected: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU found — running on CPU (will be slow for training)")
    return device


# ^ ------------------------------- reproducibility ----------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.debug(f"Random seed fixed: {seed}")


# Worker-level seed fn — passed as worker_init_fn to DataLoader
def seed_worker(worker_id: int) -> None:  # noqa: ARG001
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Shared Generator for DataLoader reproducibility
g = torch.Generator()
g.manual_seed(42)


# ^ -------------------------- transforms --------------------------

val_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(360),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

logger.debug("Train and val transforms defined")


# ^ -------------------------- dataset classes --------------------------

class RetinopathyDataset(Dataset):
    """
    CSV-backed dataset.
    Used for external / test datasets where the full CSV is consumed as-is.
    """

    def __init__(self, input_path, target_path, image_col, diagnosis_col,
                 transforms, extension, num_samples=None):
        self.img_path      = input_path
        self.extension     = extension
        self.image_col     = image_col
        self.diagnosis_col = diagnosis_col
        self.df = pd.read_csv(target_path).reset_index(drop=True)
        if num_samples is not None:
            self.df = self.df.sample(n=num_samples, random_state=42).reset_index(drop=True)
            logger.debug(f"RetinopathyDataset: sampled {num_samples} rows from CSV")
        self.transform = transforms
        logger.info(
            f"RetinopathyDataset | path={target_path} | rows={len(self.df)} | ext={extension}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        img_id = row[self.image_col]
        label  = row[self.diagnosis_col]
        img_path = os.path.join(self.img_path, f"{img_id}.{self.extension}")
        image  = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class RetinopathyDatasetFromDF(Dataset):
    """
    DataFrame-backed dataset.
    Used for APTOS train/val splits: the caller does the stratified split
    and passes each sub-DataFrame here, avoiding the need to write new CSVs.
    """

    def __init__(self, df, image_path, image_col, diagnosis_col,
                 transform, extension):
        self.df            = df.reset_index(drop=True)
        self.img_path      = image_path
        self.image_col     = image_col
        self.diagnosis_col = diagnosis_col
        self.transform     = transform
        self.extension     = extension
        logger.info(
            f"RetinopathyDatasetFromDF | rows={len(self.df)} | ext={extension}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        img_id = row[self.image_col]
        label  = int(row[self.diagnosis_col])
        img_file = os.path.join(self.img_path, f"{img_id}.{self.extension}")
        image  = Image.open(img_file).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
