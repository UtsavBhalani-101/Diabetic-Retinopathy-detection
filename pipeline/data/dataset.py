# pipeline/dataset.py
# ============================================================
# Everything related to loading images:
#   - GPU setup + reproducibility helpers
#   - train / val transforms
#   - RetinopathyDataset      (CSV-backed, used for test-time)
#   - RetinopathyDatasetFromDF (DF-backed,  used for train/val split)
# ============================================================

import logging
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

logger = logging.getLogger(__name__)



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
