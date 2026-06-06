# pipeline/dataset.py
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
    def __init__(self, 
                 img_path, 
                 img_col, 
                 label_col, 
                 transforms, 
                 extension, 
                 num_samples=None, 
                 dataframe=None, 
                 target_path=None):


        if (target_path is None) == (dataframe is None):
            raise ValueError("Provide exactly one, either target_path OR dataframe")

        self.img_path = img_path
        self.img_col = img_col
        self.label_col = label_col
        self.extension = extension
        self.transforms = transforms
        
        if target_path is None:
            self.df = dataframe.copy().reset_index(drop=True)
        else:
            self.df = pd.read_csv(target_path, usecols=[self.img_col, self.label_col]).reset_index(drop=True)
        
        if num_samples:
            self.df = self.df.head(num_samples)
            logger.debug(f"RetinopathyDataset: sampled {num_samples} rows from CSV")

        logger.info(
            f"RetinopathyDataset | rows={len(self.df)} | ext={extension}"
        )        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        image_id = row[self.img_col]
        label = row[self.label_col]
        image_path = os.path.join(self.img_path, f"{image_id}{self.extension}")
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        
        return image, label

