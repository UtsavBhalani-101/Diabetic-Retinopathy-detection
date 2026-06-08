# pipeline/dataset.py
# ============================================================

import logging
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ^ -------------------------- Ben Graham preprocessing --------------------------

# In dataset.py — a custom transform class
# class BenGrahamPreprocess:
#     """Circle crop + Gaussian color subtraction (Ben Graham, 2015)."""
#     def __init__(self, sigmaX=10):
#         self.sigmaX = sigmaX

#     def __call__(self, img):
#         # img is a PIL Image — convert to numpy, apply, convert back
#         img = np.array(img)
#         img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), self.sigmaX), -4, 128)
#         return Image.fromarray(img)


# ^ -------------------------- CLAHE preprocessing --------------------------

class CLAHEPreprocess:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel of LAB color space."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        # img is a PIL Image — convert to numpy, apply, convert back
        img = np.array(img)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Re-create CLAHE object inside __call__ to avoid multiprocessing pickle issues
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l2 = clahe.apply(l)
        
        lab2 = cv2.merge((l2, a, b))
        img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img2)


# ^ -------------------------- transforms --------------------------

val_transformer = transforms.Compose([
    CLAHEPreprocess(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_transformer = transforms.Compose([
    CLAHEPreprocess(clip_limit=2.0, tile_grid_size=(8, 8)),
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
        
        if num_samples and len(self.df) > num_samples:
            try:
                from sklearn.model_selection import train_test_split
                _, sample_df = train_test_split(
                    self.df,
                    test_size=num_samples,
                    random_state=42,
                    stratify=self.df[self.label_col]
                )
                self.df = sample_df.reset_index(drop=True)
                logger.debug(f"RetinopathyDataset: stratified sampled {num_samples} rows from CSV")
            except ValueError:
                # Fallback to random sample if stratify fails (e.g., if a class has fewer than 2 samples)
                self.df = self.df.sample(n=num_samples, random_state=42).reset_index(drop=True)
                logger.debug(f"RetinopathyDataset: randomly sampled {num_samples} rows from CSV")

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

