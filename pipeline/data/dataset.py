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

# ^ -------------------------- CLAHE preprocessing --------------------------

class CLAHEPreprocess:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to the L channel of LAB colour space.

    Used as a fallback transform when offline-preprocessed images are not
    available. When preprocess_clahe.py has been run, RetinopathyDataset
    redirects img_path to the preprocessed directory and CLAHE is skipped
    entirely at training time — this class is then never called.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit     = clip_limit
        self.tile_grid_size = tile_grid_size
        # Create the CLAHE object ONCE here, not inside __call__.
        # The old comment "Re-create CLAHE inside __call__ to avoid multiprocessing
        # pickle issues" was incorrect — cv2.CLAHE is picklable in modern OpenCV
        # (≥4.x) and creating it every call added unnecessary allocation overhead.
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )

    def __call__(self, img):
        # img is a PIL Image — convert to numpy, apply CLAHE, convert back
        img  = np.array(img)
        lab  = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l2   = self._clahe.apply(l)   # reuse pre-created object
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
                 target_path=None,
                 clahe_image_path=None):
        """
        Parameters
        ----------
        img_path          : str  — path to the raw image directory
        img_col           : str  — CSV column containing image IDs / filenames
        label_col         : str  — CSV column containing class labels
        transforms        : torchvision.transforms.Compose
        extension         : str  — file extension including the dot, e.g. ".png"
        num_samples       : int  — if set, subsample the dataset to this many rows
        dataframe         : pd.DataFrame  — pass a pre-split DataFrame instead of target_path
        target_path       : str  — path to the CSV (mutually exclusive with dataframe)
        clahe_image_path  : str or None
            If provided AND the directory exists on disk, img_path is silently
            replaced by this path so the DataLoader reads offline-preprocessed
            images instead of raw ones. The train/val transforms then skip
            the CLAHE step because it has already been applied offline.
            Populated automatically from DATASET_REGISTRY['clahe_image_path'].
        """
        if (target_path is None) == (dataframe is None):
            raise ValueError("Provide exactly one: target_path OR dataframe")

        # ── Auto-detect offline CLAHE directory ──────────────────────────────
        # If preprocess_clahe.py has been run, clahe_image_path will exist.
        # We swap img_path silently so __getitem__ reads the fast pre-processed
        # images without any changes needed at the call site.
        if clahe_image_path is not None and os.path.isdir(clahe_image_path):
            self.img_path = clahe_image_path
            logger.info(
                f"RetinopathyDataset | CLAHE-offline mode: reading from "
                f"{clahe_image_path}"
            )
        else:
            self.img_path = img_path
            if clahe_image_path is not None:
                # Path provided but doesn't exist yet — warn so it's obvious
                logger.warning(
                    f"RetinopathyDataset | clahe_image_path not found: "
                    f"{clahe_image_path}  → falling back to on-the-fly CLAHE"
                )
            else:
                logger.debug(
                    "RetinopathyDataset | No clahe_image_path provided "
                    "→ on-the-fly CLAHE mode (slower)"
                )

        self.img_col    = img_col
        self.label_col  = label_col
        self.extension  = extension
        self.transforms = transforms

        if target_path is None:
            self.df = dataframe.copy().reset_index(drop=True)
        else:
            self.df = pd.read_csv(
                target_path, usecols=[self.img_col, self.label_col]
            ).reset_index(drop=True)

        if num_samples and len(self.df) > num_samples:
            try:
                from sklearn.model_selection import train_test_split
                _, sample_df = train_test_split(
                    self.df,
                    test_size=num_samples,
                    random_state=42,
                    stratify=self.df[self.label_col],
                )
                self.df = sample_df.reset_index(drop=True)
                logger.debug(
                    f"RetinopathyDataset: stratified sampled {num_samples} rows from CSV"
                )
            except ValueError:
                # Fallback if a class has too few samples to stratify
                self.df = self.df.sample(
                    n=num_samples, random_state=42
                ).reset_index(drop=True)
                logger.debug(
                    f"RetinopathyDataset: randomly sampled {num_samples} rows from CSV"
                )

        # Pre-extract image IDs and labels as plain Python lists.
        # list[idx] is 3–5× faster than df.iloc[idx] and avoids pandas
        # indexing overhead across millions of __getitem__ calls per epoch.
        self.image_ids = self.df[self.img_col].tolist()
        self.labels    = self.df[self.label_col].tolist()

        logger.info(
            f"RetinopathyDataset | rows={len(self.image_ids)} | ext={extension} "
            f"| img_dir={self.img_path}"
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id   = self.image_ids[idx]    # O(1) Python list lookup
        label      = self.labels[idx]
        image_path = os.path.join(self.img_path, f"{image_id}{self.extension}")
        image      = Image.open(image_path).convert("RGB")
        image      = self.transforms(image)
        return image, label

