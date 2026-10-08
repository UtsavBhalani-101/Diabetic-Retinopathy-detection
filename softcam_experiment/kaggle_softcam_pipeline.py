"""Self-contained SoftCAM pipeline designed for direct execution in a Kaggle notebook.

Includes:
  1. Hardcoded Kaggle paths for APTOS 2019 and IDRiD
  2. SoftCAMEfficientNet architecture and SoftCAMLoss (L1 Lasso on evidence maps)
  3. Stratified training with class-balanced cross-entropy
  4. Temperature calibration
  5. Single-pass IDRiD heatmap and overlay generation
  6. Targeted vs. Random Occlusion sensitivity benchmark (Baseline vs 10% vs 30% x 3 seeds)
  7. Final results summary table
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# ==============================================================================
# 1. HARDCODED KAGGLE PATHS & CONFIG
# ==============================================================================
CONFIG = {
    # APTOS 2019 training paths
    "aptos_csv": "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "aptos_img_dir": "/kaggle/input/aptos2019-blindness-detection/train_images",
    "aptos_img_col": "id_code",
    "aptos_label_col": "diagnosis",
    "aptos_ext": ".png",

    # IDRiD official test split paths
    "idrid_csv": "/kaggle/input/idrid-testing-dataset/IDRiD/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
    "idrid_img_dir": "/kaggle/input/idrid-testing-dataset/IDRiD/B. Disease Grading/1. Original Images/b. Testing Set",
    "idrid_img_col": "Image name",
    "idrid_label_col": "Retinopathy grade",
    "idrid_ext": ".jpg",

    # Alternate fallback Kaggle paths
    "alt_aptos_csv": "/kaggle/input/competitions/aptos2019-blindness-detection/train.csv",
    "alt_aptos_img_dir": "/kaggle/input/competitions/aptos2019-blindness-detection/train_images",
    "alt_idrid_csv": "/kaggle/input/datasets/antiti/idrid-testing-dataset/IDRiD/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
    "alt_idrid_img_dir": "/kaggle/input/datasets/antiti/idrid-testing-dataset/IDRiD/B. Disease Grading/1. Original Images/b. Testing Set",

    # Working outputs
    "output_dir": "/kaggle/working/artifacts",
    "weights_path": "/kaggle/working/artifacts/weights/aptos_softcam.pth",
    "temp_path": "/kaggle/working/artifacts/calibration/optimal_T_softcam.npy",
    "heatmap_dir": "/kaggle/working/artifacts/softcam_heatmaps/idrid/test",
    "overlay_dir": "/kaggle/working/artifacts/softcam_overlays/idrid/test",
    "results_json": "/kaggle/working/artifacts/softcam_occlusion/results.json",
    "summary_csv": "/kaggle/working/artifacts/softcam_occlusion/summary.csv",

    # Hyperparameters
    "epochs": 10,
    "batch_size": 64,  # Scaled for 2x T4 GPUs (32 per GPU)
    "num_workers": 4,  # Utilize all 4 Kaggle vCPUs
    "lr": 1e-4,
    "lambda1": 1e-4,   # Lasso sparsity on evidence maps
    "lambda2": 0.0,    # Ridge smoothness
    "dropout_rate": 0.3,
    "num_classes": 5,
    "seed": 42,
    "top_k": [10.0, 30.0],
    "random_seeds": [42, 43, 44],
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def resolve_path(primary: str, fallback: str) -> str:
    if os.path.exists(primary):
        return primary
    if os.path.exists(fallback):
        return fallback
    return primary


# ==============================================================================
# 2. DATASETS & TRANSFORMS
# ==============================================================================
def gpu_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(x.device).view(1, 3, 1, 1)
    std = IMAGENET_STD.to(x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class RetinopathyDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        img_dir: str,
        img_col: str,
        label_col: str,
        ext: str = "",
        is_train: bool = False,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_col = img_col
        self.label_col = label_col
        self.ext = ext
        self.is_train = is_train

        # Pre-extract filenames and labels to plain Python lists
        # Avoids costly pandas df.iloc[idx] indexing overhead during data loading
        self.filenames = [
            str(f) + (self.ext if self.ext and not str(f).endswith(self.ext) else "")
            for f in self.df[self.img_col]
        ]
        self.labels = self.df[self.label_col].astype(int).tolist()

        if is_train:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(20),
                T.ToTensor(),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        label = self.labels[idx]
        return tensor, label


class HeatmapOccludedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, heatmap_dir: str, top_k_percent: float):
        self.base_dataset = base_dataset
        self.heatmap_dir = heatmap_dir
        self.top_k_percent = top_k_percent
        self._percentile = 100.0 - top_k_percent

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]
        npy_path = os.path.join(self.heatmap_dir, f"{idx}.npy")
        heatmap = np.load(npy_path).astype(np.float32)
        threshold = float(np.percentile(heatmap, self._percentile))
        mask = torch.from_numpy(heatmap >= threshold)

        image = image.clone()
        fill = IMAGENET_MEAN.to(image.device).view(3, 1)
        image[:, mask] = fill
        return image, label


class RandomMeanOccludedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, top_k_percent: float, base_seed: int = 42):
        self.base_dataset = base_dataset
        self.top_k_percent = top_k_percent
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]
        _, height, width = image.shape
        n_pixels = height * width
        n_occlude = int(round(n_pixels * self.top_k_percent / 100.0))

        rng = np.random.default_rng(seed=self.base_seed + idx)
        flat_indices = rng.choice(n_pixels, size=n_occlude, replace=False)
        rows = flat_indices // width
        cols = flat_indices % width

        mask = torch.zeros((height, width), dtype=torch.bool)
        mask[rows, cols] = True

        image = image.clone()
        fill = IMAGENET_MEAN.to(image.device).view(3, 1)
        image[:, mask] = fill
        return image, label


# ==============================================================================
# 3. SOFTCAM MODEL & ELASTICNET LOSS
# ==============================================================================
class SoftCAMEfficientNet(nn.Module):
    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.base = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        in_channels = self.base.num_features
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.evidence_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, bias=True)

    def forward_maps(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base.forward_features(x)
        features = self.dropout(features)
        return self.evidence_conv(features)

    def forward(self, x: torch.Tensor, return_maps: bool = False):
        class_maps = self.forward_maps(x)
        logits = class_maps.mean(dim=(2, 3))
        if return_maps:
            return logits, class_maps
        return logits

    @torch.no_grad()
    def heatmaps(self, x: torch.Tensor, class_indices: torch.Tensor | None = None, output_size=(224, 224)):
        self.eval()
        class_maps = self.forward_maps(x)
        logits = class_maps.mean(dim=(2, 3))

        if class_indices is None:
            class_indices = logits.argmax(dim=1)

        batch_idx = torch.arange(x.shape[0], device=x.device)
        selected_maps = F.relu(class_maps[batch_idx, class_indices])
        selected_maps = F.interpolate(
            selected_maps.unsqueeze(1), size=output_size, mode="bilinear", align_corners=False
        ).squeeze(1)

        flat = selected_maps.flatten(start_dim=1)
        mins = flat.min(dim=1).values.view(-1, 1, 1)
        maxs = flat.max(dim=1).values.view(-1, 1, 1)
        heatmaps = (selected_maps - mins) / (maxs - mins + 1e-8)
        return heatmaps, logits, class_indices


class SoftCAMLoss(nn.Module):
    def __init__(self, base_criterion: nn.Module, lambda1: float = 1e-4, lambda2: float = 0.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, logits, targets, class_maps):
        ce_loss = self.base_criterion(logits, targets)
        l1_reg = class_maps.abs().mean()
        l2_reg = (class_maps**2).mean()
        total_loss = ce_loss + (self.lambda1 * l1_reg) + (self.lambda2 * l2_reg)
        return total_loss, ce_loss, l1_reg, l2_reg


# ==============================================================================
# 4. TRAINING & CALIBRATION FUNCTIONS
# ==============================================================================
def train_softcam(device: torch.device):
    print("--- 1. PREPARING DATA ---")
    csv_path = resolve_path(CONFIG["aptos_csv"], CONFIG["alt_aptos_csv"])
    img_dir = resolve_path(CONFIG["aptos_img_dir"], CONFIG["alt_aptos_img_dir"])

    df = pd.read_csv(csv_path)
    print(f"APTOS loaded: {len(df)} samples from {csv_path}")

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=CONFIG["seed"], stratify=df[CONFIG["aptos_label_col"]]
    )

    train_ds = RetinopathyDataset(
        train_df, img_dir, CONFIG["aptos_img_col"], CONFIG["aptos_label_col"], CONFIG["aptos_ext"], is_train=True
    )
    val_ds = RetinopathyDataset(
        val_df, img_dir, CONFIG["aptos_img_col"], CONFIG["aptos_label_col"], CONFIG["aptos_ext"], is_train=False
    )

    num_workers = CONFIG.get("num_workers", 4)
    loader_kwargs = dict(
        batch_size=CONFIG["batch_size"],
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Class-balanced loss
    labels = train_df[CONFIG["aptos_label_col"]].values
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weights = torch.FloatTensor(weights).to(device)
    base_crit = nn.CrossEntropyLoss(weight=class_weights)
    loss_fn = SoftCAMLoss(base_crit, lambda1=CONFIG["lambda1"], lambda2=CONFIG["lambda2"])

    # Model - SINGLE GPU (DataParallel on T4 is slower due to GIL overhead)
    model = SoftCAMEfficientNet(
        num_classes=CONFIG["num_classes"], dropout_rate=CONFIG["dropout_rate"], pretrained=True
    ).to(device)

    # cuDNN auto-tuner for fixed 224x224 tensor shapes
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Mixed precision AMP scaler for Turing Tensor Cores
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=5e-4)

    print(f"\n--- 2. TRAINING FOR {CONFIG['epochs']} EPOCHS (L1={CONFIG['lambda1']}, AMP=True, Single GPU) ---")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss, train_ce, train_l1 = 0.0, 0.0, 0.0
        for images, targets in train_loader:
            images = gpu_normalize(images.to(device, non_blocking=True))
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, maps = model(images, return_maps=True)
                total_loss, ce, l1, _ = loss_fn(logits, targets, maps)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()
            train_ce += ce.item()
            train_l1 += l1.item()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = gpu_normalize(images.to(device, non_blocking=True))
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(images)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_targets.extend(targets.numpy())

        qwk = cohen_kappa_score(val_targets, val_preds, weights="quadratic")
        n_batches = len(train_loader)
        print(
            f"Epoch {epoch+1:02d}/{CONFIG['epochs']:02d} | "
            f"Train Loss: {train_loss/n_batches:.4f} (CE: {train_ce/n_batches:.4f}, L1: {train_l1/n_batches:.4f}) | "
            f"Val QWK: {qwk:.4f}"
        )

    # Temperature Calibration
    print("\n--- 3. TEMPERATURE CALIBRATION ---")
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            images = gpu_normalize(images.to(device, non_blocking=True))
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)
            all_logits.append(logits.float().cpu().numpy())
            all_labels.extend(targets.numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.array(all_labels)

    # Grid search optimal T
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 3.0, 51):
        scaled = all_logits / t
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        nll = -np.log(probs[np.arange(len(all_labels)), all_labels] + 1e-12).mean()
        if nll < best_nll:
            best_nll = nll
            best_t = t
    print(f"Optimal Temperature T = {best_t:.4f}")

    # Save artifacts
    os.makedirs(os.path.dirname(CONFIG["weights_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["temp_path"]), exist_ok=True)
    torch.save(model.state_dict(), CONFIG["weights_path"])
    np.save(CONFIG["temp_path"], np.array(best_t))
    print(f"Saved weights -> {CONFIG['weights_path']}")
    print(f"Saved temperature -> {CONFIG['temp_path']}")

    return model, best_t


# ==============================================================================
# 5. HEATMAP GENERATION & OCCLUSION BENCHMARK
# ==============================================================================
def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    h = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def generate_heatmaps(model: SoftCAMEfficientNet, idrid_ds: RetinopathyDataset, device: torch.device):
    print("\n--- 4. GENERATING IDRiD HEATMAPS IN A SINGLE FORWARD PASS ---")
    os.makedirs(CONFIG["heatmap_dir"], exist_ok=True)
    os.makedirs(CONFIG["overlay_dir"], exist_ok=True)

    workers = CONFIG.get("num_workers", 4)
    loader = DataLoader(
        idrid_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=4,
    )
    offset = 0
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = gpu_normalize(images.to(device, non_blocking=True))
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                heatmaps, logits, selected_classes = model.heatmaps(images, output_size=(224, 224))

            for i in range(images.shape[0]):
                idx = offset + i
                h_np = heatmaps[i].float().cpu().numpy().astype(np.float32)
                np.save(os.path.join(CONFIG["heatmap_dir"], f"{idx}.npy"), h_np)

                if idx < 12:  # Save first 12 visual overlays
                    img_np = np.clip(images[i].cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)
                    h_rgb = colorize_heatmap(h_np)
                    overlay = np.clip(0.45 * img_np + 0.55 * h_rgb, 0.0, 1.0)
                    Image.fromarray((overlay * 255).astype(np.uint8)).save(
                        os.path.join(CONFIG["overlay_dir"], f"{idx}.png")
                    )
            offset += images.shape[0]

    print(f"Generated {offset} heatmaps in {CONFIG['heatmap_dir']}")


def evaluate_dataset(model: nn.Module, dataset: Dataset, device: torch.device, temp: float) -> dict:
    workers = CONFIG.get("num_workers", 4)
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=4,
    )
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = gpu_normalize(images.to(device, non_blocking=True))
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)
            all_logits.append(logits.float().cpu().numpy())
            all_labels.extend(labels.numpy())

    logits = np.concatenate(all_logits, axis=0) / temp
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    labels = np.array(all_labels)

    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    acc = float((preds == labels).mean())
    conf = float(probs.max(axis=1).mean())
    true_probs = probs[np.arange(len(labels)), labels]

    return {"qwk": qwk, "acc": acc, "conf": conf, "true_probs": true_probs}


def run_occlusion_experiment(model: SoftCAMEfficientNet, idrid_ds: RetinopathyDataset, device: torch.device, temp: float):
    print("\n--- 5. RUNNING TARGETED VS. RANDOM OCCLUSION BENCHMARK ---")
    base_res = evaluate_dataset(model, idrid_ds, device, temp)
    base_probs = base_res["true_probs"]

    rows = [{
        "condition": "Baseline",
        "top_k": 0.0,
        "seed": "-",
        "qwk": base_res["qwk"],
        "acc": base_res["acc"],
        "conf": base_res["conf"],
        "mean_drop": 0.0,
        "pos_drop_pct": 0.0,
    }]

    for k in CONFIG["top_k"]:
        # Targeted SoftCAM
        t_ds = HeatmapOccludedDataset(idrid_ds, CONFIG["heatmap_dir"], top_k_percent=k)
        t_res = evaluate_dataset(model, t_ds, device, temp)
        t_drops = base_probs - t_res["true_probs"]
        rows.append({
            "condition": "SoftCAM Targeted",
            "top_k": k,
            "seed": "-",
            "qwk": t_res["qwk"],
            "acc": t_res["acc"],
            "conf": t_res["conf"],
            "mean_drop": float(t_drops.mean()),
            "pos_drop_pct": float((t_drops > 0).mean() * 100),
        })

        # Random Controls
        for s in CONFIG["random_seeds"]:
            r_ds = RandomMeanOccludedDataset(idrid_ds, top_k_percent=k, base_seed=s)
            r_res = evaluate_dataset(model, r_ds, device, temp)
            r_drops = base_probs - r_res["true_probs"]
            rows.append({
                "condition": "Random Control",
                "top_k": k,
                "seed": s,
                "qwk": r_res["qwk"],
                "acc": r_res["acc"],
                "conf": r_res["conf"],
                "mean_drop": float(r_drops.mean()),
                "pos_drop_pct": float((r_drops > 0).mean() * 100),
            })

    # Summary
    df_res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CONFIG["summary_csv"]), exist_ok=True)
    df_res.to_csv(CONFIG["summary_csv"], index=False)
    print("\n================ FINAL OCCLUSION BENCHMARK RESULTS ================")
    print(df_res.to_string(index=False))
    print(f"\nSaved summary -> {CONFIG['summary_csv']}")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    # Force single GPU mode - DataParallel on T4 is slower due to GIL overhead
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device} | GPUs visible: {n_gpus} (forced single GPU mode)")
    if n_gpus > 0:
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("  -> Single GPU + AMP + batch_size=64 + 4 workers = optimal T4 throughput")

    # Step 1: Train SoftCAM
    model, optimal_t = train_softcam(device)

    # Step 2: Load IDRiD test dataset
    idrid_csv = resolve_path(CONFIG["idrid_csv"], CONFIG["alt_idrid_csv"])
    idrid_img_dir = resolve_path(CONFIG["idrid_img_dir"], CONFIG["alt_idrid_img_dir"])
    idrid_df = pd.read_csv(idrid_csv)
    print(f"\nIDRiD test set loaded: {len(idrid_df)} images from {idrid_csv}")
    idrid_ds = RetinopathyDataset(
        idrid_df, idrid_img_dir, CONFIG["idrid_img_col"], CONFIG["idrid_label_col"], CONFIG["idrid_ext"], is_train=False
    )

    # Step 3: Generate Heatmaps
    generate_heatmaps(model, idrid_ds, device)

    # Step 4: Run Occlusion Test
    run_occlusion_experiment(model, idrid_ds, device, optimal_t)
