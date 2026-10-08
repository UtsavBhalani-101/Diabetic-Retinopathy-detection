"""Self-contained SoftCAM pipeline for a Kaggle notebook (single T4 GPU).

Steps:
  1. Cache APTOS 2019 + IDRiD test images once (decoded + resized uint8 arrays)
  2. Train SoftCAM EfficientNet-B0 on APTOS (optional lambda1 sweep, best-QWK checkpoint)
  3. Temperature calibration on MC-dropout validation logits
  4. Single-pass IDRiD evidence maps, heatmaps and overlays
  5. Targeted vs. random occlusion benchmark with MC dropout
     - "patch" mode: rank patches by evidence, random control = same number of
       random patches inside the retina (fair comparison, like the paper)
     - "pixel" mode: same protocol as the old version (tie bug fixed),
       kept so numbers stay comparable with your other methods
  6. Summary CSV, per-image CSV, full JSON and paired Wilcoxon tests

Run in Kaggle:  !python kaggle_softcam_pipeline.py
"""

import os

# Use one GPU. Must be set before torch touches CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import csv
import json
import math
import random
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ==============================================================================
# 1. CONFIG
# ==============================================================================
CONFIG = {
    # APTOS 2019 training data
    "aptos_csv": "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "aptos_img_dir": "/kaggle/input/aptos2019-blindness-detection/train_images",
    "aptos_img_col": "id_code",
    "aptos_label_col": "diagnosis",
    "aptos_ext": ".png",

    # IDRiD official test split
    "idrid_csv": "/kaggle/input/idrid-testing-dataset/IDRiD/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
    "idrid_img_dir": "/kaggle/input/idrid-testing-dataset/IDRiD/B. Disease Grading/1. Original Images/b. Testing Set",
    "idrid_img_col": "Image name",
    "idrid_label_col": "Retinopathy grade",
    "idrid_ext": ".jpg",

    # Fallback Kaggle paths
    "alt_aptos_csv": "/kaggle/input/competitions/aptos2019-blindness-detection/train.csv",
    "alt_aptos_img_dir": "/kaggle/input/competitions/aptos2019-blindness-detection/train_images",
    "alt_idrid_csv": "/kaggle/input/datasets/antiti/idrid-testing-dataset/IDRiD/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
    "alt_idrid_img_dir": "/kaggle/input/datasets/antiti/idrid-testing-dataset/IDRiD/B. Disease Grading/1. Original Images/b. Testing Set",

    # Image cache. Built once in cache_dir. To skip rebuilding in later sessions,
    # save /kaggle/working/cache as a Kaggle dataset and add its path here.
    "cache_dir": "/kaggle/working/cache",
    "extra_cache_dirs": ["/kaggle/input/softcam-cache"],
    "num_preprocess_workers": 4,

    "output_dir": "/kaggle/working/artifacts",

    # Image settings. Keep 224 + no crop to match your other methods.
    # 384 gives a 12x12 evidence map instead of 7x7 (sharper explanations).
    "img_size": 224,
    "circle_crop": False,

    # Training
    "pretrained": True,
    "epochs": 15,
    "batch_size": 64,
    "lr": 1e-4,
    "weight_decay": 5e-4,
    "dropout_rate": 0.3,
    "num_classes": 5,
    "seed": 42,

    # ElasticNet on evidence maps (per-image SUM, like the paper and official repo).
    # Each lambda1 is trained; the largest one whose best val QWK is within
    # qwk_tolerance of the best run is kept (the paper's selection rule).
    # Set to a single value, e.g. [5e-5], to skip the sweep.
    "lambda1_sweep": [0.0, 5e-5, 2e-4, 1e-3],
    "lambda2": 0.0,
    "qwk_tolerance": 0.01,

    # MC dropout + occlusion
    "mc_passes": 30,
    "occlusion_modes": ["patch", "pixel"],
    "patch_size": 16,
    "top_k": [10.0, 30.0],
    "random_seeds": [42, 43, 44],
    "overlay_limit": 24,
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

# Uncertainty thresholds: use your project's values when the repo is importable,
# so the quadrant numbers match your other methods.
try:
    from pipeline.setup.config import (
        UNCERTAINTY_ENTROPY_THRESHOLD,
        UNCERTAINTY_MARGIN_THRESHOLD,
        UNCERTAINTY_MC_STD_THRESHOLD,
    )
    THRESHOLD_SOURCE = "pipeline.setup.config"
except ImportError:
    UNCERTAINTY_ENTROPY_THRESHOLD = 0.8
    UNCERTAINTY_MARGIN_THRESHOLD = 0.2
    UNCERTAINTY_MC_STD_THRESHOLD = 0.1
    THRESHOLD_SOURCE = "FALLBACK values in this file - replace with your project's values"


def resolve_path(primary, fallback):
    if os.path.exists(primary):
        return primary
    if os.path.exists(fallback):
        return fallback
    return primary


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 2. IMAGE CACHE (decode once, keep in RAM / GPU)
# ==============================================================================
def crop_to_fundus(img):
    """Crop to the bounding box of the non-black retina and pad to a square."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > 10)
    if len(ys) == 0:
        return img
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = img.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, 3), dtype=img.dtype)
    top = (side - h) // 2
    left = (side - w) // 2
    out[top:top + h, left:left + w] = img
    return out


def load_one_image(task):
    """Worker: read one image, optionally crop, resize to a square uint8 array."""
    path, size, circle_crop = task
    cv2.setNumThreads(0)
    if path.lower().endswith((".jpg", ".jpeg")):
        # JPEG can be decoded at half size directly, which is much faster
        img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if circle_crop:
        img = crop_to_fundus(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def load_or_build_cache(name, df, img_dir, img_col, label_col, ext):
    size = CONFIG["img_size"]
    crop = CONFIG["circle_crop"]
    tag = "crop" if crop else "resize"
    filename = f"{name}_{size}_{tag}.npz"

    search_dirs = [CONFIG["cache_dir"]] + list(CONFIG["extra_cache_dirs"])
    for folder in search_dirs:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            data = np.load(path, allow_pickle=False)
            print(f"[cache] loaded {name} from {path} ({len(data['labels'])} images)")
            return data["images"], data["labels"], data["names"]

    names = []
    for value in df[img_col]:
        file_name = str(value)
        if ext and not file_name.endswith(ext):
            file_name = file_name + ext
        names.append(file_name)
    labels = df[label_col].astype(int).to_numpy()
    tasks = [(os.path.join(img_dir, n), size, crop) for n in names]

    print(f"[cache] building {name}: {len(tasks)} images -> {size}x{size} ({tag}) ...")
    start = time.time()
    images = np.zeros((len(tasks), size, size, 3), dtype=np.uint8)
    with Pool(processes=CONFIG["num_preprocess_workers"]) as pool:
        for i, img in enumerate(pool.imap(load_one_image, tasks, chunksize=4)):
            images[i] = img
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)} done ({time.time() - start:.0f}s)")
    print(f"[cache] {name} built in {time.time() - start:.0f}s")

    os.makedirs(CONFIG["cache_dir"], exist_ok=True)
    out_path = os.path.join(CONFIG["cache_dir"], filename)
    np.savez(out_path, images=images, labels=labels, names=np.array(names))
    print(f"[cache] saved -> {out_path}")
    return images, labels, np.array(names)


def to_device_uint8(images_hwc, device):
    """[N,H,W,3] uint8 numpy -> [N,3,H,W] uint8 tensor on the GPU."""
    return torch.from_numpy(images_hwc).permute(0, 3, 1, 2).contiguous().to(device)


def prepare_batch(batch):
    """uint8 or [0,1] float batch -> ImageNet-normalized float batch."""
    if batch.dtype == torch.uint8:
        batch = batch.float() / 255.0
    mean = IMAGENET_MEAN.to(batch.device).view(1, 3, 1, 1)
    std = IMAGENET_STD.to(batch.device).view(1, 3, 1, 1)
    batch = (batch - mean) / std
    return batch.contiguous(memory_format=torch.channels_last)


def gpu_augment(x):
    """Random flips, rotation (+-20 deg), small zoom and brightness/contrast, on GPU.

    x: [B,3,H,W] float in [0,1].
    """
    b = x.shape[0]
    device = x.device

    flip_h = torch.rand(b, device=device) < 0.5
    x = torch.where(flip_h.view(-1, 1, 1, 1), x.flip(3), x)
    flip_v = torch.rand(b, device=device) < 0.5
    x = torch.where(flip_v.view(-1, 1, 1, 1), x.flip(2), x)

    angle = (torch.rand(b, device=device) * 2 - 1) * math.radians(20)
    scale = 1.0 + (torch.rand(b, device=device) * 2 - 1) * 0.1
    cos = torch.cos(angle) / scale
    sin = torch.sin(angle) / scale
    theta = torch.zeros(b, 2, 3, device=device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    brightness = 1.0 + (torch.rand(b, 1, 1, 1, device=device) * 2 - 1) * 0.2
    contrast = 1.0 + (torch.rand(b, 1, 1, 1, device=device) * 2 - 1) * 0.2
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    x = x * brightness
    return x.clamp(0.0, 1.0)


# ==============================================================================
# 3. SOFTCAM MODEL & ELASTICNET LOSS
# ==============================================================================
class SoftCAMEfficientNet(nn.Module):
    """EfficientNet-B0 backbone + 1x1 conv class-evidence layer (no GAP, no FC)."""

    def __init__(self, num_classes=5, dropout_rate=0.3, pretrained=True):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.base = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        in_channels = self.base.num_features  # 1280
        self.dropout = nn.Dropout(p=dropout_rate)
        self.evidence_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, bias=True)

    def forward_maps(self, x):
        features = self.base.forward_features(x)
        features = self.dropout(features)
        return self.evidence_conv(features)  # [B, C, h, w]

    def forward(self, x, return_maps=False):
        class_maps = self.forward_maps(x)
        logits = class_maps.mean(dim=(2, 3))  # spatial average pooling of evidence
        if return_maps:
            return logits, class_maps
        return logits


def softcam_loss(logits, targets, class_maps, criterion, lambda1, lambda2):
    """CE + lambda1 * sum|A| + lambda2 * ||A||_2, per image, averaged over the batch.

    Matches Eq. 4 of the paper and torch.norm(A, 1) / torch.norm(A, 2) in the
    official repo. (The old version used .mean() over all elements, which made
    the penalty ~245x weaker for 7x7x5 maps.)
    """
    ce = criterion(logits, targets)
    maps = class_maps.float().flatten(start_dim=1)
    l1 = maps.abs().sum(dim=1).mean()
    l2 = maps.norm(p=2, dim=1).mean()
    total = ce + lambda1 * l1 + lambda2 * l2
    return total, ce, l1, l2


def check_backbone_is_deterministic(model):
    """MC dropout reuses backbone features across passes, so the backbone must have
    no stochastic layers. timm efficientnet_b0 has none by default (drop_path_rate=0)."""
    for module in model.base.modules():
        name = type(module).__name__
        if isinstance(module, nn.Dropout) or name == "DropPath":
            if getattr(module, "p", 0) > 0 or getattr(module, "drop_prob", 0) > 0:
                raise RuntimeError(f"Backbone has a stochastic layer ({name}); MC feature reuse is invalid.")


# ==============================================================================
# 4. TRAINING
# ==============================================================================
@torch.no_grad()
def predict_logits(model, images, batch_size, use_amp):
    """Deterministic logits (eval mode, no dropout)."""
    model.eval()
    outputs = []
    for start in range(0, images.shape[0], batch_size):
        x = prepare_batch(images[start:start + batch_size])
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        outputs.append(logits.float())
    return torch.cat(outputs)


def train_one_model(lambda1, train_x, train_y, val_x, val_y, class_weights, device):
    set_seed(CONFIG["seed"])
    use_amp = device.type == "cuda"
    batch_size = CONFIG["batch_size"]
    epochs = CONFIG["epochs"]

    model = SoftCAMEfficientNet(
        num_classes=CONFIG["num_classes"],
        dropout_rate=CONFIG["dropout_rate"],
        pretrained=CONFIG["pretrained"],
    ).to(device)
    model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Same optimizer as your previous runs (Adam, lr 1e-4, wd 5e-4) for parity
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    steps_per_epoch = train_x.shape[0] // batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch, eta_min=CONFIG["lr"] * 0.01
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_qwk = -1.0
    best_epoch = 0
    best_state = None
    val_labels_np = val_y.cpu().numpy()

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        perm = torch.randperm(train_x.shape[0], device=device)
        sum_loss = torch.zeros((), device=device)
        sum_ce = torch.zeros((), device=device)
        sum_l1 = torch.zeros((), device=device)

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size:(step + 1) * batch_size]
            x = train_x[idx].float() / 255.0
            x = gpu_augment(x)
            x = prepare_batch(x)
            y = train_y[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, class_maps = model(x, return_maps=True)
            # Loss in fp32 for stability
            loss, ce, l1, _ = softcam_loss(
                logits.float(), y, class_maps, criterion, lambda1, CONFIG["lambda2"]
            )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            sum_loss += loss.detach()
            sum_ce += ce.detach()
            sum_l1 += l1.detach()

        val_logits = predict_logits(model, val_x, batch_size, use_amp)
        val_preds = val_logits.argmax(dim=1).cpu().numpy()
        qwk = cohen_kappa_score(val_labels_np, val_preds, weights="quadratic")
        acc = float((val_preds == val_labels_np).mean())

        marker = ""
        if qwk > best_qwk:
            best_qwk = float(qwk)
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"

        print(
            f"  [l1={lambda1:g}] epoch {epoch + 1:02d}/{epochs} | "
            f"loss {sum_loss.item() / steps_per_epoch:.4f} "
            f"(CE {sum_ce.item() / steps_per_epoch:.4f}, sum|A| {sum_l1.item() / steps_per_epoch:.1f}) | "
            f"val QWK {qwk:.4f} acc {acc:.4f} | {time.time() - start_time:.1f}s{marker}"
        )

    return best_qwk, best_epoch, best_state


# ==============================================================================
# 5. MC DROPOUT + CALIBRATION
# ==============================================================================
@torch.no_grad()
def mc_logits(model, images, passes, batch_size, seed, use_amp):
    """MC-dropout logits [T, N, C].

    The backbone is deterministic in eval mode, so it runs once per batch and only
    dropout + the 1x1 evidence conv are repeated T times. This gives exactly the
    same result as T full forward passes with dropout on, ~T times faster.
    BatchNorm stays in eval mode (the old train_aptos.py used model.train(),
    which also switched BatchNorm to batch statistics).
    """
    model.eval()
    torch.manual_seed(seed)  # same dropout masks for every condition -> paired comparison
    p = model.dropout_rate
    per_batch = []
    for start in range(0, images.shape[0], batch_size):
        x = prepare_batch(images[start:start + batch_size])
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=use_amp):
            features = model.base.forward_features(x)
        features = features.float()
        runs = []
        for _ in range(passes):
            dropped = F.dropout(features, p=p, training=True)
            runs.append(model.evidence_conv(dropped).mean(dim=(2, 3)))
        per_batch.append(torch.stack(runs))  # [T, b, C]
    return torch.cat(per_batch, dim=1)


def find_temperature(logits, labels):
    """Grid search the temperature that minimises NLL (numpy inputs)."""
    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.25, 5.0, 96):
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        nll = -log_probs[np.arange(len(labels)), labels].mean()
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t, float(best_nll)


def summarize_mc(run_logits, labels, temperature):
    """Turn MC logits [T,N,C] into calibrated probabilities + uncertainty signals."""
    run_logits = run_logits.cpu()
    mean_logits = run_logits.mean(dim=0)
    probs = torch.softmax(mean_logits / temperature, dim=1).numpy()
    per_pass = torch.softmax(run_logits / temperature, dim=2).numpy()  # [T,N,C]

    n = probs.shape[0]
    rows = np.arange(n)
    preds = probs.argmax(axis=1)
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    mc_std = per_pass[:, rows, preds].std(axis=0)

    uncertain = (
        (entropy > UNCERTAINTY_ENTROPY_THRESHOLD)
        | (margin < UNCERTAINTY_MARGIN_THRESHOLD)
        | (mc_std > UNCERTAINTY_MC_STD_THRESHOLD)
    )
    correct = preds == labels

    metrics = {
        "qwk": float(cohen_kappa_score(labels, preds, weights="quadratic")),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "mean_entropy": float(entropy.mean()),
        "mean_margin": float(margin.mean()),
        "mean_mc_uncertainty": float(mc_std.mean()),
        "uncertain_fraction": float(uncertain.mean()),
        "quadrant_certain_wrong": int((~uncertain & ~correct).sum()),
        "quadrant_certain_right": int((~uncertain & correct).sum()),
        "quadrant_uncertain_wrong": int((uncertain & ~correct).sum()),
        "quadrant_uncertain_right": int((uncertain & correct).sum()),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(CONFIG["num_classes"]))).tolist(),
        "predictions": preds.tolist(),
    }
    return metrics, probs


# ==============================================================================
# 6. HEATMAPS
# ==============================================================================
def colorize_heatmap(heatmap):
    h = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


@torch.no_grad()
def compute_evidence_maps(model, images, batch_size, use_amp):
    """Single deterministic forward pass -> evidence maps [N,C,h,w] and logits [N,C]."""
    model.eval()
    all_maps = []
    for start in range(0, images.shape[0], batch_size):
        x = prepare_batch(images[start:start + batch_size])
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=use_amp):
            maps = model.forward_maps(x)
        all_maps.append(maps.float())
    maps = torch.cat(all_maps)
    return maps, maps.mean(dim=(2, 3))


def save_heatmaps(idrid_imgs_hwc, evidence_up, mapped_class, labels, det_preds, maps_low, out_dir):
    heatmap_dir = os.path.join(out_dir, "softcam_heatmaps", "idrid", "test")
    overlay_dir = os.path.join(out_dir, "softcam_overlays", "idrid", "test")
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # Display heatmap: positive evidence only, min-max to [0,1] (same as before)
    positive = F.relu(evidence_up)
    flat = positive.flatten(start_dim=1)
    mins = flat.min(dim=1).values.view(-1, 1, 1)
    maxs = flat.max(dim=1).values.view(-1, 1, 1)
    display = ((positive - mins) / (maxs - mins + 1e-8)).cpu().numpy()

    # Raw evidence maps for all classes (low resolution, signed) for later analysis
    np.savez(
        os.path.join(heatmap_dir, "raw_evidence_maps.npz"),
        maps=maps_low.cpu().numpy(),
        mapped_class=mapped_class,
    )

    rows = []
    for i in range(display.shape[0]):
        np.save(os.path.join(heatmap_dir, f"{i}.npy"), display[i].astype(np.float32))
        if i < CONFIG["overlay_limit"]:
            image = idrid_imgs_hwc[i].astype(np.float32) / 255.0  # raw image, correct colours
            overlay = np.clip(0.45 * image + 0.55 * colorize_heatmap(display[i]), 0.0, 1.0)
            Image.fromarray((overlay * 255).astype(np.uint8)).save(os.path.join(overlay_dir, f"{i}.png"))
        signed = maps_low[i, mapped_class[i]]
        rows.append({
            "idx": i,
            "true": int(labels[i]),
            "pred": int(det_preds[i]),
            "mapped_class": int(mapped_class[i]),
            "positive_evidence_fraction": float((signed > 0).float().mean()),
            "evidence_max": float(signed.max()),
            "evidence_min": float(signed.min()),
        })

    with open(os.path.join(heatmap_dir, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} heatmaps -> {heatmap_dir}")
    print(f"Saved overlays -> {overlay_dir}")


# ==============================================================================
# 7. OCCLUSION MASKS
# ==============================================================================
def build_masks(mode, kind, top_k, seed, evidence_up, fundus_mask):
    """Boolean masks [N,H,W] covering exactly top_k percent of each image.

    mode "pixel": rank single pixels (targeted) / random pixels anywhere (random).
    mode "patch": rank patch_size x patch_size patches by mean evidence (targeted).
                  kind "random": uniform random patches anywhere in the image,
                      matching the sampling distribution used by the TTE-CAM/
                      GradCAM++ random baselines (RandomMeanOccludedDataset) —
                      this is the cross-method comparison baseline.
                  kind "random_retina": random patches drawn from inside the
                      retina first, background only as overflow — a harder,
                      retina-aware control, reported as a separate ablation.
    Ranking uses the signed evidence with topk, so ties can no longer
    make the mask cover the whole image.
    """
    n, size, _ = evidence_up.shape
    device = evidence_up.device
    masks = torch.zeros((n, size, size), dtype=torch.bool, device=device)

    if mode == "pixel":
        num_pixels = size * size
        k = int(round(num_pixels * top_k / 100.0))
        for i in range(n):
            if kind == "targeted":
                chosen = torch.topk(evidence_up[i].flatten(), k).indices
            elif kind == "random":
                generator = torch.Generator().manual_seed(seed + i)
                chosen = torch.randperm(num_pixels, generator=generator)[:k].to(device)
            else:
                raise ValueError(f"Unsupported kind '{kind}' for mode 'pixel'")
            flat_mask = masks[i].view(-1)
            flat_mask[chosen] = True
        return masks

    patch = CONFIG["patch_size"]
    grid = size // patch
    num_patches = grid * grid
    k = max(1, int(round(num_patches * top_k / 100.0)))
    patch_scores = F.avg_pool2d(evidence_up[:, None, :grid * patch, :grid * patch], patch)[:, 0]
    patch_in_retina = F.avg_pool2d(fundus_mask[:, None, :grid * patch, :grid * patch].float(), patch)[:, 0] > 0.5

    for i in range(n):
        if kind == "targeted":
            chosen = torch.topk(patch_scores[i].flatten(), k).indices
        elif kind == "random":
            # True uniform random over ALL patches (retina + background).
            generator = torch.Generator().manual_seed(seed + i)
            chosen = torch.randperm(num_patches, generator=generator)[:k].to(device)
        elif kind == "random_retina":
            # Retina-biased random control: prefer patches on the retina,
            # only use background if there are not enough.
            generator = torch.Generator().manual_seed(seed + i)
            order = torch.randperm(num_patches, generator=generator)
            inside = patch_in_retina[i].flatten().cpu()[order]
            ordered = torch.cat([order[inside], order[~inside]])
            chosen = ordered[:k].to(device)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        patch_mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
        patch_mask[chosen] = True
        patch_mask = patch_mask.view(grid, grid)
        full = patch_mask.repeat_interleave(patch, dim=0).repeat_interleave(patch, dim=1)
        masks[i, :grid * patch, :grid * patch] = full
    return masks


def apply_occlusion(images_u8, masks):
    """Fill masked pixels with the ImageNet mean (becomes 0 after normalization)."""
    x = images_u8.float() / 255.0
    fill = IMAGENET_MEAN.to(x.device).view(1, 3, 1, 1)
    return torch.where(masks[:, None], fill, x)


def paired_wilcoxon(targeted, random_mean):
    """One-sided test: targeted drop > random drop. Returns p-value or NaN."""
    diff = targeted - random_mean
    if len(diff) < 5 or np.allclose(diff, 0):
        return float("nan")
    try:
        return float(wilcoxon(targeted, random_mean, alternative="greater").pvalue)
    except ValueError:
        return float("nan")


# ==============================================================================
# 8. MAIN
# ==============================================================================
def main():
    total_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    set_seed(CONFIG["seed"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Device: {torch.cuda.get_device_name(0)} | AMP on | GPU-resident data")
    else:
        print("Device: CPU (no GPU found)")
    print(f"Uncertainty thresholds from: {THRESHOLD_SOURCE}")

    out_dir = CONFIG["output_dir"]
    for sub in ["weights", "calibration", "softcam_occlusion"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # ---- 1. Data ------------------------------------------------------------
    print("\n--- 1. LOADING DATA ---")
    aptos_csv = resolve_path(CONFIG["aptos_csv"], CONFIG["alt_aptos_csv"])
    aptos_dir = resolve_path(CONFIG["aptos_img_dir"], CONFIG["alt_aptos_img_dir"])
    idrid_csv = resolve_path(CONFIG["idrid_csv"], CONFIG["alt_idrid_csv"])
    idrid_dir = resolve_path(CONFIG["idrid_img_dir"], CONFIG["alt_idrid_img_dir"])

    aptos_df = pd.read_csv(aptos_csv).reset_index(drop=True)
    idrid_df = pd.read_csv(idrid_csv)
    idrid_df = idrid_df.dropna(subset=[CONFIG["idrid_img_col"]]).reset_index(drop=True)

    aptos_imgs, aptos_labels, _ = load_or_build_cache(
        "aptos", aptos_df, aptos_dir, CONFIG["aptos_img_col"], CONFIG["aptos_label_col"], CONFIG["aptos_ext"]
    )
    idrid_imgs, idrid_labels, idrid_names = load_or_build_cache(
        "idrid", idrid_df, idrid_dir, CONFIG["idrid_img_col"], CONFIG["idrid_label_col"], CONFIG["idrid_ext"]
    )

    # Same stratified 80/20 split as before
    all_idx = np.arange(len(aptos_labels))
    train_idx, val_idx = train_test_split(
        all_idx, test_size=0.2, random_state=CONFIG["seed"], stratify=aptos_labels
    )
    print(f"APTOS train {len(train_idx)} | val {len(val_idx)} | IDRiD test {len(idrid_labels)}")

    train_x = to_device_uint8(aptos_imgs[train_idx], device)
    val_x = to_device_uint8(aptos_imgs[val_idx], device)
    train_y = torch.from_numpy(aptos_labels[train_idx]).long().to(device)
    val_y = torch.from_numpy(aptos_labels[val_idx]).long().to(device)
    val_labels_np = aptos_labels[val_idx]
    del aptos_imgs

    weights = compute_class_weight(
        "balanced", classes=np.arange(CONFIG["num_classes"]), y=aptos_labels[train_idx]
    )
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # ---- 2. Training / lambda1 sweep ---------------------------------------
    print("\n--- 2. TRAINING ---")
    sweep_results = []
    for lambda1 in CONFIG["lambda1_sweep"]:
        run_start = time.time()
        best_qwk, best_epoch, state = train_one_model(
            lambda1, train_x, train_y, val_x, val_y, class_weights, device
        )
        path = os.path.join(out_dir, "weights", f"aptos_softcam_l1_{lambda1:g}.pth")
        torch.save(state, path)
        sweep_results.append({"lambda1": lambda1, "best_val_qwk": best_qwk, "best_epoch": best_epoch, "path": path, "state": state})
        print(f"  -> lambda1={lambda1:g}: best val QWK {best_qwk:.4f} at epoch {best_epoch} ({time.time() - run_start:.0f}s)")

    top_qwk = max(r["best_val_qwk"] for r in sweep_results)
    chosen = None
    for r in sweep_results:
        if r["best_val_qwk"] >= top_qwk - CONFIG["qwk_tolerance"]:
            if chosen is None or r["lambda1"] > chosen["lambda1"]:
                chosen = r
    print("\nLambda1 sweep:")
    for r in sweep_results:
        mark = "  <- selected" if r is chosen else ""
        print(f"  lambda1={r['lambda1']:<8g} val QWK {r['best_val_qwk']:.4f} (epoch {r['best_epoch']}){mark}")

    model = SoftCAMEfficientNet(
        num_classes=CONFIG["num_classes"], dropout_rate=CONFIG["dropout_rate"], pretrained=False
    ).to(device)
    model.load_state_dict(chosen["state"])
    model = model.to(memory_format=torch.channels_last)
    model.eval()
    check_backbone_is_deterministic(model)
    weights_path = os.path.join(out_dir, "weights", "aptos_softcam.pth")
    torch.save(chosen["state"], weights_path)
    print(f"Selected model saved -> {weights_path}")
    del train_x
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- 3. Temperature calibration ---------------------------------------
    print("\n--- 3. TEMPERATURE CALIBRATION (MC dropout) ---")
    val_runs = mc_logits(model, val_x, CONFIG["mc_passes"], CONFIG["batch_size"], CONFIG["seed"], use_amp)
    temperature, nll = find_temperature(val_runs.mean(dim=0).cpu().numpy(), val_labels_np)
    val_metrics, _ = summarize_mc(val_runs, val_labels_np, temperature)
    np.save(os.path.join(out_dir, "calibration", "optimal_T_softcam.npy"), np.array(temperature))
    print(f"Optimal T = {temperature:.3f} (NLL {nll:.4f}) | MC val QWK {val_metrics['qwk']:.4f}")
    del val_x

    # ---- 4. IDRiD evidence maps -------------------------------------------
    print("\n--- 4. IDRiD EVIDENCE MAPS (single forward pass) ---")
    idrid_x = to_device_uint8(idrid_imgs, device)
    size = CONFIG["img_size"]
    maps_low, det_logits = compute_evidence_maps(model, idrid_x, CONFIG["batch_size"], use_amp)
    det_preds = det_logits.argmax(dim=1)
    mapped_class = det_preds.cpu().numpy()  # explain the predicted class
    row_idx = torch.arange(maps_low.shape[0], device=device)
    class_map = maps_low[row_idx, det_preds]  # [N,h,w], signed
    evidence_up = F.interpolate(class_map[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    print(f"Evidence map resolution: {maps_low.shape[-2]}x{maps_low.shape[-1]} -> upsampled to {size}x{size}")
    save_heatmaps(idrid_imgs, evidence_up, mapped_class, idrid_labels, det_preds.cpu().numpy(), maps_low, out_dir)

    fundus_mask = idrid_x.float().mean(dim=1) > 10.0  # non-black retina pixels

    # ---- 5. Occlusion benchmark -------------------------------------------
    print("\n--- 5. OCCLUSION BENCHMARK ---")
    n = len(idrid_labels)
    rows_idx = np.arange(n)
    correct_mask = mapped_class == idrid_labels  # map explains a correct prediction
    dr_mask = idrid_labels > 0

    def run_condition(images_float):
        runs = mc_logits(model, images_float, CONFIG["mc_passes"], CONFIG["batch_size"], CONFIG["seed"], use_amp)
        return summarize_mc(runs, idrid_labels, temperature)

    base_metrics, base_probs = run_condition(idrid_x)
    base_true = base_probs[rows_idx, idrid_labels]
    base_mapped = base_probs[rows_idx, mapped_class]

    results = {
        "dataset": "IDRiD official test split",
        "method": "SoftCAM",
        "img_size": size,
        "circle_crop": CONFIG["circle_crop"],
        "selected_lambda1": chosen["lambda1"],
        "lambda1_sweep": [{k: v for k, v in r.items() if k != "state"} for r in sweep_results],
        "temperature": temperature,
        "mc_passes": CONFIG["mc_passes"],
        "patch_size": CONFIG["patch_size"],
        "uncertainty_threshold_source": THRESHOLD_SOURCE,
        "n_images": n,
        "n_correct_mapped": int(correct_mask.sum()),
        "n_dr_positive": int(dr_mask.sum()),
        "conditions": {"baseline": base_metrics},
        "wilcoxon": [],
    }
    summary_rows = [{"mode": "-", "condition": "baseline", "top_k_percent": 0, "seed": "-", **base_metrics}]
    per_image = {
        "idx": rows_idx.tolist(),
        "image": [str(x) for x in idrid_names],
        "label": idrid_labels.tolist(),
        "mapped_class": mapped_class.tolist(),
        "baseline_true_prob": base_true.tolist(),
        "baseline_mapped_prob": base_mapped.tolist(),
    }
    stats_rows = []

    def add_drops(metrics, probs):
        true_drop = base_true - probs[rows_idx, idrid_labels]
        mapped_drop = base_mapped - probs[rows_idx, mapped_class]
        metrics["mean_true_prob_drop"] = float(true_drop.mean())
        metrics["positive_true_drop_fraction"] = float((true_drop > 0).mean())
        metrics["mean_mapped_prob_drop"] = float(mapped_drop.mean())
        metrics["mean_mapped_prob_drop_correct"] = float(mapped_drop[correct_mask].mean()) if correct_mask.any() else float("nan")
        metrics["mean_mapped_prob_drop_dr"] = float(mapped_drop[dr_mask].mean()) if dr_mask.any() else float("nan")
        return mapped_drop

    for mode in CONFIG["occlusion_modes"]:
        for top_k in CONFIG["top_k"]:
            # Targeted
            masks = build_masks(mode, "targeted", top_k, 0, evidence_up, fundus_mask)
            metrics, probs = run_condition(apply_occlusion(idrid_x, masks))
            targeted_drop = add_drops(metrics, probs)
            metrics["occluded_fraction"] = float(masks.float().mean())
            key = f"{mode}_softcam_top_{top_k:g}"
            results["conditions"][key] = metrics
            per_image[f"{key}_mapped_drop"] = targeted_drop.tolist()
            summary_rows.append({"mode": mode, "condition": "softcam", "top_k_percent": top_k, "seed": "-", **metrics})

            # Random controls. "random" (uniform over the whole image) is the
            # cross-method comparison baseline, matching TTE-CAM/GradCAM++'s
            # random condition. "random_retina" (patch mode only) is reported
            # as a separate, harder ablation and does not feed the Wilcoxon test.
            random_kinds = ["random", "random_retina"] if mode == "patch" else ["random"]
            random_mean_by_kind = {}

            for rand_kind in random_kinds:
                random_drops = []
                random_list = []
                for seed in CONFIG["random_seeds"]:
                    masks = build_masks(mode, rand_kind, top_k, seed, evidence_up, fundus_mask)
                    metrics, probs = run_condition(apply_occlusion(idrid_x, masks))
                    drop = add_drops(metrics, probs)
                    metrics["occluded_fraction"] = float(masks.float().mean())
                    metrics["seed"] = seed
                    random_drops.append(drop)
                    random_list.append(metrics)
                    summary_rows.append({"mode": mode, "condition": rand_kind, "top_k_percent": top_k, **metrics})
                results["conditions"][f"{mode}_{rand_kind}_top_{top_k:g}"] = random_list
                random_mean = np.mean(np.stack(random_drops), axis=0)
                random_mean_by_kind[rand_kind] = random_mean
                per_image[f"{mode}_{rand_kind}_top_{top_k:g}_mean_mapped_drop"] = random_mean.tolist()

            random_mean = random_mean_by_kind["random"]

            # Paired test: is the targeted drop bigger than the (uniform) random drop?
            stat = {
                "mode": mode,
                "top_k_percent": top_k,
                "targeted_mean_drop": float(targeted_drop.mean()),
                "random_mean_drop": float(random_mean.mean()),
                "p_all": paired_wilcoxon(targeted_drop, random_mean),
                "targeted_mean_drop_correct": float(targeted_drop[correct_mask].mean()) if correct_mask.any() else float("nan"),
                "random_mean_drop_correct": float(random_mean[correct_mask].mean()) if correct_mask.any() else float("nan"),
                "p_correct": paired_wilcoxon(targeted_drop[correct_mask], random_mean[correct_mask]),
                "targeted_mean_drop_dr": float(targeted_drop[dr_mask].mean()) if dr_mask.any() else float("nan"),
                "random_mean_drop_dr": float(random_mean[dr_mask].mean()) if dr_mask.any() else float("nan"),
                "p_dr": paired_wilcoxon(targeted_drop[dr_mask], random_mean[dr_mask]),
            }

            if "random_retina" in random_mean_by_kind:
                rr_mean = random_mean_by_kind["random_retina"]
                stat["random_retina_mean_drop"] = float(rr_mean.mean())
                stat["p_vs_random_retina"] = paired_wilcoxon(targeted_drop, rr_mean)

            stats_rows.append(stat)
            results["wilcoxon"].append(stat)
            
    # ---- 6. Save outputs ---------------------------------------------------
    occ_dir = os.path.join(out_dir, "softcam_occlusion")
    with open(os.path.join(occ_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_fields = [
        "mode", "condition", "top_k_percent", "seed", "occluded_fraction", "qwk", "accuracy",
        "mean_confidence", "mean_entropy", "mean_margin", "mean_mc_uncertainty", "uncertain_fraction",
        "mean_true_prob_drop", "positive_true_drop_fraction", "mean_mapped_prob_drop",
        "mean_mapped_prob_drop_correct", "mean_mapped_prob_drop_dr",
    ]
    summary_df = pd.DataFrame([{k: r.get(k, "") for k in summary_fields} for r in summary_rows])
    summary_df.to_csv(os.path.join(occ_dir, "summary.csv"), index=False)
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(occ_dir, "wilcoxon.csv"), index=False)
    pd.DataFrame(per_image).to_csv(os.path.join(occ_dir, "per_image.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n================ OCCLUSION RESULTS ================")
    print(summary_df[["mode", "condition", "top_k_percent", "seed", "qwk", "accuracy",
                      "mean_true_prob_drop", "mean_mapped_prob_drop", "uncertain_fraction"]].round(4).to_string(index=False))
    print("\n========== TARGETED vs RANDOM (paired Wilcoxon, one-sided) ==========")
    print(stats_df.round(4).to_string(index=False))
    print(f"\nImages: {n} | correct (mapped class == label): {int(correct_mask.sum())} | DR-positive: {int(dr_mask.sum())}")
    print(f"Saved results -> {occ_dir}")
    print(f"Total time: {(time.time() - total_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
