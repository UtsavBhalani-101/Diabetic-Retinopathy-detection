# pipeline/smoke_test.py
# ============================================================
# CPU-only smoke test — no image files, no GPU, no Kaggle paths, no wandb.
#
# Tests every pipeline component end-to-end with synthetic data.
# Runs in < 2 minutes on any laptop.
#
# Usage:
#   python -m pipeline.smoke_test       (from project root)
#   python pipeline/smoke_test.py
# ============================================================

import os
import sys
import logging
import tempfile

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ---- make project root importable ----
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.config      import (
    setup_logging,
    UNCERTAINTY_ENTROPY_THRESHOLD,
    UNCERTAINTY_MARGIN_THRESHOLD,
    UNCERTAINTY_MC_STD_THRESHOLD,
)
from pipeline.dataset     import setting_gpu, set_seed, val_transformer
from pipeline.model       import EfficientNetMC, get_loss_criterion
from pipeline.evaluate    import evaluate, mc_evaluate_full, compute_uncertainty_signals
from pipeline.calibration import find_temperature, apply_temperature, per_class_calibration

logger = logging.getLogger(__name__)

# ============================================================
# Smoke-test configuration — tiny numbers so it runs fast
# ============================================================
SMOKE_CFG = {
    "num_samples":       40,    # synthetic samples total
    "num_classes":       5,
    "batch_size":        8,
    "epochs":            2,
    "mc_passes":         3,     # T=3 instead of T=30
    "lr":                1e-3,
    "dropout_rate":      0.3,
    "seed":              42,
    "num_workers":       0,     # MUST be 0 on Windows without __main__ guard
    "pin_memory":        False, # No CUDA on laptop
}

PASS = "[PASS]"
FAIL = "[FAIL]"


# ============================================================
# Tiny synthetic dataset (no disk I/O)
# ============================================================

class SyntheticRetinopathyDataset(Dataset):
    """
    Random 224×224 RGB tensors with stratified integer labels 0-4.
    No image files required — lives entirely in memory.
    """
    def __init__(self, n: int, n_classes: int = 5, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.images = torch.randn(n, 3, 224, 224)          # [N, C, H, W]
        # Stratified labels — same distribution every time
        self.labels = torch.tensor(
            [i % n_classes for i in range(n)], dtype=torch.long
        )
        logger.debug(f"SyntheticDataset | n={n} | label dist: {np.bincount(self.labels.numpy())}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx].item()


# ============================================================
# Individual component tests
# ============================================================

def _section(title: str) -> None:
    logger.info("")
    logger.info(f"{'─'*50}".replace('─', '-'))
    logger.info(f"  {title}")
    logger.info("-" * 50)


def test_seed_and_device():
    _section("1. Seed + Device")
    set_seed(SMOKE_CFG["seed"])
    device = setting_gpu()
    logger.info(f"  device={device}")
    assert device is not None
    logger.info(f"  {PASS}")
    return device


def test_synthetic_dataset():
    _section("2. Synthetic Dataset + DataLoader")
    ds = SyntheticRetinopathyDataset(
        n=SMOKE_CFG["num_samples"],
        n_classes=SMOKE_CFG["num_classes"],
        seed=SMOKE_CFG["seed"]
    )
    loader = DataLoader(ds, batch_size=SMOKE_CFG["batch_size"],
                        shuffle=False, num_workers=SMOKE_CFG["num_workers"])

    imgs, labels = next(iter(loader))
    assert imgs.shape   == (SMOKE_CFG["batch_size"], 3, 224, 224), f"Bad image shape: {imgs.shape}"
    assert labels.shape == (SMOKE_CFG["batch_size"],),             f"Bad label shape: {labels.shape}"
    assert labels.min() >= 0 and labels.max() < SMOKE_CFG["num_classes"]
    logger.info(f"  batch image shape : {tuple(imgs.shape)}")
    logger.info(f"  batch label range : [{labels.min()}, {labels.max()}]")
    logger.info(f"  {PASS}")
    return loader


def test_model_forward(device):
    _section("3. EfficientNetMC forward pass (pretrained=False)")
    model = EfficientNetMC(
        num_classes=SMOKE_CFG["num_classes"],
        dropout_rate=SMOKE_CFG["dropout_rate"],
        pretrained=False          # skip ImageNet download
    ).to(device)

    dummy = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (2, SMOKE_CFG["num_classes"]), f"Bad output shape: {out.shape}"
    logger.info(f"  output shape : {tuple(out.shape)}")
    logger.info(f"  {PASS}")
    return model


def test_loss_criterion(device):
    _section("4. Class-balanced loss criterion")
    # Build a minimal DataFrame matching what get_loss_criterion expects
    labels_arr = [i % SMOKE_CFG["num_classes"]
                  for i in range(SMOKE_CFG["num_samples"])]
    df = pd.DataFrame({"diagnosis": labels_arr})
    criterion = get_loss_criterion(df, "diagnosis", device)
    assert hasattr(criterion, "weight")
    assert criterion.weight.shape[0] == SMOKE_CFG["num_classes"]
    logger.info(f"  class weights : {criterion.weight.cpu().numpy().round(3).tolist()}")
    logger.info(f"  {PASS}")
    return criterion


def test_evaluate(model, loader, criterion, device):
    _section("5. evaluate() — standard validation pass")
    loss, qwk, matrix, preds, labels = evaluate(model, loader, criterion, device)
    assert isinstance(loss, float)
    assert -1.0 <= qwk <= 1.0, f"QWK out of range: {qwk}"
    assert matrix.shape == (SMOKE_CFG["num_classes"], SMOKE_CFG["num_classes"])
    logger.info(f"  loss={loss:.4f} | QWK={qwk:.4f}")
    logger.info(f"  {PASS}")


def test_mc_evaluate(model, loader, device):
    _section("6. mc_evaluate_full() — MC Dropout passes")
    mean_probs, uncertainties, gt_labels, logits = mc_evaluate_full(
        model, loader, device, T=SMOKE_CFG["mc_passes"]
    )
    N = SMOKE_CFG["num_samples"]
    C = SMOKE_CFG["num_classes"]

    assert mean_probs.shape    == (N, C), f"Bad mean_probs shape: {mean_probs.shape}"
    assert uncertainties.shape == (N,),   f"Bad uncertainty shape: {uncertainties.shape}"
    assert gt_labels.shape     == (N,),   f"Bad labels shape: {gt_labels.shape}"
    assert logits.shape        == (N, C), f"Bad logits shape: {logits.shape}"

    # Probabilities must sum to ~1
    row_sums = mean_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"Probs don't sum to 1: {row_sums[:3]}"

    logger.info(f"  mean_probs shape  : {mean_probs.shape}")
    logger.info(f"  mean uncertainty  : {uncertainties.mean():.4f}")
    logger.info(f"  {PASS}")
    return mean_probs, uncertainties, gt_labels, logits


def test_uncertainty_signals(mean_probs, uncertainties):
    _section("7. compute_uncertainty_signals()")
    entropy, margin, mc_unc = compute_uncertainty_signals(mean_probs, uncertainties)

    N = SMOKE_CFG["num_samples"]
    assert entropy.shape == (N,), f"Bad entropy shape: {entropy.shape}"
    assert margin.shape  == (N,), f"Bad margin shape:  {margin.shape}"
    assert (entropy >= 0).all(),  "Negative entropy detected"
    assert (margin  >= 0).all(),  "Negative margin detected"

    logger.info(f"  entropy range : [{entropy.min():.3f}, {entropy.max():.3f}]")
    logger.info(f"  margin  range : [{margin.min():.3f},  {margin.max():.3f}]")
    logger.info(f"  {PASS}")


def test_temperature_scaling(logits, gt_labels, tmp_dir):
    _section("8. find_temperature() + apply_temperature()")
    optimal_T = find_temperature(logits, gt_labels)
    assert 0.1 <= optimal_T <= 10.0, f"T out of bounds: {optimal_T}"

    cal_probs = apply_temperature(logits, optimal_T)
    assert cal_probs.shape == logits.shape
    row_sums = cal_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)
    logger.info(f"  optimal_T = {optimal_T:.4f}")
    logger.info(f"  {PASS}")
    return optimal_T, cal_probs


def test_calibration_plot(cal_probs, gt_labels, tmp_dir):
    _section("9. per_class_calibration() — reliability diagrams")
    save_path = os.path.join(tmp_dir, "smoke_calibration.png")
    per_class_calibration(cal_probs, gt_labels, save_path=save_path)
    assert os.path.isfile(save_path), f"Plot file not created: {save_path}"
    size_kb = os.path.getsize(save_path) / 1024
    logger.info(f"  saved → {save_path}  ({size_kb:.1f} KB)")
    logger.info(f"  {PASS}")


def test_mini_train_loop(device, tmp_dir):
    _section("10. Mini end-to-end train loop (2 epochs, no wandb, CPU)")

    n      = SMOKE_CFG["num_samples"]
    n_cls  = SMOKE_CFG["num_classes"]
    bs     = SMOKE_CFG["batch_size"]

    ds     = SyntheticRetinopathyDataset(n, n_cls, seed=0)
    loader = DataLoader(ds, batch_size=bs, shuffle=True,
                        num_workers=SMOKE_CFG["num_workers"])
    val_ds = SyntheticRetinopathyDataset(n, n_cls, seed=1)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=SMOKE_CFG["num_workers"])

    labels_arr = [i % n_cls for i in range(n)]
    df         = pd.DataFrame({"diagnosis": labels_arr})
    criterion  = get_loss_criterion(df, "diagnosis", device)

    model = EfficientNetMC(n_cls, dropout_rate=0.3, pretrained=False).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=SMOKE_CFG["lr"])

    for epoch in range(SMOKE_CFG["epochs"]):
        model.train()
        epoch_loss = 0.0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        logger.info(f"  Epoch {epoch+1} | loss={epoch_loss/len(loader):.4f}")

    # save weights
    model_path = os.path.join(tmp_dir, "smoke_model.pth")
    torch.save(model.state_dict(), model_path)
    assert os.path.isfile(model_path)
    logger.info(f"  Model saved -> {model_path}")

    # reload and run MC eval
    loaded = EfficientNetMC(n_cls, dropout_rate=0.3, pretrained=False).to(device)
    loaded.load_state_dict(torch.load(model_path, map_location=device))

    mean_probs, uncerts, gt_labels, logits = mc_evaluate_full(
        loaded, val_loader, device, T=SMOKE_CFG["mc_passes"]
    )
    optimal_T  = find_temperature(logits, gt_labels)
    cal_probs  = apply_temperature(logits, optimal_T)
    entropy, margin, mc_unc = compute_uncertainty_signals(cal_probs, uncerts)
    uncertain_mask = (
        (entropy > UNCERTAINTY_ENTROPY_THRESHOLD) |
        (margin < UNCERTAINTY_MARGIN_THRESHOLD) |
        (mc_unc > UNCERTAINTY_MC_STD_THRESHOLD)
    )

    logger.info(f"  MC eval done | N={len(gt_labels)} | T={optimal_T:.4f}")
    logger.info(f"  Uncertain fraction: {uncertain_mask.mean():.3f}")
    logger.info(f"  {PASS}")


# ============================================================
# Runner
# ============================================================

def run_all_tests():
    setup_logging(log_dir="artifacts/logs")
    set_seed(SMOKE_CFG["seed"])

    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE SMOKE TEST - CPU only, synthetic data")
    logger.info("  No image files / GPU / Kaggle paths / wandb required")
    logger.info("=" * 60)

    passed = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            device    = test_seed_and_device();    passed += 1
        except Exception as e:
            logger.error(f"  {FAIL}: test_seed_and_device → {e}"); failed += 1; device = torch.device("cpu")

        try:
            loader    = test_synthetic_dataset();  passed += 1
        except Exception as e:
            logger.error(f"  {FAIL}: test_synthetic_dataset → {e}"); failed += 1; return

        try:
            model     = test_model_forward(device);  passed += 1
        except Exception as e:
            logger.error(f"  {FAIL}: test_model_forward → {e}"); failed += 1; return

        try:
            criterion = test_loss_criterion(device); passed += 1
        except Exception as e:
            logger.error(f"  {FAIL}: test_loss_criterion → {e}"); failed += 1; return

        try:
            test_evaluate(model, loader, criterion, device); passed += 1
        except Exception as e:
            logger.error(f"  {FAIL}: test_evaluate → {e}"); failed += 1

        try:
            mean_probs, uncerts, gt_labels, logits = test_mc_evaluate(model, loader, device)
            passed += 1
        except Exception as e:
            logger.error(f"  {FAIL}: test_mc_evaluate → {e}"); failed += 1; gt_labels = logits = None

        if gt_labels is not None:
            try:
                test_uncertainty_signals(mean_probs, uncerts); passed += 1
            except Exception as e:
                logger.error(f"  {FAIL}: test_uncertainty_signals → {e}"); failed += 1

            try:
                optimal_T, cal_probs = test_temperature_scaling(logits, gt_labels, tmp_dir)
                passed += 1
            except Exception as e:
                logger.error(f"  {FAIL}: test_temperature_scaling → {e}"); failed += 1; cal_probs = None

            if cal_probs is not None:
                try:
                    test_calibration_plot(cal_probs, gt_labels, tmp_dir); passed += 1
                except Exception as e:
                    logger.error(f"  {FAIL}: test_calibration_plot → {e}"); failed += 1

        try:
            test_mini_train_loop(device, tmp_dir); passed += 1
        except Exception as e:
            logger.error(f"  {FAIL}: test_mini_train_loop → {e}"); failed += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  RESULTS: {passed} passed | {failed} failed")
    if failed == 0:
        logger.info("  ALL TESTS PASSED [OK]")
    else:
        logger.warning(f"  {failed} TEST(S) FAILED - check errors above")
    logger.info("=" * 60)
    logger.info("")

    return failed == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
