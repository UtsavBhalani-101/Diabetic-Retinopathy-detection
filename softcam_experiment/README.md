# Standalone SoftCAM Occlusion Experiment

Train a self-explainable **SoftCAM** model on APTOS 2019, generate single-pass class-evidence heatmaps on the IDRiD test split, and run targeted-vs-random occlusion sensitivity testing.

---

## Background & Methodology

Unlike post-hoc CAM methods (GradCAM) or test-time conversion (TTE-CAM), **SoftCAM** transforms the CNN into an inherently interpretable model:
1. **Removes the Global Average Pooling (GAP) layer** prior to classification.
2. **Replaces the Linear head with a $1 \times 1$ convolutional class-evidence layer** $h_\psi(Z) \in \mathbb{R}^{B \times C \times H \times W}$.
3. **Derives predictions directly via spatial average pooling** of the evidence maps: $\text{logits}_c = \frac{1}{HW}\sum A_{c, i, j}$.
4. **Applies an ElasticNet penalty ($\ell_1 + \ell_2$) on the evidence maps** during backpropagation to penalize non-focal activations.

---

## Step 1 — Train SoftCAM on APTOS 2019

```powershell
python -m softcam_experiment.train_aptos `
  --epochs 10 `
  --lr 1e-4 `
  --batch-size 32 `
  --num-workers 0 `
  --dropout-rate 0.3 `
  --lambda1 1e-4 `
  --lambda2 0.0 `
  --seed 42
```

### Outputs
- `artifacts/weights/aptos_softcam.pth` — retrained SoftCAMEfficientNet checkpoint
- `artifacts/calibration/optimal_T_softcam.npy` — temperature scalar from MC Dropout validation calibration

---

## Step 2 — Generate IDRiD SoftCAM Heatmaps

Extracts class-specific evidence maps and predictions in a **single forward pass** (no backpropagation or perturbations):

```powershell
python -m softcam_experiment.generate_softcam_heatmaps `
  --batch-size 16 `
  --num-workers 0 `
  --heatmap-dir artifacts/softcam_heatmaps/idrid/test `
  --overlay-dir artifacts/softcam_overlays/idrid/test `
  --heatmap-png-dir artifacts/softcam_heatmap_pngs/idrid/test `
  --overlay-limit 24 `
  --class-source predicted
```

### Outputs
- `artifacts/softcam_heatmaps/idrid/test/{idx}.npy` — normalized $[H, W]$ float32 heatmap per image
- `artifacts/softcam_heatmaps/idrid/test/metadata.csv` — true label, predicted class, and heatmap distribution statistics
- `artifacts/softcam_overlays/idrid/test/{idx}.png` — visual overlay on original image
- `artifacts/softcam_heatmap_pngs/idrid/test/{idx}.png` — standalone colorized heatmap PNG

---

## Step 3 — Targeted vs. Random Occlusion Sensitivity Test

Tests whether the regions SoftCAM highlights are causally necessary for model predictions by comparing against matched random occlusions:

```powershell
python -m softcam_experiment.run_softcam_occlusion `
  --batch-size 16 `
  --num-workers 0 `
  --mc-passes 30 `
  --top-k 10 30 `
  --random-seeds 42 43 44 `
  --heatmap-dir artifacts/softcam_heatmaps/idrid/test `
  --output-json artifacts/softcam_occlusion/idrid/results.json `
  --summary-csv artifacts/softcam_occlusion/idrid/summary.csv
```

### Conditions Evaluated
| Condition | What is occluded | Fill Color |
|---|---|---|
| **Baseline** | Nothing (full image) | None |
| **SoftCAM top-10%** | Top 10% highest activation pixels | ImageNet mean `[0.485, 0.456, 0.406]` (becomes 0 after normalization) |
| **Random 10% (x3 seeds)** | Random 10% of pixels | ImageNet mean |
| **SoftCAM top-30%** | Top 30% highest activation pixels | ImageNet mean |
| **Random 30% (x3 seeds)** | Random 30% of pixels | ImageNet mean |

### Outputs
- `artifacts/softcam_occlusion/idrid/results.json` — full per-condition results, confusion matrices, and true class probability drops
- `artifacts/softcam_occlusion/idrid/summary.csv` — summary comparison table
