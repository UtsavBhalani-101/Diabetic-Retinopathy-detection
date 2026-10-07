# Standalone TTECAM Occlusion Experiment

Train on APTOS, generate TTE CAM heatmaps on the IDRiD test split, then run
a targeted-vs-random occlusion sensitivity test — the same design used for the
GradCAM++ occlusion experiment (EXP_015).

---

## Step 1 — Train on APTOS

`powershell
python -m ttecam_experiment.train_aptos `
  --epochs 10 `
  --lr 1e-4 `
  --batch-size 64 `
  --num-workers 4 `
  --dropout-rate 0.3 `
  --seed 42
`

Outputs:

- `artifacts/weights/aptos_efficientnet.pth` — EfficientNetMC checkpoint
- `artifacts/calibration/optimal_T.npy` — temperature scalar from MC Dropout calibration

Uses `DATASET_REGISTRY["APTOS_2019"]` paths (Kaggle). No WandB required.

---

## Step 2 — Generate IDRiD TTECAM Heatmaps

`powershell
python -m ttecam_experiment.generate_ttecam_heatmaps `
  --batch-size 16 `
  --num-workers 0 `
  --heatmap-dir artifacts/ttecam_heatmaps/idrid/test `
  --overlay-dir artifacts/ttecam_overlays/idrid/test `
  --heatmap-png-dir artifacts/ttecam_heatmap_pngs/idrid/test `
  --overlay-limit 24 `
  --class-source predicted
`

Outputs:

- `artifacts/ttecam_heatmaps/idrid/test/{idx}.npy` — normalized [H,W] float32 heatmap per image
- `artifacts/ttecam_heatmaps/idrid/test/metadata.csv` — true label, pred class, heatmap stats
- `artifacts/ttecam_overlays/idrid/test/{idx}.png` — heatmap blended onto original image
- `artifacts/ttecam_heatmap_pngs/idrid/test/{idx}.png` — standalone heatmap PNG

Use `--class-source true` to generate heatmaps for the ground-truth class instead.

---

## Step 3 — Targeted vs. Random Occlusion

`powershell
python -m ttecam_experiment.run_ttecam_occlusion `
  --batch-size 16 `
  --num-workers 0 `
  --mc-passes 30 `
  --top-k 10 30 `
  --random-seeds 42 43 44 `
  --heatmap-dir artifacts/ttecam_heatmaps/idrid/test `
  --output-json artifacts/ttecam_occlusion/idrid/results.json `
  --summary-csv artifacts/ttecam_occlusion/idrid/summary.csv
`

Outputs:

- `artifacts/ttecam_occlusion/idrid/results.json` — full per-condition results
- `artifacts/ttecam_occlusion/idrid/summary.csv` — flat table for quick comparison

Conditions evaluated:

| Condition | What is occluded |
|---|---|
| Baseline | Nothing (full image) |
| TTECAM top-10% | Top 10% of TTE CAM activation pixels → filled with ImageNet mean |
| Random 10% (x3 seeds) | Random 10% of pixels → filled with ImageNet mean |
| TTECAM top-30% | Top 30% of TTE CAM activation pixels |
| Random 30% (x3 seeds) | Random 30% of pixels |

Multiple random seeds guard against single-seed collapse (same issue seen in EXP_015 at 30%).

---

## Notes

- Uses `artifacts/weights/aptos_efficientnet.pth` (written by Step 1).
- Uses `artifacts/calibration/optimal_T.npy` for temperature-scaled probabilities.
- IDRiD test split: `datasets/IDRiD/B. Disease Grading/1. Original Images/b. Testing Set`
- TTECAM converts the trained linear classifier into an equivalent 1x1 convolution at test time.
  Global average pooling of the spatial maps recovers the original logits exactly.
- Occlusion fill: ImageNet channel means [0.485, 0.456, 0.406] in raw [0,1] space.
  After gpu_normalize these become exactly 0 — semantically neutral input.
