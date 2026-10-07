# Standalone TTECAM Occlusion Experiment

This folder keeps the TTECAM work separate from the current pipeline. The
scripts import the existing model, dataset, and occlusion utilities, but do not
modify any existing pipeline files.

## 1. Generate IDRiD TTECAM Heatmaps

```powershell
.\.venv\Scripts\python.exe -m ttecam_experiment.generate_ttecam_heatmaps `
  --batch-size 32 `
  --num-workers 0 `
  --heatmap-dir artifacts/ttecam_heatmaps/idrid/test `
  --overlay-dir artifacts/ttecam_overlays/idrid/test `
  --heatmap-png-dir artifacts/ttecam_heatmap_pngs/idrid/test
```

Outputs:

- `artifacts/ttecam_heatmaps/idrid/test/{idx}.npy`
- `artifacts/ttecam_heatmaps/idrid/test/metadata.csv`
- `artifacts/ttecam_overlays/idrid/test/{idx}.png`
- `artifacts/ttecam_heatmap_pngs/idrid/test/{idx}.png`

By default, heatmaps are generated for the model's predicted class. To generate
true-class maps instead, add:

```powershell
--class-source true
```

## 2. Run TTECAM vs Random Occlusion

```powershell
.\.venv\Scripts\python.exe -m ttecam_experiment.run_ttecam_occlusion `
  --batch-size 32 `
  --num-workers 0 `
  --mc-passes 30 `
  --top-k 10 30 `
  --random-seeds 42 43 44 `
  --heatmap-dir artifacts/ttecam_heatmaps/idrid/test `
  --output-json artifacts/ttecam_occlusion/idrid/results.json `
  --summary-csv artifacts/ttecam_occlusion/idrid/summary.csv
```

Outputs:

- `artifacts/ttecam_occlusion/idrid/results.json`
- `artifacts/ttecam_occlusion/idrid/summary.csv`

## Notes

- Uses the pre-DANN checkpoint at `artifacts/weights/aptos_efficientnet.pth`.
- Uses the IDRiD official test split under `datasets/IDRiD`.
- Reuses the existing top-k heatmap occlusion contract: one normalized
  `224x224` heatmap saved as `{idx}.npy` per dataset item.
- Random occlusion is run with multiple seeds by default because the previous
  GradCAM++ 30% random condition showed a possible single-seed collapse.
