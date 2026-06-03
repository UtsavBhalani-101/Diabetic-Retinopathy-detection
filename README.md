# Diabetic Retinopathy Severity Classification

Automated grading of diabetic retinopathy from retinal fundus images into 5 severity levels (No DR → Proliferative DR), built on the APTOS 2019 dataset.

## Why This Is Hard

The dataset is simultaneously **ordinal**, **imbalanced**, and **noisy**. Disease severity is a continuum — adjacent classes share subtle visual differences that even ophthalmologists disagree on. Class distribution is heavily skewed (1805 No DR vs 193 Proliferative). Standard classification metrics hide dangerous failure modes: a model can achieve high accuracy by being confidently wrong on minority classes, which in a clinical screening context is the worst possible outcome.

## Approach

EfficientNet-B0 with weighted cross-entropy, trained iteratively across **11 experiments** — each driven by a specific diagnostic observation from the previous one, not by hyperparameter sweeping. The progression: baseline → validation-based evaluation → frequency-based class weights → augmentation for structural feature learning → MC Dropout for uncertainty quantification → per-class calibration measurement (ECE) → zero-shot cross-dataset evaluation → Mahalanobis distance OOD analysis.

Key design decisions: focal loss was tested and rejected (hard samples in this dataset are noisy, not informative). Model capacity was validated as sufficient early — the bottleneck was generalization, not representation power.

## Results

- **QWK: 0.87** — ordinal ranking accuracy held stable from experiment 4 onward
- **Per-class ECE: 0.02–0.08** — calibration is worst on the middle severity classes (Mild, Moderate), confirming the transition-zone confusion pattern
- **Uncertainty quantification** via entropy, prediction margin, and MC Dropout std — the system flags which predictions are confident vs fragile
- **Zero-shot cross-dataset evaluation** on IDRiD (QWK 0.62), Messidor (0.37–0.49), EyePACS, and DDR — reveals silent failure modes where the model is confidently wrong without triggering uncertainty
- **Mahalanobis distance OOD analysis** — all external datasets sit ~2x further from the training cluster than APTOS val, but distance does not predict performance. IDRiD is furthest yet performs best; Messidor is closest yet fails worst. The shift's direction matters more than its magnitude — the feature space entangles DR-invariant and scanner-specific features

## Documentation

- [`experiments/Decision_log.md`](experiments/Decision_log.md) — why each technical choice was made
- [`experiments/Findings.md`](experiments/Findings.md) — cross-experiment analysis and recurring patterns
- [`experiments/EXP_001` through `EXP_011`](experiments/) — individual experiment logs with hypothesis, metrics, and interpretation
- [`experiments/zero shot testing on other datasets/`](experiments/zero%20shot%20testing%20on%20other%20datasets/) — cross-dataset evaluation logs (IDRiD, Messidor, EyePACS, DDR)
