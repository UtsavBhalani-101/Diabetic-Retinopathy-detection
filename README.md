# Diabetic Retinopathy Severity Classification

Automated grading of diabetic retinopathy from retinal fundus images into 5 severity levels (No DR → Proliferative DR), with uncertainty quantification and systematic cross-dataset failure analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange) ![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B0-green) ![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)

![APTOS](https://img.shields.io/badge/Dataset-APTOS%202019-red) ![IDRiD](https://img.shields.io/badge/Dataset-IDRiD-red) ![Messidor](https://img.shields.io/badge/Dataset-Messidor-red) ![DDR](https://img.shields.io/badge/Dataset-DDR-red) ![EyePACS](https://img.shields.io/badge/Dataset-EyePACS-red)

---

## The Core Finding

A model achieving **0.87 QWK** on APTOS drops to **0.37–0.62** on four external datasets. The failure is not random — the uncertainty system works on IDRiD (flags 55% of predictions as uncertain, Certain+Wrong drops to 22%) but **fails silently on Messidor**, where the model is **30–36% confidently wrong** with no internal warning signal. The model's uncertainty detection catches some distribution shifts but completely misses others.

This project is an investigation of why — 11 iterative experiments on APTOS, zero-shot evaluation on 4 external datasets (6 test sets), Mahalanobis OOD analysis, and a Ben Graham preprocessing experiment that cleanly decomposes the failure into surface appearance shift vs. representational shift.

---

## System Overview

```
pipeline/
├── data/
│   ├── dataset.py              # APTOS + external dataset loading, augmentation
│   └── loaders.py              # DataLoader construction, class-weight computation
├── setup/
│   ├── config.py               # All hyperparameters, paths, thresholds
│   └── utils.py                # Seeding, device setup, W&B logging
├── training_loop_setup/
│   └── model.py                # EfficientNet-B0 + MC Dropout architecture
├── evaluation/
│   ├── evaluate.py             # QWK, confusion matrix, entropy, margin, MC std
│   └── calibration.py          # Temperature scaling, ECE, triage routing
├── orchestrators/
│   ├── train.py                # Full training loop with uncertainty metrics
│   ├── test.py                 # Zero-shot cross-dataset evaluation
│   ├── predict.py              # Single-image inference with triage
│   └── run_pipeline.py         # CLI entry point (train / test / predict)
api/
└── app.py                      # FastAPI deployment: /predict, /health, /model/info
artifacts/
├── weights/                    # Trained model checkpoint
├── calibration/                # Optimal temperature T, calibration plots
└── logs/                       # W&B run logs
```

**Flow:** Training (WCE + augmentation + MC Dropout) → Temperature calibration on validation set → Zero-shot testing on external datasets → Mahalanobis OOD distance analysis → Triage routing via FastAPI endpoint.

The API layer accepts a retinal image and returns a severity prediction, calibrated confidence, three uncertainty signals (entropy, margin, MC std), and a clinical triage label (`ROUTINE` / `HIGH SEVERITY - urgent review` / `UNCERTAIN - refer to specialist`).

---

## Why This Is Hard

The dataset is simultaneously **ordinal**, **imbalanced**, and **noisy**. Disease severity is a continuum — adjacent classes share subtle visual differences that even ophthalmologists disagree on. Class distribution is heavily skewed (1805 No DR vs 193 Proliferative). Standard classification metrics hide dangerous failure modes: a model can achieve high accuracy by being confidently wrong on minority classes, which in a clinical screening context is the worst possible outcome.

---

## Approach

EfficientNet-B0 with weighted cross-entropy, trained iteratively across **11 experiments** — each driven by a specific diagnostic observation from the previous one, not by hyperparameter sweeping.

**Concrete example of the methodology:** After adding augmentation in EXP_007, the confusion matrix showed *bidirectional errors on middle classes* (Class 1 ↔ 2 ↔ 3) — the model was making mistakes in both directions, not just defaulting to the majority class. This meant the decision boundaries in that region were genuinely unstable, but the confusion matrix treated a 94%-confident correct prediction identically to a 51%-confident correct prediction. MC Dropout was added in EXP_008 specifically to surface that per-prediction instability: run 30 stochastic forward passes and measure how much the output varies. If the prediction flips under dropout perturbation, it's fragile — exactly the kind of dangerous prediction a clinical system needs to flag.

**Key design decisions:**
- Focal loss was tested and rejected — hard samples in this dataset are noisy, not informative. When combined with augmentation, QWK dropped to 0.76 (the worst in the series) because augmentation creates harder samples and focal loss doubles down on them
- Model capacity was validated as sufficient early — training loss reached ~0.0 while val loss plateaued at 0.8, confirming the bottleneck was generalization, not representation power
- Global temperature scaling chosen over per-class calibration despite a 3.5× ECE gap between classes (Class 2: 0.077 vs Class 0: 0.022), because fitting 5+ parameters on 733 validation samples risks overfitting the calibration itself

---

## Results

### APTOS Performance

| Metric | Value |
|--------|-------|
| QWK (ordinal ranking) | **0.87** |
| Per-class ECE range | 0.022 – 0.077 |
| Worst-calibrated class | Class 2 (Moderate) — ECE 0.077 |
| Best-calibrated class | Class 0 (No DR) — ECE 0.022 |
| Calibration temperature | T optimized via post-hoc scaling |
| Uncertainty metrics | Entropy, prediction margin, MC Dropout std (30 passes) |

### Cross-Dataset Evaluation

| Dataset | QWK | Uncertain Fraction | Certain+Wrong | Avg Mahalanobis Distance | Failure Mode |
|---------|-----|--------------------|---------------|--------------------------|--------------|
| APTOS val | 0.87 | 0.20 | 13% | 48.2 | Baseline |
| IDRiD | 0.62 | 0.55 | 22% | 103.3 | Partial uncertainty detection |
| DDR | 0.54 | 0.30 | 32% | 100.6 | Class 2 collapse |
| Messidor G1 | 0.49 | 0.32 | 31% | 92.8 | Silent failure |
| Messidor G2 | 0.42 | 0.22 | 36% | 92.7 | Silent failure |
| Messidor G3 | 0.37 | 0.21 | 30% | 92.9 | Silent failure |
| EyePACS | 0.38 | 0.21 | 21% | 106.6 | Silent failure at scale |

This table is the core result. QWK drops across every external dataset — expected. What's unexpected is the Certain+Wrong column: on APTOS, the model is confidently wrong 13% of the time. On Messidor, that rate doubles to 30–36%, and the uncertainty system *does not flag it*. Meanwhile, IDRiD triggers the uncertainty system correctly (55% flagged as uncertain) despite being geometrically *further* from the training distribution than Messidor (103.3 vs 92.8 Mahalanobis distance).

---

## Key Findings

### 1. Distance doesn't predict performance — direction matters more than magnitude

Mahalanobis distance in the 1280-d EfficientNet feature space confirms all external datasets are ~2× the APTOS baseline distance (avg 92–107 vs 48). But the ranking contradicts expectations: **IDRiD has the highest distance (103.3) yet the best QWK (0.62). Messidor has the lowest distance (~92.8) yet the worst QWK (0.37–0.49).**

The feature space entangles DR-invariant features (lesion morphology, vessel patterns) with dataset-specific features (scanner color profiles, resolution artifacts). IDRiD's shift is large but in scanner-artifact dimensions — class separations survive. Messidor's shift is smaller overall but targets DR-relevant dimensions — severe cases land where healthy retinas should be. The model can't tell the difference.

### 2. Silent failure on Messidor — the uncertainty system's blind spot

On Messidor G2, the model achieves the lowest uncertain fraction in the entire evaluation (0.22) while being wrong on 36% of confident predictions. QWK is 0.42. The model is *more confident on Messidor G2 than it is on its own validation set* (uncertain fraction 0.20 on APTOS val), despite performing dramatically worse. This is the most dangerous failure mode: a model that is confidently wrong with no internal warning signal.

The mechanism: Messidor images land in the Class 0 confidence region of feature space — a large, well-sampled region (Class 0 = 49% of training data) with uniformly low dropout variance. The model sees a Messidor Proliferative DR image, maps it to a feature region indistinguishable from APTOS healthy retinas, and confidently predicts No DR.

### 3. Ben Graham preprocessing exposes the two-layer decomposition

Retraining with Ben Graham preprocessing (radius-based cropping + Gaussian blur subtraction) and testing with the same preprocessing collapsed Mahalanobis distances from ~92–107 to ~50–58 (converging toward the APTOS baseline of 48). The surface appearance gap was real and large. **But QWK dropped or held flat on every dataset except IDRiD.**

This cleanly separates two independent failure layers:
- **Layer 1 — Surface appearance shift:** Different scanners produce different illumination, contrast, and color profiles. Ben Graham removes this. Confirmed and quantified (~40–50 distance units of the original gap were purely appearance).
- **Layer 2 — Representational shift:** Even with identical preprocessing, the model's learned features don't transfer. Images converge toward the APTOS Class 0 centroid (the majority class), not toward their correct class. This is the dominant problem.

**The key result:** On IDRiD after Ben Graham retraining, Certain+Wrong dropped from 22% to 5% and uncertain fraction rose from 0.55 to 0.77. The model became *more honest* — it correctly signaled uncertainty instead of being accidentally correct. On Messidor, Certain+Wrong *increased* from 31% to 37% (G1). Ben Graham helps the model be honest about IDRiD but makes it more confidently wrong on Messidor. The fix for one failure mode worsens another.

### 4. Severity determines survivability across datasets

Across DDR and EyePACS post-Ben Graham, Class 4 (Proliferative DR) partially survives — 49% accuracy on DDR, 31% on EyePACS — while Class 1/2 (Mild/Moderate) collapses almost entirely (~7% accuracy on DDR). Proliferative DR features (neovascularization, large hemorrhages, fibrous tissue) are visually distinctive enough to transfer across scanners. Moderate DR features, which are clinically ambiguous by definition, don't survive preprocessing or domain changes.

---

## Limitations and Next Steps

**Unresolved:**
- The 1280-d feature space mixes DR-invariant and scanner-specific features. Every metric computed on this space inherits the ambiguity
- Uncertainty thresholds tuned on APTOS val don't transfer — catches IDRiD shifts but misses Messidor
- Messidor's label mapping ({0,1,2,3} → APTOS {0,1,2,4}) may not capture the same clinical boundaries

**Planned:**
- **DANN** — penalize the encoder for learning dataset-specific features, forcing DR-invariant representations
- **Per-class Mahalanobis distance** — distance from each of 5 class centroids to reveal whether failures are feature corruption vs. boundary problems
- **Post-hoc threshold adjustment** — sweep per-class decision thresholds to separate feature-level failure from boundary-level failure
- **GradCAM activation maps** — verify whether the model attends to lesions on APTOS but scanner artifacts on Messidor
- **CLAHE preprocessing** — test whether the two-layer decomposition holds with a different standardization method

---

## Documentation

- [`experiments/Decision_log.md`](experiments/Decision_log.md) — why each technical choice was made
- [`experiments/Findings.md`](experiments/Findings.md) — cross-experiment analysis and recurring patterns
- [`experiments/EXP_001` through `EXP_011`](experiments/) — individual experiment logs with hypothesis, metrics, and interpretation
- [`experiments/zero shot testing on other datasets/`](experiments/zero%20shot%20testing%20on%20other%20datasets/) — cross-dataset evaluation logs (IDRiD, Messidor, EyePACS, DDR)
- [`experiments/ben_graham_hypothesis_analysis.md`](experiments/ben_graham_hypothesis_analysis.md) — Ben Graham retraining experiment with two-layer decomposition and 6 competing hypotheses
- [`experiments/pipeline_architecture.md`](experiments/pipeline_architecture.md) — full pipeline architecture documentation
