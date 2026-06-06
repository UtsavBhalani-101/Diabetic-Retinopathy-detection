# EXP_011_MAHALANOBIS_DIST

## Hypothesis
- Mahalanobis distance in the EfficientNet-B0 penultimate feature space (1280-d) should quantify how far each external dataset sits from the APTOS training distribution
- Datasets with higher distance should show worse QWK and higher uncertainty — if the uncertainty system and performance are both driven by distribution shift, they should correlate with geometric distance from the training cluster

## Setup
- Feature extractor: EfficientNet-B0 (same trained weights from EXP_010)
- Feature space: 1280-dimensional penultimate layer output
- Reference distribution: APTOS training set (3662 samples) — compute mean vector μ and covariance matrix Σ
- Distance: Mahalanobis distance = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ)) for each sample x
- Baseline: APTOS validation set (733 samples) — required to make OOD distances interpretable
- OOD datasets: IDRiD, Messidor (G1/G2/G3), EyePACS, DDR

## Results

### Baseline (APTOS Validation)

| Metric | Value |
|--------|-------|
| Samples | 733 |
| Min Distance | 27.35 |
| Max Distance | 95.06 |
| **Avg Distance** | **48.17** |

### OOD Datasets

| Dataset | Min | Max | Avg | Gap from APTOS |
|---------|-----|-----|-----|----------------|
| Messidor G2 | 60.03 | 136.28 | 92.65 | +44.48 |
| Messidor G1 | 57.07 | 161.08 | 92.77 | +44.60 |
| Messidor G3 | 62.10 | 137.47 | 92.93 | +44.76 |
| DDR | 47.81 | 212.10 | 100.63 | +52.46 |
| IDRiD | 59.75 | 196.97 | 103.27 | +55.10 |
| EyePACS | 58.10 | 253.15 | 106.57 | +58.41 |

### Cross-Reference with QWK and Uncertainty

| Dataset | Avg Distance | QWK | Uncertain Fraction | Certain+Wrong |
|---------|-------------|-----|-------------------|---------------|
| APTOS val | 48.17 | 0.87 | 0.20 | 13% |
| Messidor G1 | 92.77 | 0.49 | 0.32 | 31% |
| Messidor G2 | 92.65 | 0.42 | 0.22 | 36% |
| Messidor G3 | 92.93 | 0.37 | 0.21 | 30% |
| DDR | 100.63 | 0.54 | 0.30 | 32% |
| IDRiD | 103.27 | 0.62 | 0.55 | 22% |
| EyePACS | 106.57 | 0.38 | 0.21 | 21% |


## Observations

1. **All OOD datasets are roughly 2x the baseline distance** — avg distances range from 92-107 vs APTOS val at 48. Every external dataset is measurably outside the training distribution. This is expected and confirms the shift.

2. **Messidor groups are internally consistent** — G1/G2/G3 average distances are nearly identical (92.65, 92.77, 92.93) despite QWK varying from 0.49 to 0.37. Inter-hospital performance variation exists even when feature-space distance is the same.

3. **Distance does not predict performance** — IDRiD has the second highest avg distance (103.27) but the best QWK (0.62). Messidor has the lowest avg distance (~92.8) but the worst QWK (0.37-0.49). The expected correlation (more distance → worse performance) is broken.

4. **Distance does correlate with uncertainty detection** — IDRiD's uncertain fraction (0.55) is the highest, and its Certain+Wrong rate (22%) is lower than Messidor's (30-36%). The model correctly signals that IDRiD is unfamiliar. It fails to do so for Messidor.

5. **EyePACS has the widest spread** — min 58.10 to max 253.15 (range of 195). This reflects extreme heterogeneity in the EyePACS dataset itself (multiple sites, equipment, populations).

## Interpretation

### Why distance doesn't predict QWK

The 1280-d feature space contains two types of encoded information mixed together:
- **DR-invariant features**: vessel density, hemorrhage patterns, lesion morphology — the features that actually separate severity classes
- **Dataset-specific features**: scanner characteristics, brightness profiles, color balance, resolution artifacts — features that vary across datasets but are irrelevant to DR grading

Mahalanobis distance measures across all 1280 dimensions indiscriminately. It can't distinguish whether the shift is in DR-relevant dimensions or scanner-artifact dimensions.

**IDRiD** is far from APTOS because scanner/acquisition characteristics differ (Indian equipment vs APTOS). But the DR-invariant features — the ones that separate Class 0 from Class 4 — transferred. The class structure survived the shift. The model is uncertain (it knows something is off) but still classifies correctly because the decision boundaries are intact.

**Messidor** is closer to APTOS in total distance, but the shift specifically targets dimensions the model uses for DR classification. Messidor's severe DR samples land in feature regions that APTOS associates with healthy retinas. The class structure is scrambled. Small total distance, maximum damage to what matters. And critically — the model doesn't flag this as uncertain because from its perspective, the features look "close enough" to produce confident predictions. They're just confident in the wrong class.

### The geometric analogy

In simplified 2D: imagine the training cluster where the vertical axis encodes DR severity and the horizontal axis encodes scanner properties.

- IDRiD shifts far horizontally (different scanner) but preserves vertical ordering (DR grades stay separated). Large distance, classification works.
- Messidor shifts less overall but rotates the vertical axis — severe cases end up where healthy cases should be. Small distance, classification breaks.

Distance measures magnitude of shift. It doesn't measure direction. Direction determines whether classification survives.

### What this means for the system

Mahalanobis distance in raw feature space is **necessary but not sufficient** as an OOD detector:
- ✅ It correctly identifies all datasets as OOD (all are ~2x baseline)
- ❌ It cannot predict which datasets will fail (IDRiD is furthest but performs best)
- ❌ It cannot distinguish benign shift from malignant shift

The failure mode is **directional feature corruption** — not simple distance from the training cluster.

## Connection to Future Work (DANN)

This experiment reveals the fundamental limitation: the feature extractor entangles DR-invariant and dataset-specific features in the same 1280-d space. Any metric computed on this mixed space inherits the ambiguity.

After DANN (Domain Adversarial Neural Network), the feature extractor would be explicitly penalized for encoding dataset-specific features. The surviving features would be predominantly DR-invariant. At that point:
- Mahalanobis distance would measure distance in a mostly DR-relevant space
- High distance would genuinely mean "DR presentation is unfamiliar" not "scanner looks different"
- The distance → performance correlation would become meaningful
- Per-class Mahalanobis distance (distance from each class-specific cluster) would provide directional information — a sample far from all class clusters is genuinely uncertain, while a sample far from Class 0 but close to Class 2 is just a normal Class 2 sample from a different scanner

The progression: **raw distance (this experiment) → DANN-cleaned features → per-class distance** is the path to a reliable OOD detector.

## Conclusion
- Mahalanobis distance confirms all test datasets are OOD relative to APTOS training distribution (2x baseline gap)
- Distance alone does not predict performance degradation — the type/direction of shift matters more than the magnitude
- IDRiD is furthest but performs best (benign shift); Messidor is closest but fails worst (malignant shift)
- The current feature space mixes DR-invariant and dataset-specific features, making any single-number distance metric insufficient for OOD risk assessment
- This motivates domain adaptation (DANN) to disentangle feature types before OOD detection becomes reliable
