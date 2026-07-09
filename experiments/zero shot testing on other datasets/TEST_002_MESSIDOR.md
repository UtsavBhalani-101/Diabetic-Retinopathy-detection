# ZERO_SHOT_002_MESSIDOR

## Dataset
- Source: Messidor-1 (French multicenter diabetic retinopathy dataset)
- Population: French patients, Topcon TRC NW6 camera, 3 hospital sites
- Total images: ~1200 across 3 groups (~400 per group)
- Label mapping: {0:0, 1:1, 2:2, 3:4} — Messidor grade 3 maps to APTOS 
  Proliferative. APTOS Class 3 (Severe) never appears in ground truth.

## Performance
| Metric | APTOS Val | G1 | G2 | G3 |
|--------|-----------|-----|-----|-----|
| QWK (4-class corrected)¹ | 0.87 | **0.2049** | **0.1581** | **0.0689** |
| QWK (5-class, original logged) | — | 0.4854 | 0.4217 | 0.3706 |
| Uncertain Fraction | 0.20 | 0.32 | 0.22 | 0.21 |
| Certain+Wrong | 79 (13%) | 125 (31%) | 144 (36%) | 119 (30%) |

> ¹ **Label-mapping correction.** Messidor uses a 4-class grading scheme (grades 0–3)
> mapped to APTOS labels `{0:0, 1:1, 2:2, 3→4}` — APTOS class 3 (Severe) is structurally
> absent in Messidor ground truth (nothing in the mapping produces label 3). Corrected QWK
> is computed with `cohen_kappa_score(weights='quadratic', labels=[0,1,2,4])`, excluding
> class 3 from the label universe entirely. The 5-class weight matrix included a phantom
> class 3 that biased the distance normalization.

## Failure Pattern
- Proliferative DR (Class 4) collapses entirely to Class 0 across all groups.
  G1: 1/149 correct. G2: 0/52 correct. G3: 0/53 correct.
  Model sees severe Messidor pathology and confidently predicts healthy retina.
- Class 3 (Severe) row is all zeros in every group — expected, not a bug.
  Messidor has no dedicated Severe DR grade; nothing maps to APTOS Class 3.
- QWK degrades across groups: G1→G2→G3 (0.2049→0.1581→0.0689) on the corrected
  4-class metric. Inter-hospital equipment and protocol variation compounds the base shift.

## Uncertainty Behavior
- Uncertain fraction barely rises above APTOS baseline (0.20→0.32/0.22/0.21).
  Model does not know it is out of distribution — this is silent failure.
- Certain+Wrong rate is 30-36% across all groups vs 13% on APTOS val.
  Uncertainty mechanism provides no meaningful protection here.
- Contrast with IDRiD: uncertain fraction rose to 0.55 there, catching more errors.
  Messidor's covariate shift is larger but triggers less uncertainty — the model's
  wrong predictions land deep inside its Class 0 confidence region, not near 
  decision boundaries where dropout variance would increase.

## Calibration
- Class 0 ECE: 0.25/0.32/0.27 across groups — worst calibration in entire evaluation.
  Model is overconfident on No-DR predictions that are largely incorrect.
- Class 4 ECE G1: 0.337 — severe miscalibration on the class failing most.
- Temperature T learned on APTOS does not transfer. Calibration is 
  dataset-specific and cannot be corrected without Messidor validation data.

## Interpretation
Messidor represents the largest covariate shift in this evaluation. French patient
population, different scanner hardware, and 4-class grading scheme combine to
produce a distribution the model has never seen. The critical finding is not the
QWK drop itself but that the uncertainty system fails to detect the failure —
mean margin stays high (0.63-0.74) and uncertain fraction stays low while
Certain+Wrong rate triples vs APTOS. This is silent distribution shift: the model
is confidently wrong without any internal signal that something is wrong.

The inter-hospital degradation (G1→G2→G3) demonstrates that even within a single
dataset, equipment variation across sites produces measurable performance drops
that the uncertainty system does not catch.