# ZERO_SHOT_004_DDR

## Dataset
- Source: DDR (Diabetic Retinopathy Detection, China)
- Population: Chinese patients, mixed clinical equipment
- Total images: ~12,522
- Label mapping: direct 0-4, same scale as APTOS

## Performance
| Metric | APTOS Val | DDR |
|--------|-----------|-----|
| QWK | 0.87 | 0.54 |
| Uncertain Fraction | 0.20 | 0.30 |
| Certain+Wrong | 79 (13%) | 2731 (32%) |

## Failure Pattern
- Moderate DR (Class 2) is the primary failure: 2487/4477 true Moderate cases
  predicted as Class 0. Model collapses middle-severity cases to healthy.
- Class 0 performance is relatively intact: 5949/6266 correct. Unlike Messidor,
  the model is not systematically misclassifying healthy retinas.
- Classes 3 and 4 show partial recovery: 57/236 and 456/913 correct respectively.
  Severe and Proliferative DR have distinctive enough lesion patterns to partially
  transfer across Chinese vs Indian patient populations.
- Class 2 ECE 0.257 — worst calibration in this evaluation. Model is underconfident
  on Moderate DR, hedging toward Class 0 instead of committing.

## Uncertainty Behavior
- Uncertain fraction rises to 0.30 vs 0.20 on APTOS — partial detection.
  Better than Messidor (0.21-0.32) but uncertainty is catching the wrong cases.
- Certain+Wrong rate 32% vs 13% on APTOS. Uncertainty reduces dangerous failures
  proportionally but does not prevent them at scale — 2731 absolute dangerous
  failures due to dataset size.
- The shift pattern here is different from Messidor: model is uncertain about
  some things but confidently wrong about Moderate DR specifically.

## Calibration
- Class 2 ECE 0.257 is the dominant calibration problem — underconfidence on
  the class that's failing most. Model knows something is off on Moderate DR
  but not enough to trigger the uncertainty threshold.
- Classes 3 and 4 ECE near zero (0.037, 0.011) — well calibrated on severe cases,
  consistent with those classes partially transferring.
- Temperature T from APTOS partially transfers for severe classes but not for
  the middle of the severity spectrum.

## Interpretation
DDR sits between IDRiD and Messidor in difficulty. Chinese patient population
and different equipment produce meaningful shift, but the same 0-4 label space
and some shared lesion morphology allow partial transfer. The failure is
concentrated in Moderate DR — the most ambiguous class clinically — where the
model loses confidence and collapses to Class 0. Unlike Messidor's silent failure
on severe disease, DDR's failure is noisier and partially flagged by uncertainty,
making it a less dangerous failure mode but still clinically unacceptable.