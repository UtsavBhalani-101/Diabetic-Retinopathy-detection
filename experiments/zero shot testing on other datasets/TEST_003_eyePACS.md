# ZERO_SHOT_003_EYEPACS

## Dataset
- Source: EyePACS (USA, multinational screening program)
- Population: Multinational, mixed demographics, raw clinical images
- Total images: ~35,126 (sampled subset used for evaluation)
- Label mapping: direct 0-4, same scale as APTOS
- Note: APTOS training data is a curated subset derived from EyePACS.
  This evaluation tests on the raw uncurated parent dataset.

## Performance
| Metric | APTOS Val | EyePACS |
|--------|-----------|---------|
| QWK | 0.87 | 0.38 |
| Uncertain Fraction | 0.20 | 0.21 |
| Certain+Wrong | 79 (13%) | 5722 (21%) |

## Failure Pattern
- Class 0 dominates predictions at scale: 23,512 correct No-DR but minority
  classes collapse heavily. 2204/2443 true Mild cases predicted as Class 0.
  3774/5292 true Moderate cases predicted as Class 0.
- Severe and Proliferative show partial survival: 176/873 and 240/708 correct.
  Better than Messidor but far below APTOS performance.
- The parent-child relationship does not protect generalization. APTOS is a
  cleaned subset of EyePACS — training on the clean version and testing on
  the raw version is a harder shift than it appears on paper.

## Uncertainty Behavior
- Uncertain fraction 0.21 — essentially identical to APTOS baseline (0.20).
  Despite QWK dropping from 0.87 to 0.38, the model signals almost no
  additional uncertainty. This is the clearest silent failure in the evaluation.
- Certain+Wrong rate 21% vs 13% on APTOS — meaningful increase, but the
  uncertainty system is not responsible for catching it. The 5722 dangerous
  failures are the largest absolute count across all datasets.
- Mean margin 0.72, mean entropy 0.51 — nearly identical to APTOS val.
  The model does not register that it has left its training distribution.

## Calibration
- ECE is the best across all external datasets: Classes 3 and 4 near zero
  (0.015, 0.017). This is misleading — low ECE on severe classes reflects
  that the model is consistently wrong with consistent confidence, not that
  it is well calibrated in a useful sense.
- Class 0 ECE 0.152 — moderate miscalibration on the dominant prediction class.
- Overall ECE pattern suggests temperature T transfers better here than to
  Messidor, likely because the label distribution is closer to APTOS.

## Interpretation
EyePACS is the most counterintuitive result in the evaluation. The expectation
was that training on a curated APTOS subset would transfer well to the raw
EyePACS parent — the opposite is true. Raw clinical data contains variable image
quality, inconsistent lighting, poor focus, and partial coverage that the model
has never seen. The model learned features from polished images and fails on
the messy originals.

The critical finding is that EyePACS produces the most silent failure in the
entire evaluation — uncertain fraction stays at 0.21 while QWK drops to 0.38,
meaning 5722 confident wrong predictions with no internal warning signal.
Combined with dataset scale, this is the highest absolute dangerous failure
count observed, making it the most clinically dangerous deployment scenario
of all datasets tested.