# ZERO_SHOT_001_IDRID

## Dataset
- Source: IDRiD (Indian Diabetic Retinopathy Image Dataset)
- Population: Indian patients, same 0-4 grading scale as APTOS
- Split: 413 train images, 103 test images (official held-out split)
- Label mapping: direct, no conversion needed

## Performance
| Metric | APTOS Val | IDRiD Train | IDRiD Test |
|--------|-----------|-------------|------------|
| QWK | 0.87 | 0.76 | 0.62 |
| Uncertain Fraction | 0.20 | 0.46 | 0.55 |
| Certain+Wrong | 79 (13%) | 85 (20%) | 23 (22%) |

## Failure Pattern
- Class 0 collapse: 34 true No-DR cases, only 7 correctly identified on test.
  21 predicted as Mild DR — model hedges toward Class 1 under distribution shift.
- Class 2 holds reasonably: 22/32 correct. Moderate DR lesion patterns transfer better.
- Classes 3 and 4 have small sample sizes (<20 each), results are noisy.

## Uncertainty Behavior
- Uncertainty mechanism partially works: fraction rises from 0.20 (APTOS) to 0.55 (IDRiD test).
  Model correctly signals confusion at the aggregate level.
- Certain+Wrong rate doubles from APTOS (13%) to IDRiD test (22%).
  Uncertainty catches more errors but does not eliminate dangerous failures.

## Calibration
- ECE degrades significantly on Classes 0 and 1 (0.021→0.232, 0.041→0.255).
  Temperature T=1.22 learned on APTOS does not transfer.
- Classes 3 and 4 ECE remains low — model is appropriately uncertain on severe cases.

## Interpretation
IDRiD is the smallest distribution shift in this evaluation — same population,
similar equipment, identical label space. Performance drop from 0.87→0.62 is
entirely attributable to scanner and protocol differences. The uncertainty system
partially compensates but calibration breaks down on the classes where shift is
largest (No DR, Mild DR).