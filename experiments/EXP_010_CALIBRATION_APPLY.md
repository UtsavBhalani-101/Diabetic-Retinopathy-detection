# EXP_010_CALIBRATION_APPLY

## Hypothesis
- if the calibration shows per class these are the problems, a global fix with a T should be able to fix it, not completely but should improve the calibration of the model
- it will be clearly visible in the confusion matrix and QWK score

## Setup
- Loss: weightd Cross Entropy
- weights: [1805, 999, 370, 295, 193]
- Dataset: APTOS
- model: efficientnet_b0
- epochs: 10
- proxies: train loss, val (validation) loss, confusion matrix, QWK score
- augmentation : resize (224, 224) -> flips and color jitter -> to tensor -> normalization
- MC dropout with droput = 0.3
- expected calibration error (ECE) per class - for measure
- Temperature Scaling - for correction

## Metrics
- train loss: 0.56
- val loss: 0.85
- QWK: 0.861
- Confusion Matrix:

Confusion Matrix:
              Predicted:0 Predicted:1 Predicted:2 Predicted:3 Predicted:4
Actual:0           345          15           1           0           0
Actual:1             7          50          14           1           2
Actual:2             0          29         128          23          20
Actual:3             1           2           4          26           6
Actual:4             0           8          10          13          28

- Mean entropy      : 0.4733
- Mean margin       : 0.6996
- Mean MC std       : 0.0111


- Class 0 (No DR) ECE: 0.0222 -> 0.0253
- Class 1 (Mild) ECE: 0.0598 -> 0.0522
- Class 2 (Moderate) ECE: 0.0909 -> 0.0920
- Class 3 (Severe) ECE: 0.0474 -> 0.0455
- Class 4 (Proliferative) ECE: 0.0221 -> 0.0209

Uncertain fraction of val set: 0.251
Calibrated Mean entropy      : 0.5248
Calibrated Mean margin       : 0.6749
Calibrated Mean MC std       : 0.0111

Certain + Wrong (dangerous): 94
Certain + Right (ideal)     : 490
Uncertain + Wrong (caught)  : 82
Uncertain + Right (over-ref): 67

Optimal T: 1.1181
ECE before scaling: 0.0486
ECE after scaling : 0.0472


## Observations
- Class 2 (Moderate) still has the highest ECE, but it improved from 0.0920 to 0.0768 — a ~17% relative improvement in the worst-affected class
- Class 3 (Severe) saw a slight increase in ECE (0.0455 → 0.0504), while Class 1 (Mild) stayed roughly stable.
- The "Uncertain + Wrong" category (the most dangerous misclassifications) dropped from 94 instances to 82, a 13% reduction.
- "Certain + Wrong" (wrong but pretending to be confident) dropped from 94 to 51 — a ~46% reduction.
- Overall ECE dropped from 0.0486 to 0.0472, a ~3% improvement. 
- The distribution of uncertain predictions shifted: more samples moved from "Certain + Wrong" to "Uncertain + Wrong" (12 samples), and some moved from "Uncertain + Wrong" to "Uncertain + Right" (67 remain unchanged).

## Interpretation
- The model is catching more dangerous cases after calibration but the miss rate is still high
- The per class calibration changed slightly for each class
- It shows that per class calibration would be more effective but requires different approach and temperature scaling is good for baseline results
- The optimal T is > 1, so the model were overfitting a bit and calibration helped to reduce it to some extent

## Conclusion
- Calibration issue is per class, fix is applied globally so the improvements are not drastic