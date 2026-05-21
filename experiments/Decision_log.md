

1. why choose cross entropy and why switch to WCE ?
- the task is a classification task and the first standard loss fuction is CE
- the data we are dealing with is ordinal + imbalance
- the CE assumes every class is equally important, to fix that weighted CE is used

2. Why added augmentation and specifically what is the reasoning behind each of those augmentations ?
- the task is to force the model to learn underlying structural patterns and not make decisions based on pixel level information 
- the another reason to add augmentation is this model is also going to be tested on different datasets, to capture cross dataset features accurately and to make it robust 

- the normal flow of augmentation is image transformations -> convert to tensor -> normalization
- each transformation is made with a reason that it should force model to not make predictions from pixels _and_ also not lose information as medical data is sensitive 

3. Why not use focal loss ?
- ans in  [Findings](Findings.md)

4. Why used efficient net model b0 ?
- it's fast and efficient, the b0 version (smallest in the family) is used in the start to quickly test the scope and difficulty of the task 
- it performed better then expected and there was no need to change the model, as the model was not a problem but representation and data
- when testing on other dataset, I will change it to a bigger model when I will be testing other datasets and if it's required
- Training loss reached 0.5 within 5-10 epochs and overfit to near 0.0, indicating the model has sufficient capacity to learn the data. The bottleneck was generalization, not model capacity — confirmed by val loss plateauing while train loss continued dropping. Switching to a larger model would worsen overfitting, not improve generalization

5. What's the goal and why ?
- The goal is not just to get high numbers, the goal is to make this system robust enough that if tomorrow this is deployed in a clinic, it should handle the messy real world and be grounded by the things it actually learned and not overconfident
- this is also a part of the reason why I am added different metrics, testing on other datasets, showing where the model is uncertain and optimizing 


6. Why used MC dropout ?
- ans in [Findings](Findings.md)

7. Why use calibration ?
- 2 reasons: softmax outputs from NNs are systematically overconfident — a prediction of 0.9 doesn't mean 90% accuracy, it's usually lower _and_ in a DR screening system, a confident-but-wrong prediction (e.g. predicting "No DR" at 95% when the patient has Moderate DR) directly delays treatment
- ECE (Expected Calibration Error) is used to measure miscalibration. It bins predictions by confidence, compares predicted confidence to actual accuracy per bin, and produces a single scalar gap. It's the standard metric for calibration in classification
- per-class ECE is calculated because the calibration problems are class-specific: Class 2 (Moderate) has the highest ECE at 0.0768 — consistent with its confusion matrix errors and decision boundary instability from EXP_007/008. Class 0 (No DR) has the lowest at 0.0222 — majority class with distinct visual features. Classes 3/4 have low ECE (0.032, 0.0317) despite being minority classes, because severe DR features (hemorrhages, neovascularization) are visually discriminative. A single aggregate ECE would hide this 3.5x gap between class 2 and class 0
- temperature scaling is used to correct calibration: it divides the logit vector by a single learned parameter T before softmax, softening or sharpening the entire distribution
- T is applied globally (one scalar for all 5 classes) because softmax couples all logit values — changing one logit affects all 5 output probabilities, so per-class correction isn't possible with a single temperature parameter
- per-class correction is not applied because would require vector scaling (5 parameters, one per class) or matrix scaling (25 parameters). With only 733 validation samples, fitting 5+ calibration parameters risks overfitting the calibration correction itself
- the tradeoff is explicit: global T can't fix Class 2's 0.0768 ECE without also shifting Class 0's already-good 0.0222. But one robust parameter on 733 samples is more reliable than five noisy ones

