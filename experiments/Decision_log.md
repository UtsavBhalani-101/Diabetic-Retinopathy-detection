

## 1. why choose cross entropy and why switch to WCE ?
- the task is a classification task and the first standard loss fuction is CE
- the data we are dealing with is ordinal + imbalance
- the CE assumes every class is equally important, to fix that weighted CE is used

## 2. Why added augmentation and specifically what is the reasoning behind each of those augmentations ?
- the task is to force the model to learn underlying structural patterns and not make decisions based on pixel level information 
- the another reason to add augmentation is this model is also going to be tested on different datasets, to capture cross dataset features accurately and to make it robust 

- the normal flow of augmentation is image transformations -> convert to tensor -> normalization
- each transformation is made with a reason that it should force model to not make predictions from pixels _and_ also not lose information as medical data is sensitive 

## 3. Why not use focal loss ?
- ans in  [Findings](Findings.md)

## 4. Why used efficient net model b0 ?
- it's fast and efficient, the b0 version (smallest in the family) is used in the start to quickly test the scope and difficulty of the task 
- it performed better then expected and there was no need to change the model, as the model was not a problem but representation and data
- when testing on other dataset, I will change it to a bigger model when I will be testing other datasets and if it's required
- Training loss reached 0.5 within 5-10 epochs and overfit to near 0.0, indicating the model has sufficient capacity to learn the data. The bottleneck was generalization, not model capacity — confirmed by val loss plateauing while train loss continued dropping. Switching to a larger model would worsen overfitting, not improve generalization

## 5. What's the goal and why ?
- The goal is not just to get high numbers, the goal is to make this system robust enough that if tomorrow this is deployed in a clinic, it should handle the messy real world and be grounded by the things it actually learned and not overconfident
- this is also a part of the reason why I am added different metrics, testing on other datasets, showing where the model is uncertain and optimizing 


## 6. Why used MC dropout ?
- ans in [Findings](Findings.md)

## 7. Why use calibration ?
- 2 reasons: softmax outputs from NNs are systematically overconfident — a prediction of 0.9 doesn't mean 90% accuracy, it's usually lower _and_ in a DR screening system, a confident-but-wrong prediction (e.g. predicting "No DR" at 95% when the patient has Moderate DR) directly delays treatment
- ECE (Expected Calibration Error) is used to measure miscalibration. It bins predictions by confidence, compares predicted confidence to actual accuracy per bin, and produces a single scalar gap. It's the standard metric for calibration in classification
- per-class ECE is calculated because the calibration problems are class-specific: Class 2 (Moderate) has the highest ECE at 0.0768 — consistent with its confusion matrix errors and decision boundary instability from EXP_007/008. Class 0 (No DR) has the lowest at 0.0222 — majority class with distinct visual features. Classes 3/4 have low ECE (0.032, 0.0317) despite being minority classes, because severe DR features (hemorrhages, neovascularization) are visually discriminative. A single aggregate ECE would hide this 3.5x gap between class 2 and class 0
- temperature scaling is used to correct calibration: it divides the logit vector by a single learned parameter T before softmax, softening or sharpening the entire distribution
- T is applied globally (one scalar for all 5 classes) because softmax couples all logit values — changing one logit affects all 5 output probabilities, so per-class correction isn't possible with a single temperature parameter
- per-class correction is not applied because would require vector scaling (5 parameters, one per class) or matrix scaling (25 parameters). With only 733 validation samples, fitting 5+ calibration parameters risks overfitting the calibration correction itself
- the tradeoff is explicit: global T can't fix Class 2's 0.0768 ECE without also shifting Class 0's already-good 0.0222. But one robust parameter on 733 samples is more reliable than five noisy ones

## 8. Why use Mahalanobis distance for OOD detection ?
- after zero-shot testing showed performance degradation across all external datasets, the question was: can we quantify _how far_ each dataset is from APTOS in feature space ?
- uncertainty metrics (entropy, margin, MC std) showed inconsistent behavior — IDRiD triggered high uncertainty, Messidor didn't, despite Messidor performing worse
- Mahalanobis distance measures geometric distance from the training cluster in the 1280-d EfficientNet feature space, accounting for feature correlations via the covariance matrix
- unlike Euclidean distance, Mahalanobis normalizes by the shape of the training distribution — a shift along a high-variance direction is penalized less than a shift along a low-variance direction
- the APTOS validation set baseline (avg distance 48.17) is required as a reference — without it, OOD distances are numbers without meaning

## 9. Why is raw Mahalanobis distance insufficient as an OOD detector ?
- the key finding from EXP_011: distance does not predict performance. IDRiD has the highest avg distance (103.27) but the best QWK (0.62). Messidor has the lowest avg distance (~92.8) but the worst QWK (0.37-0.49)
- reason: the 1280-d feature space entangles two types of features — DR-invariant (vessel density, hemorrhage patterns) and dataset-specific (scanner artifacts, brightness profiles)
- raw distance measures shift across all dimensions equally. It can't distinguish "far because the scanner looks different" from "far because the DR features are corrupted"
- IDRiD is far in scanner-artifact dimensions but DR-invariant features survived → classification works despite high distance
- Messidor is closer overall but the shift specifically targets DR-relevant dimensions → classification breaks despite lower distance
- this is directional feature corruption: the magnitude of shift matters less than its direction relative to the model's decision boundaries
- implication: reliable OOD detection requires either (a) per-class distance to capture directional information, or (b) domain adaptation (DANN) to disentangle invariant from dataset-specific features before computing distance

## 10. Why use Cosine Similarity for feature analysis after Mahalanobis distance?
- Mahalanobis distance showed us that feature space shift happens, but it could not differentiate between benign shift (IDRiD) and malignant shift (Messidor). It only measured the *magnitude* of the shift, not the *direction*.
- additionally, calculating a *per-class* Mahalanobis distance failed due to the curse of dimensionality. The feature space is 1280-D, but some classes have fewer than 50-100 samples.
- Mahalanobis distance requires computing the inverse of the covariance matrix. When the number of samples is less than the number of dimensions (N < D), the covariance matrix is singular (non-invertible) and statistically impossible to measure accurately. This caused the per-class Mahalanobis distances to explode into the 1000s.
- using a global centroid also didn't work for class-specific directional analysis. Thus, switching to Cosine similarity became a statistical necessity to handle high-dimensional, low-sample-size clusters without needing a covariance matrix.
- we needed a metric to measure direction: if a sample from a middle class (like Moderate DR) is shifted, is it shifted randomly due to scanner noise, or is it specifically pulled toward another class centroid?
- Cosine similarity measures the angle between vectors, ignoring magnitude. By measuring the cosine similarity of OOD samples to the APTOS training centroids, we can see exactly where the features are pointing.
- this allowed us to confirm the "Directional Feature Corruption" hypothesis: on Messidor, the features for all classes are directionally pulled toward the Class 0 (No DR) centroid (avg similarity **0.5658**), while severe classes collapse near zero (Class 4 at **-0.0088**), causing silent, confident misclassifications.
- this also explained the "Two-Party System" in DDR and EyePACS, where features are pulled toward the extremes (Class 0 and Class 4) because those classes have robust, visually distinct signatures. Middle classes (1, 2, 3) lose their directional identity and are absorbed by the extremes. For IDRiD on the other hand, the shift was purely benign with stable alignment across all pathological classes (**~0.32-0.44** avg similarity).
