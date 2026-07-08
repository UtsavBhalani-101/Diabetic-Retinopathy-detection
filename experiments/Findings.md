

### 1. why did qwk stayed robust and loss so low ?
- the data we are dealing with is ordinal + imbalance
- Ordinal:
- diseases have natural progression
- CNN is made for detecting temporal structure and relationships
- So because of that model was able to capture the order of classes fairly well
- It formed overlapping clusters and the mistakes were mostly adjacent classes 

- Imbalance:
- This is related to not being sample enough
- so use frequency based weights to fix this
- it didn't changed the numbers a lot but the confusion matrix is much better and accurate 

### 2. why focal loss failed ?
- it forces the model to focus on hard samples rather then easy ones
- this is not inheriently in bad direction but the hard samples here means noisy, blurry, contradictory images with label noise
- it's not necessary that focusing on the hard samples will improve the performance and learning. These hard samples might not be informative at all
- so the numbers stayed similar
- but when added augmentation to this, focal loss made things significantly worse (QWK dropped to 0.76, the lowest in the entire series). Augmentation creates more varied and harder samples, and focal loss doubles down on exactly those hard samples — which in this dataset are noisy, clinically ambiguous, and uninformative. The two compound each other's worst tendencies: augmentation makes more hard samples, focal loss forces the model to obsess over them

### 3. why added augmentation ?
- the confusion matrix showed a consistent behavior of error throughout the experiments - the error were mostly in _**adjacent classes**_
- for the adjacent classes:
- it shows that the model is getting confused and can't distinguish between adjacent classes
- it's happening because the model has not seen enough independent samples of minority classes, hence defaulting to class 2 
- augmentation forces the model to not  learn more deep, structural features that distinguishes the classes 
- it resulted in improved QWK score as shown in 7th exp and better discriminative info to distinguish classes


### 4. why add MC dropout ?
- after augmentation, the confusion matrix had another problem, the errors were _**bi directional on middle classes**_
- why does that matter ?
- A model with a stable decision boundary outputs something like [0.01, 0.02, 0.94, 0.02, 0.01]. Confident, narrow peak.
- A model that's genuinely uncertain on a class 2 image outputs something like [0.05, 0.28, 0.35, 0.22, 0.10]. Spread across multiple classes, no clear peak.
- The argmax of both is class 2. the confusion matrix treats them identically. But these are completely different situations.
- this is what's happening, we can't distinguish that either the model is 94% confident or 51%, both resulting in same output
- and also the problem domain is where confident wrong predictions are worse then uncertain correct ones
- this is where MC dropout is used
- it surfaces the uncertainty for the predictions made 
- So MC Dropout is not just useful because wrong predictions are costly. It's useful because your confusion matrix is showing you that a specific region of your prediction space, classes 1, 2, 3 has genuinely unstable decision boundaries. MC Dropout quantifies that instability per prediction.
- Bidirectional errors → model has unstable decision boundary for middle classes → some predictions in that region are genuinely uncertain → MC Dropout surfaces which specific predictions are uncertain
- Reason 1 — Regularization during training. This was the accidental benefit you discovered. Dropout forced redundant feature learning, reduced co-adaptation, val loss dropped from 1.6 to 0.8. This had nothing to do with context or uncertainty — it just made the model better.
- Reason 2 — Surfacing fragile confidence. This is the context-driven reason. Entropy and margin read a single forward pass. They miss the case where a prediction looks confident but flips under perturbation. MC std catches that specific failure mode. In a clinical context a confident-but-fragile prediction is dangerous so you need a signal that detects it.


### 5. Why used different proxies ?
- the current proxies like train loss, val loss, QWK score and confusion matrix hides uncertainity in the model predictions
- it's just argmax of the result and there's no way to tell if the model is confidently wrong or just barely wrong 
- we use entropy and margin to surface that


### 6. Why used calibration ?
- a generally good practice in classification problem _and_ also the task demands it even more, a DR system that is not calibrated is dangerous 


### 7. Why not applying Calibration per class ?

- different classes have different calibration needs (e.g., Class 2 ECE is 0.077, Class 0 is 0.022), which theoretically require individual corrections.
- however, temperature scaling applies a single global temperature T to the logit vector; because softmax couples all classes together, adjusting one logit affects all other probabilities.
- class-specific calibration (like vector scaling with 5 parameters or matrix scaling with 25 parameters) can overfit on a relatively small validation set link in this case (733 samples).
- global temperature scaling is robust, average calibration that generalizes well without overfitting. 
- It's not the best but for the given situation it's good enough


### 8. Why doesn't Mahalanobis distance correlate with QWK ?
- Mahalanobis distance in the 1280-d feature space confirmed all OOD datasets are ~2x the APTOS baseline distance (avg 92-107 vs 48). But the ranking contradicts expectations:
  - IDRiD: highest distance (103.27) → best QWK (0.62)
  - Messidor: lowest distance (~92.8) → worst QWK (0.37-0.49)
- the feature space entangles two types of information: DR-invariant features (lesion morphology, vessel patterns) and dataset-specific features (scanner color profiles, brightness, resolution artifacts)
- distance measures shift across all 1280 dimensions equally — it cannot distinguish a benign shift (scanner artifacts differ but DR features transfer) from a malignant shift (DR-relevant dimensions are corrupted)
- IDRiD's shift is large but in scanner-artifact dimensions → class separations survive → model is uncertain (0.55 uncertain fraction) but still classifies correctly
- Messidor's shift is smaller overall but specifically targets DR-relevant dimensions → Proliferative DR features land in the healthy retina region → model is confident (0.21 uncertain fraction) and wrong (30-36% Certain+Wrong)
- this is the same pattern from the zero-shot testing (TEST_001, TEST_002) but now with a geometric explanation: **the type/direction of distribution shift matters more than its magnitude**
- connects directly to DANN: domain adaptation would penalize the encoder for learning dataset-specific features, leaving only DR-invariant features. Mahalanobis distance computed on DANN features would measure shift in a DR-relevant subspace, making the distance → performance correlation meaningful
- another hypothesis is that the model is learning invariant features and dataset specific features, the distance in dataset specific features is high causing the model to be uncertain but the model still classifies correctly because it has learned the invariant features so the distance and performance are not correlated
- but in Messidor, the distance in dataset specific features is low

### 9. How does Cosine Similarity explain directional feature corruption?
- While Mahalanobis distance measures the magnitude of shift, Cosine similarity measures direction.
- **Messidor (Malignant Shift):** Features for non-Class-0 cases directionally collapse toward the Class 0 centroid. This is why Messidor has a low Mahalanobis distance but terrible QWK. The scanner artifacts cause mild/moderate DR features to look like healthy retinas to the model, leading to confident and wrong predictions. (e.g., Messidor-Grp2 Class 0 avg similarity is **0.5658**, while Class 4 is **-0.0088**).
- **IDRiD (Benign Shift):** Cosine similarity is balanced across classes, meaning the DR-specific features transferred well. The high Mahalanobis distance comes from scanner artifact differences, which increases uncertainty but preserves correct classification. (e.g., IDRiD Class 1, 2, 3, 4 similarities average **0.40**, **0.43**, **0.32**, and **0.34** respectively).
- **DDR & EyePACS:** These datasets exhibit a "two-party system" where features are pulled toward the extremes (Class 0 and Class 4) because those classes have robust, visually distinct signatures. Middle classes (1, 2, 3) lose their directional identity and are absorbed by the extremes.
- This confirms that **severity determines feature robustness across domains**. The model's representations for extreme cases survive distribution shift better than those for the ambiguous middle classes.

### 10. Why CLAHE over Ben Graham preprocessing?
- The Ben Graham preprocessing method, which relies on heavy blurring and subtraction, was found to be adversarial and destructive to the mid-to-low frequency shape features (like large hemorrhages) that the model relies on. 
- It caused a performance collapse on external datasets (like DDR and Messidor) because it altered the underlying texture profile the EfficientNet-B0 model was heavily relying on (revealing an ImageNet texture bias).
- CLAHE equalizes contrast without destroying these spatial frequencies. Switching to CLAHE largely preserved performance on external datasets (e.g., DDR QWK jumped from 0.40 to 0.58) and fixed the "Silent Collapse" failure mode where diseased eyes were incorrectly, but confidently, predicted as Class 0 (No DR).

### 11. Are the decision-relevant regions localized to clinical lesions? (GradCAM & Occlusion findings)
- GradCAM++ produced diffuse, low-resolution activation heatmaps due to the coarse 7x7 final feature map of EfficientNet-B0. It could not visually distinguish correct from incorrect predictions.
- To test causal necessity, an Occlusion Sensitivity experiment was conducted (targeted occlusion of GradCAM hot-spots vs. random occlusion of equal size). 
- The result showed that random occlusion degraded performance *more* than targeted occlusion (e.g., at 10% occlusion, QWK dropped to 0.34 with random vs. 0.54 with targeted). This proved that the model's decision-relevant signal is diffuse rather than lesion-localized, strengthening the hypothesis that the model relies on broad signals (like texture or global contrast) rather than strictly localized clinical lesions.
