this # Experiment 012: Cosine Similarity for Feature Direction Analysis

## Goal
To investigate whether the poor performance on external datasets like Messidor (which showed a surprisingly low Mahalanobis distance in EXP 011) is due to a directional shift in feature space. We analyze the cosine similarity between the extracted features of out-of-distribution (OOD) test sets and the class centroids computed from the APTOS training set.

## Why Cosine Similarity over Mahalanobis Distance?
An initial attempt was made to measure directional shifts using *per-class* Mahalanobis distances. However, this failed due to a fundamental statistical limitation:
- **Curse of Dimensionality:** The EfficientNet feature space is 1280-D, while some of our classes have fewer than 50 or 100 samples. 
- **Singular Covariance Matrix:** Mahalanobis distance relies on computing the inverse of the covariance matrix. When the number of samples (N) is less than the number of dimensions (D), the covariance matrix is singular (non-invertible) and cannot be accurately estimated without heavy regularization. 
- **Distance Explosion:** As a result of these unstable calculations, the per-class Mahalanobis distances exploded into the 1000s. Using a global centroid also failed to capture class-specific directional shifts properly.
- **The Solution:** Cosine similarity measures the angle between vectors directly, bypassing the need for a covariance matrix altogether. This makes it a robust necessity for directional analysis in high-dimensional, low-sample-size scenarios.

## Hypothesis
- **H1 (Directional Feature Corruption):** The distribution shift in datasets like Messidor corrupts the direction of the features, pulling them toward a specific class (e.g., Class 0), which explains the high confident-and-wrong predictions.
- **H2 (Middle Class Ambiguity):** The model struggles to separate adjacent middle classes (1, 2, 3) because they form overlapping clusters due to smooth disease progression, whereas the extremes (0 and 4) are clearly distinguishable.

## Observations (from Heatmap Analysis)
1. **Three Distinct Groupings**: The datasets form three distinct behavioral groups. DDR and EyePACS form one group; all Messidor groups form another; and IDRiD behaves as a completely separate entity.
2. **Two-Party System (DDR and EyePACS)**: The model performs decently around Class 0 (No DR) and Class 4 (Proliferative DR). These act as two dominant "attractors" in feature space, while the middle classes (1, 2, 3) lose their distinct identity and get pulled toward either extreme. For example, in DDR-China, Class 0 has a high average cosine similarity of **0.5294** (max **0.8731**), while Classes 2, 3, 4 drop significantly in average similarity (**0.0962, 0.0353, 0.0599** respectively).
3. **Single Ruler (Messidor)**: In Messidor, there is a single dominant attractor—everything points toward Class 0. This explains the "silent failure" where the model confidently misclassifies middle and severe cases as No DR, because there is no competing signal from other classes. For Messidor-Grp2, Class 0 average similarity is a massive **0.5658**, whereas Class 1 is **0.1173**, Class 2 is **0.0589**, Class 3 is **0.0208**, and Class 4 is **-0.0088**. 
4. **Balanced Alignment (IDRiD)**: IDRiD is the most balanced cluster. Every class has reasonably high similarity to its own centroid. The cluster is denser and broader than the first group. Its average cosine similarity across pathological classes is robust: Class 1 (**0.4058**), Class 2 (**0.4397**), Class 3 (**0.3253**), and Class 4 (**0.3421**).
5. **Class Imbalance Impact**: The lack of severe cases (Class 3) in Messidor contributes to the problem, as that region of feature space is unanchored, though it's not the primary cause of the collapse.
6. **Extremes Distinguishability**: Classes 0 and 4 are extreme cases and clearly distinguishable (clean retina vs. extensive damage). For middle classes, there is a smooth progression curve, making it harder for the model to draw a stable decision boundary.

## Chat / Analysis Response

**What's Solid:**
- **Three groups** is correct and meaningful. IDRiD behaves fundamentally differently from everything else. The fact that it forms its own group despite being geographically closest to APTOS is a key finding.
- **DDR/EyePACS two-party system** is accurate. The feature space is pulled between Class 0 and Class 4, with Classes 1, 2, 3 losing their distinct identity. This confirms that DR-invariant features surviving distribution shift are the extreme ones.
- **Messidor single ruler** is correct and is the strongest evidence of directional shift. Everything points toward Class 0, explaining the lack of uncertainty.

**Where to Push Back (Refinements):**
- **Messidor's collapse** isn't primarily due to the missing Class 3. The real driver is two-fold: the Class 0 centroid in APTOS is dominant (1805 training samples), and Messidor's scanner produces images that inherently land closer to the APTOS Class 0 region. The missing Class 3 exacerbates this but isn't the root cause.
- **Middle class struggle** is two phenomena: 
  1. On APTOS itself, it's genuine clinical ambiguity (smooth progression). 
  2. On external datasets, it's directional shift. Middle classes on DDR/Messidor don't just spread proportionally between extremes; they collapse specifically toward Class 0. The mild/moderate DR features from external scanners look like healthy retinas to the model.

**The IDRiD Anomaly (Key Insight):**
IDRiD's heatmap is balanced. This means IDRiD's DR features transferred successfully to APTOS's feature space. The model's learned representations for each severity grade are directionally aligned with IDRiD images. The distribution shift here is in scanner-artifact dimensions (high Mahalanobis distance), but NOT in DR-feature dimensions (balanced cosine similarity). 
This contrasts cleanly with Messidor: Messidor is geometrically closer (lower Mahalanobis) but directionally collapsed (cosine similarity dominated by Class 0). This separates benign vs. malignant shift clearly.

## Conclusion / Refined Hypotheses
- **H1 Confirmed (Directional Feature Corruption):** Cosine similarity heatmaps show that for DDR, EyePACS, and Messidor, images from non-Class-0 true labels have their highest cosine similarity pointing toward Class 0. The features are genuinely in the wrong direction, leading to silent failures.
- **Severity determines feature robustness:** Class 0 and Class 4 have distinct visual signatures that survive distribution shift. Classes 1, 2, and 3 have ambiguous features that lose directional identity under shift and get absorbed into the majority class (Class 0).
- **IDRiD as a Control:** It is the only dataset where DR features transferred per-class, explaining why its uncertainty correctly increased and Certain+Wrong dropped. Directionally correct features allow internal uncertainty signals to work properly, whereas directionally corrupted features (Messidor) cause them to fail.
