## Experiment: Occlusion Sensitivity — Targeted vs. Random (IDRiD Test)

### Goal
Test whether the regions GradCAM++ identifies as high-activation are causally necessary for the model's predictions, as opposed to merely correlated with gradient flow. GradCAM++ showed diffuse activation with no visually discernible difference between correct and incorrect predictions, so this experiment replaces visual interpretation with a direct, interventional test: does removing the region change the outcome, and does it change it more than removing an arbitrary region of the same size.

### Why
Attribution methods like GradCAM/GradCAM++ show where gradient signal concentrates, not what the model depends on. High-contrast anatomical structures (e.g., the optic disc) can attract gradient signal without being decision-relevant. The only way to test necessity is intervention — delete the region, rerun the model, measure the change. A targeted-only test is insufficient on its own: any sufficiently large deletion will degrade predictions, so a random-region control at matched occlusion size is required to determine whether the GradCAM-identified region is special or whether the drop is just a function of how much of the image was removed.

### Hypothesis
H1: If the model depends specifically on the regions GradCAM++ highlights, occluding those regions should degrade performance (QWK, mean confidence) more than occluding a random region of equal size.
H0 (null): If GradCAM++ is not identifying decision-relevant regions, targeted and random occlusion should produce similar or GradCAM-favoring-worse degradation — i.e., random occlusion should not outperform targeted occlusion in preserving model performance.

### Result

| Condition | QWK | Mean Confidence | Mean Prob Drop vs. Baseline | % Images w/ Positive Drop |
|---|---|---|---|---|
| Baseline (no occlusion) | 0.6211 | 0.5901 | — | — |
| GradCAM++ top 10% occluded | 0.5425 | 0.4797 | 0.1104 | 74.8% |
| Random 10% occluded | 0.3434 | 0.4452 | 0.1449 | 76.7% |
| GradCAM++ top 30% occluded | 0.3311 | 0.4274 | 0.1627 | 78.6% |
| Random 30% occluded | 0.1579 | 0.3377 | 0.2524 | 95.1% |

At both occlusion sizes, random occlusion degraded the model more than targeted occlusion. At 10%, random occlusion dropped QWK to 0.34 vs. 0.54 for targeted. At 30%, random occlusion dropped QWK to 0.16 vs. 0.33 for targeted. Mean probability drop was also larger under random occlusion at both sizes.

### Observations
- This result is the opposite direction from H1. If the model depended specifically on the GradCAM++-identified region, targeted occlusion should have hurt more than random occlusion of equal size. It didn't.
- I'm not certain of the exact mechanism, but the straightforward reading is that the GradCAM++ hot region is not where the model's decision-relevant signal is concentrated — removing it costs less than removing an arbitrary same-sized region elsewhere, meaning the rest of the image carries more average decision-relevant information than the highlighted region does.
- This is consistent with, and adds interventional evidence to, the diffuse/non-lesion-specific reliance already suggested by the CLAHE-vs-Ben-Graham texture bias finding and by the visual indistinguishability of success/failure GradCAM++ maps.
- Flagging a data quality concern before this gets treated as settled: the random-30% confusion matrix shows a degenerate collapse — Class 4 accuracy is 100% (13/13) while every other class is at or near 0%, with the large majority of predictions across all true classes landing in Class 4. This is not "confused, spread-out errors" — it's the model defaulting to one class under heavy occlusion. This could mean the model behaves in a specific, informative way when most of the image is destroyed (e.g., falls back on some prior or default), or it could be an artifact of this single random seed's occlusion pattern coincidentally removing informative regions from most images. I have not run additional seeds. Until that's checked, I would not treat the 30%-random result as fully stable — the 10%-random result, which doesn't show this degenerate pattern, is on firmer ground.
- Sample size caveat: this is n=103 for the full test set, but per-class breakdowns (e.g., Class 1 with only 5 samples) are too small to draw reliable per-class conclusions from any single condition.

### Conclusion / Refined Hypothesis
H1 is not supported by this experiment — targeted occlusion of GradCAM++-identified regions consistently preserved more model performance than random occlusion of the same size, at both 10% and 30% thresholds. This indicates GradCAM++, in this setup, is not identifying the regions the model actually depends on most; whatever signal the model uses is distributed more broadly across the image than the attribution method suggests.

This strengthens, via a different and independent method, the conclusion already suggested by the CLAHE/Ben Graham texture-bias finding: the model's decision-relevant signal is diffuse rather than lesion-localized. It does not, on its own, identify what that diffuse signal actually consists of (texture statistics, illumination profile, global contrast, or something else) — that remains an open question this experiment does not answer.

Before treating the 30% result as a stable finding, the random-occlusion condition should be re-run with at least 2-3 different seeds to confirm the Class 4 collapse isn't a single-seed artifact. I have not done this yet, so I'd hold that specific number as provisional.

---

One clarification I should make explicit: I don't have independent verification of the pipeline internals (how exactly `OccludedDataset` selects the top-k threshold, how `RandomOccludedDataset` samples regions, whether the "top 10%"/"top 30%" are computed per-image or globally, whether the random regions are contiguous blocks or scattered pixels matching the GradCAM mask's shape). I'm taking the log output at face value. If the random occlusion isn't shape-matched to the GradCAM mask (e.g., random scattered pixels vs. a single contiguous blob of equivalent area), that's a methodological detail worth confirming and noting in the writeup, since it could affect how directly comparable the two conditions are.