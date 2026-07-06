
## Experiment: GradCAM++ Visual Attribution Analysis on IDRiD

### Goal
Determine what spatial regions the model relies on when making predictions on IDRiD — specifically, whether the model attends to clinically meaningful lesion regions (microaneurysms, hemorrhages, exudates, neovascularization) or whether it relies on non-specific, diffuse regional signal. This was motivated by the need to understand the *mechanism* behind cross-dataset failure, not just its magnitude (already established via QWK, cosine similarity, and Mahalanobis distance in prior experiments).

### Why
Prior experiments (cosine similarity, CLAHE vs. Ben Graham) established *that* the model entangles DR-invariant and dataset-specific features, and provided indirect evidence of texture bias (BG destroys texture → model collapses; CLAHE preserves texture → model recovers). GradCAM/GradCAM++ was applied as an attempt to get direct, spatial, visual evidence of this — to see *where* the model is looking, as a complement to the purely numerical evidence already in hand.

### Hypothesis
H1: If the model has learned genuine, DR-invariant lesion features, correct predictions should show GradCAM activation concentrated on clinically plausible lesion regions (microaneurysms, hemorrhages, exudates, neovascularization sites), and this pattern should be visually distinguishable from incorrect predictions, where activation would be expected to fall on non-lesion or anatomically irrelevant regions.

### Result
Standard GradCAM produced diffuse, poorly localized activation maps — large, low-resolution blobs with no clear correspondence to specific anatomical structures. This was expected given EfficientNet-B0's coarse final feature map resolution (~7x7 before upsampling) and is a known limitation of gradient-based CAM methods on this architecture.

GradCAM++ produced visually sharper, more stable heatmaps compared to standard GradCAM. However, on manual review of both correct and incorrect predictions across multiple IDRiD samples (Moderate, Severe, and Proliferative classes), no reliable visual distinction emerged between activation patterns for correct vs. incorrect predictions. Both categories showed similarly large, centrally-located warm regions in the posterior pole area, and the failure cases could not be distinguished from success cases by visual inspection alone.

### Observations
- At 224x224 resolution with a 7x7-equivalent feature map, GradCAM++ activation regions are inherently coarse — the spatial precision needed to isolate microaneurysms or fine hemorrhage patterns is not achievable at this resolution regardless of which CAM variant is used.
- Manual visual review is not a reliable method for validating what the model attends to at this resolution — both success and failure cases were visually indistinguishable, which is itself informative but not because it reveals a specific mechanism.
- GradCAM++ output shows *that* gradient signal concentrates in the central/posterior region across nearly all samples, but this does not establish *causal* dependence on that region — gradient concentration can reflect high local contrast (e.g., near the optic disc) rather than decision-relevant content.
- I want to flag directly: an earlier attempt to produce a written GradCAM analysis (treating the heatmaps as evidence and narrating specific per-image findings) turned out to be circular — it described patterns that matched pre-existing hypotheses from other experiments rather than deriving anything new from the images, and made specific anatomical claims that the image resolution could not actually support. That analysis should not be cited or used as evidence in the writeup.

### Conclusion / Refined Hypothesis
GradCAM++ visual inspection, on its own, is not a discriminative or verifiable method at this image resolution and model architecture — it could not confirm or refute H1. The experiment's actual value was in establishing this limitation and motivating a follow-up: rather than relying on visual interpretation of *where* gradient signal concentrates, test *whether* that region is causally necessary for the prediction, via occlusion sensitivity. H1 remains untested by this experiment; it is neither confirmed nor refuted. The refined hypothesis carried forward is procedural: causal attribution requires an interventional test (occlusion), not a correlational/visual one (GradCAM).

---

One thing I'm not certain about and you should decide: whether to include this experiment in your writeup at all, given its conclusion is essentially "this method failed to produce usable evidence." That's a legitimate and honest thing to report — negative/inconclusive results with a clear methodological reason are real science — but it's a judgment call whether MLSS reviewers want to see a documented dead-end or whether you'd rather fold the one-paragraph version of this into the occlusion writeup as motivation, without giving it full experiment status. Your call — I don't have visibility into how the rest of your report is structured or how much space you're working with.

Want me to draft the occlusion sensitivity writeup next, using the same format and the actual numbers from your two log files (targeted vs. random control)?