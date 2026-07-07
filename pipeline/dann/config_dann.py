# pipeline/dann/config_dann.py
# ============================================================
# DANN_CONFIG — self-contained configuration for the DANN pipeline.
#
# Extends BASE_CONFIG (read-only import) with DANN-specific fields.
# All artifact paths are under artifacts/dann/ to avoid collision
# with the baseline pipeline's artifacts/.
# ============================================================

from pipeline.setup.config import BASE_CONFIG  # read-only import

DANN_CONFIG: dict = {
    # ---- inherit all baseline hyperparameters ----
    **BASE_CONFIG,

    # ---- project identity ----
    "project_name": "aptos-dann-adaptation",

    # ---- artifact paths (isolated from baseline pipeline) ----
    "model_save_path":       "artifacts/dann/weights/dann_efficientnet.pth",
    "optimal_T_save_path":   "artifacts/dann/calibration/optimal_T.npy",
    "calib_plot_train_path": "artifacts/dann/calibration/plots/calibration_dann.png",
    "class_centroids_save_path": "artifacts/dann/centroids/mean.npy",

    # ---- dataset split ----
    # Source: 4 labelled datasets — DR grade used in classification loss
    "source_datasets": [
        "APTOS_2019",
        "EyePACS-Resized",
        "Messidor-Grp1",
        "DDR-China",
    ],
    # Target: 3 unlabelled datasets — only used for domain alignment + post-eval
    # IDRiD added: different scanner/protocol/population → harder OOD target
    "target_datasets": [
        "Messidor-Grp2",
        "Messidor-Grp3",
        "IDRiD",
    ],

    # ---- DANN-specific hyperparameters ----
    "dann_epochs":     20,       # DANN may need more epochs than baseline (10)
    "dann_lambda_max": 1.0,      # ceiling value of the λ schedule
    "domain_hidden_dim": 512,    # domain discriminator hidden layer size
    "domain_dropout":    0.3,    # dropout inside the domain discriminator

    # ---- override batch size slightly larger for combined source ----
    # (more diverse batches → more stable domain adversarial signal)
    "batch_size": 32,

    # ---- evaluation ----
    "test_max_samples": None,    # use full target datasets at eval time
}
