# pipeline/data/gpu_transforms.py
# ============================================================
# This module was created for a GPU-in-the-loop CLAHE approach
# that was superseded by the offline preprocessing workflow in
# pipeline/dann/preprocess_clahe.py.
#
# The offline approach (run CLAHE once, save to disk, load preprocessed
# images during training) was chosen because it:
#   - Keeps the DataLoader transform pipeline simple
#   - Allows saving/mounting preprocessed images as a Kaggle Dataset
#   - Avoids touching the training loop for CLAHE logic
#
# This file is intentionally empty. It can be removed safely.
# ============================================================
