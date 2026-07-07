# pipeline/dann/dataset_dann.py
# ============================================================
# DANNDataset — domain-aware dataset wrapper.
#
# Wraps the existing RetinopathyDataset (read-only, no changes)
# and appends a domain_id integer to every sample:
#   domain_id = 0  →  source  (APTOS, EyePACS, Messidor-Grp1, DDR-China)
#   domain_id = 1  →  target  (Messidor-Grp2, Messidor-Grp3)
#
# For target datasets: class_label is still loaded from the CSV so
# post-training evaluation (QWK) works — but it is NOT fed into
# the classification loss during DANN training.
# ============================================================

import logging

from torch.utils.data import Dataset

from pipeline.data.dataset import RetinopathyDataset  # read-only import

logger = logging.getLogger(__name__)


class DANNDataset(Dataset):
    """
    Wraps a RetinopathyDataset and adds a domain label.

    Parameters
    ----------
    base_dataset : RetinopathyDataset
        An already-constructed RetinopathyDataset instance.
    domain_id    : int
        0 for source domains, 1 for target domains.

    Returns (per __getitem__)
    -------
    image        : Tensor [C, H, W]
    class_label  : int    — DR grade 0-4
    domain_id    : int    — 0 (source) or 1 (target)
    """

    def __init__(self, base_dataset: RetinopathyDataset, domain_id: int):
        if domain_id not in (0, 1):
            raise ValueError(f"domain_id must be 0 (source) or 1 (target), got {domain_id}")

        self.base      = base_dataset
        self.domain_id = domain_id

        logger.info(
            f"DANNDataset | domain_id={domain_id} "
            f"({'source' if domain_id == 0 else 'target'}) "
            f"| samples={len(base_dataset)}"
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, class_label = self.base[idx]
        return image, class_label, self.domain_id
