# pipeline/dann/run_dann.py
# ============================================================
# Entry point for the DANN pipeline.
#
# Usage (from project root):
#   python -m pipeline.dann.run_dann
# ============================================================

import logging
import os

import wandb
from dotenv import load_dotenv

from pipeline.setup.config import setup_logging          # read-only
from pipeline.dann.config_dann import DANN_CONFIG
from pipeline.dann.train_dann import train_dann

logger = logging.getLogger(__name__)


def setup_wandb() -> None:
    """
    Load .env and authenticate wandb using the API key.
    Defined locally to keep pipeline/dann/ fully self-contained.
    Falls back to interactive login if no key is found.
    """
    load_dotenv()
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
        logger.info("wandb authenticated via WANDB_API_KEY from .env")
    else:
        logger.warning("WANDB_API_KEY not found — falling back to interactive login")
        wandb.login()


def orchestrate_dann(config: dict | None = None) -> None:
    """
    Run the full DANN pipeline.

    Parameters
    ----------
    config : optional override dict; defaults to DANN_CONFIG.
             Only keys you specify are overridden — the rest keep defaults.
             Example: orchestrate_dann({"dann_epochs": 30, "lr": 5e-5})
    """
    cfg = {**DANN_CONFIG, **(config or {})}

    logger.info("=" * 60)
    logger.info("DANN Pipeline — starting")
    logger.info(f"Source datasets : {cfg['source_datasets']}")
    logger.info(f"Target datasets : {cfg['target_datasets']}")
    logger.info("=" * 60)

    optimal_T = train_dann(cfg)

    logger.info("=" * 60)
    logger.info(f"DANN Pipeline complete | optimal_T={optimal_T:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_logging()          # timestamps → console + artifacts/logs/
    setup_wandb()            # authenticate wandb from .env
    orchestrate_dann()
