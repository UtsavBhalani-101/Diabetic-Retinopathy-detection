# pipeline/predict.py
# ============================================================
# Standalone inference layer.
# Input: retinal image path
# Output: prediction, confidence, triage label, uncertainty metrics
# ============================================================

import os
import sys
import json
import logging
import argparse
from PIL import Image

import numpy as np
import torch
import torch.nn as nn

# Make the project root importable when this package is run from any CWD
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.config      import (
    UNCERTAINTY_ENTROPY_THRESHOLD,
    UNCERTAINTY_MARGIN_THRESHOLD,
    UNCERTAINTY_MC_STD_THRESHOLD
)
from pipeline.dataset     import val_transformer, setting_gpu
from pipeline.model       import EfficientNetMC
from pipeline.calibration import apply_temperature, triage_sample

# Configure logging to stderr so stdout contains only clean structured JSON
logger = logging.getLogger("pipeline.predict")


def predict_image(image_path: str,
                  model_path: str = "artifacts/weights/aptos_efficientnet.pth",
                  optimal_T_path: str = "artifacts/calibration/optimal_T.npy",
                  mc_passes: int = 30,
                  dropout_rate: float = 0.3,
                  num_classes: int = 5) -> dict:
    """
    Perform Bayesian stochastic (MC Dropout) inference on a single retinal image.

    Parameters
    ----------
    image_path : str
        Path to the input retinal image.
    model_path : str
        Path to the saved PyTorch model weights.
    optimal_T_path : str
        Path to the saved calibration temperature numpy file.
    mc_passes : int
        Number of stochastic forward passes for MC Dropout (default: 30).
    dropout_rate : float
        Dropout rate to instantiate the model classifier head (default: 0.3).
    num_classes : int
        Number of severity categories (default: 5).

    Returns
    -------
    result : dict
        Dictionary containing class prediction, confidence, triage label, and uncertainty signals.
    """
    # 1. Verify files exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Retinal image path not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}. "
                                f"Please train a model first or provide the correct path.")

    # 2. Setup device & load image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open/parse image file: {image_path}. Details: {e}")

    # Preprocess
    img_tensor = val_transformer(pil_img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # 3. Load model weights
    logger.info(f"Initializing EfficientNetMC model with dropout_rate={dropout_rate}")
    model = EfficientNetMC(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=False
    )
    
    # Load state dict
    logger.info(f"Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # 4. Activate Stochastic MC Dropout at inference time
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()  # Force dropout layers to remain active

    logger.info(f"Running stochastic MC Dropout | T={mc_passes} passes...")
    
    passes = []
    logit_passes = []

    with torch.no_grad():
        for _ in range(mc_passes):
            logits = model(img_tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            passes.append(probs)
            logit_passes.append(logits.cpu().numpy())

    passes = np.array(passes)        # [T, 1, C]
    logit_passes = np.array(logit_passes)  # [T, 1, C]

    # Average metrics over passes
    mean_probs = passes.mean(axis=0)[0]       # [C]
    mean_logits = logit_passes.mean(axis=0)   # [1, C]
    mc_std = passes.std(axis=0)[0].mean()     # Mean std across classes (scalar)

    # 5. Load calibration optimal temperature T
    optimal_T = 1.0
    if os.path.exists(optimal_T_path):
        try:
            optimal_T = float(np.load(optimal_T_path))
            logger.info(f"Loaded optimal temperature T={optimal_T:.4f} from calibration")
        except Exception as e:
            logger.warning(f"Failed to load optimal_T from {optimal_T_path} (falling back to T=1.0). Details: {e}")
    else:
        logger.warning(f"Optimal temperature file not found at {optimal_T_path} (defaulting to uncalibrated T=1.0)")

    # 6. Apply temperature scaling to logits
    calibrated_probs = apply_temperature(mean_logits, optimal_T)[0]  # [C]

    # 7. Compute uncertainty signals
    eps = 1e-8
    entropy = float(-np.sum(calibrated_probs * np.log(calibrated_probs + eps)))
    
    sorted_probs = np.sort(calibrated_probs)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1])

    # Class mappings
    DR_CLASSES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
    pred_idx = int(np.argmax(calibrated_probs))
    pred_label = DR_CLASSES[pred_idx]
    confidence = float(calibrated_probs[pred_idx])

    # 8. Apply formalized triage logic
    triage = triage_sample(
        pred=pred_idx,
        entropy=entropy,
        margin=margin,
        mc_std=mc_std,
        entropy_thresh=UNCERTAINTY_ENTROPY_THRESHOLD,
        margin_thresh=UNCERTAINTY_MARGIN_THRESHOLD,
        mc_std_thresh=UNCERTAINTY_MC_STD_THRESHOLD
    )

    return {
        "prediction": pred_label,
        "confidence": round(confidence, 4),
        "triage": triage,
        "entropy": round(entropy, 4),
        "margin": round(margin, 4),
        "mc_std": round(float(mc_std), 6)
    }


def main():
    parser = argparse.ArgumentParser(description="Standalone Diabetic Retinopathy Inference Script")
    parser.add_argument("image_path", type=str, help="Path to the retinal image to analyze")
    parser.add_argument("--model_path", type=str, default="artifacts/weights/aptos_efficientnet.pth",
                        help="Path to trained .pth model weights")
    parser.add_argument("--optimal_T_path", type=str, default="artifacts/calibration/optimal_T.npy",
                        help="Path to optimal temperature .npy scalar file")
    parser.add_argument("--mc_passes", type=int, default=30, help="Number of MC Dropout passes (default: 30)")
    parser.add_argument("--verbose", action="store_true", help="Print logging messages to stderr")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        stream=sys.stderr,
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    try:
        result = predict_image(
            image_path=args.image_path,
            model_path=args.model_path,
            optimal_T_path=args.optimal_T_path,
            mc_passes=args.mc_passes
        )
        # Print clean, formatted JSON to stdout
        print(json.dumps(result, indent=4))
        sys.exit(0)
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}, indent=4), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Inference failed: {str(e)}"}, indent=4), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
