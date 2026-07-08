# api/app.py
# ============================================================
# FastAPI deployment layer for the APTOS DR inference pipeline.
#
# Endpoints:
#   POST /predict       — upload retinal image, returns triage JSON
#   GET  /health        — returns {"status": "ok"}
#   GET  /model/info    — returns model name, version, optimal_T, mc_passes
#
# Architecture:
#   - Model is loaded ONCE at startup via FastAPI lifespan context manager
#     and stored in the module-level MODEL_STATE dict.  Individual requests
#     bear zero reload cost (~16 MB EfficientNet-B0 weights).
#   - MC Dropout is kept active at inference time by forcing every nn.Dropout
#     layer back to train() mode after model.eval().
#   - Temperature scaling (optimal_T.npy) is applied to mean logits before
#     computing calibrated probabilities and uncertainty signals.
# ============================================================

import io
import os
import sys
import logging

import numpy as np
import torch
import torch.nn as nn
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image

# ── Make the project root importable regardless of CWD ────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.setup.config import (
    BASE_CONFIG,
    UNCERTAINTY_ENTROPY_THRESHOLD,
    UNCERTAINTY_MARGIN_THRESHOLD,
    UNCERTAINTY_MC_STD_THRESHOLD,
)
from pipeline.data.dataset import val_transformer
from pipeline.training_loop_setup.model import EfficientNetMC
from pipeline.evaluation.calibration import apply_temperature, triage_sample




# ── Logger ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("api.app")

# ── Constants ──────────────────────────────────────────────────
MODEL_VERSION  = "1.0.0"
DR_CLASSES     = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
ALLOWED_TYPES  = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}

# Read hyperparameters from the same BASE_CONFIG used by the training pipeline
MODEL_PATH     = BASE_CONFIG["model_save_path"]      # artifacts/weights/aptos_efficientnet.pth
OPTIMAL_T_PATH = BASE_CONFIG["optimal_T_save_path"]  # artifacts/calibration/optimal_T.npy
MC_PASSES      = BASE_CONFIG["mc_dropout_passes"]    # 30
DROPOUT_RATE   = BASE_CONFIG["dropout_rate"]         # 0.3
NUM_CLASSES    = BASE_CONFIG["num_classes"]          # 5

# ── Singleton model state (populated at startup, cleared at shutdown) ──
MODEL_STATE: dict = {}


# ── Lifespan: load model once, release on shutdown ─────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Everything before `yield` runs at startup:
      1. Detect device (CUDA / CPU)
      2. Instantiate EfficientNetMC and load trained weights
      3. Force dropout layers to train() mode for MC Dropout
      4. Load calibration temperature T from optimal_T.npy

    Everything after `yield` runs at shutdown:
      - Clears MODEL_STATE to release GPU / CPU memory
    """
    logger.info("=" * 60)
    logger.info("APTOS DR API — startup: loading model into memory ...")
    logger.info("=" * 60)

    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 2. Verify weights file exists before trying to load
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model weights not found at '{MODEL_PATH}'. "
            "Train a model first or check the path in BASE_CONFIG."
        )

    # 3. Build model architecture (pretrained=False: we load our own weights)
    model = EfficientNetMC(
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        pretrained=False,
    )

    # 4. Load trained weights
    logger.info(f"Loading weights from: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # 5. Set to eval mode globally, then re-enable dropout for MC Dropout
    #    model.eval() stops BatchNorm updates but also disables dropout.
    #    We manually flip each nn.Dropout back to train() so stochastic
    #    sampling remains active across the MC_PASSES forward passes.
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    logger.info(f"MC Dropout active | passes per request: {MC_PASSES}")

    # 6. Load calibration temperature T
    optimal_T = 1.0  # uncalibrated fallback
    if os.path.exists(OPTIMAL_T_PATH):
        try:
            optimal_T = float(np.load(OPTIMAL_T_PATH))
            logger.info(f"Loaded optimal temperature T = {optimal_T:.4f}")
        except Exception as exc:
            logger.warning(
                f"Could not load optimal_T from '{OPTIMAL_T_PATH}' "
                f"(falling back to T=1.0). Details: {exc}"
            )
    else:
        logger.warning(
            f"optimal_T.npy not found at '{OPTIMAL_T_PATH}'. "
            "Running uncalibrated (T=1.0)."
        )

    # 7. Store everything in module-level dict so endpoints can access it
    MODEL_STATE.update({
        "model":     model,
        "device":    device,
        "optimal_T": optimal_T,
    })

    logger.info("=" * 60)
    logger.info("Model ready — API is live.")
    logger.info("=" * 60)

    yield  # ← server is live here, handling requests

    # Shutdown
    logger.info("API shutdown: releasing model state ...")
    MODEL_STATE.clear()
    logger.info("Done.")


# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title="APTOS Diabetic Retinopathy Inference API",
    description=(
        "Upload a retinal fundus image and receive a DR severity prediction "
        "with calibrated confidence, Bayesian uncertainty signals, and a "
        "clinical triage label.  Powered by EfficientNet-B0 + MC Dropout + "
        "temperature scaling.\n\n"
        "Interactive docs: `/docs` | Alternative: `/redoc`"
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ── Pydantic response schemas ──────────────────────────────────
class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name:   str
    version:      str
    num_classes:  int
    mc_passes:    int
    optimal_T:    float
    dropout_rate: float
    device:       str


class PredictResponse(BaseModel):
    prediction:    str
    confidence:    float
    triage:        str
    entropy:       float
    margin:        float
    mc_std:        float
    model_version: str


# ── GET /health ────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health():
    """
    Confirm the API server is running.

    Returns `{"status": "ok"}` if the server is up and the model is loaded.
    """
    if not MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return HealthResponse(status="ok")


# ── GET /model/info ────────────────────────────────────────────
@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model metadata",
    tags=["System"],
)
async def model_info():
    """
    Return metadata about the loaded model.

    Includes the calibration temperature (`optimal_T`) found during
    post-hoc temperature scaling, and the number of MC Dropout passes
    used per inference request.
    """
    return ModelInfoResponse(
        model_name   = "EfficientNet-B0 MC Dropout",
        version      = MODEL_VERSION,
        num_classes  = NUM_CLASSES,
        mc_passes    = MC_PASSES,
        optimal_T    = MODEL_STATE.get("optimal_T", 1.0),
        dropout_rate = DROPOUT_RATE,
        device       = str(MODEL_STATE.get("device", "unknown")),
    )


# ── POST /predict ──────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Retinal image DR triage",
    tags=["Inference"],
)
async def predict(file: UploadFile = File(...)):
    """
    Upload a retinal fundus image (JPEG / PNG / BMP / TIFF) and receive:

    - **prediction** — DR severity class (`No DR` … `Proliferative DR`)
    - **confidence** — temperature-calibrated probability for the predicted class
    - **triage** — clinical routing label (`ROUTINE` / `HIGH SEVERITY - urgent review` / `UNCERTAIN - refer to specialist`)
    - **entropy** — predictive entropy (higher = more uncertain)
    - **margin** — top-1 vs top-2 probability gap (lower = model is indecisive)
    - **mc_std** — mean standard deviation across MC Dropout passes (higher = unstable)
    - **model_version** — version string of the loaded model
    """

    # ── 1. File type validation ────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file type: '{file.content_type}'. "
                f"Accepted types: {sorted(ALLOWED_TYPES)}"
            ),
        )

    # ── 2. Read bytes & decode to PIL Image ───────────────────
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode image file: {exc}",
        )

    # ── 3. Preprocess ──────────────────────────────────────────
    model     = MODEL_STATE["model"]
    device    = MODEL_STATE["device"]
    optimal_T = MODEL_STATE["optimal_T"]

    # val_transformer: Resize(224,224) → ToTensor → Normalize(ImageNet stats)
    img_tensor = val_transformer(pil_img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # ── 4. MC Dropout stochastic forward passes ────────────────
    # Dropout layers are already in train() mode (set during startup lifespan).
    # torch.no_grad() disables gradient tracking — does NOT affect dropout.
    passes       = []
    logit_passes = []

    with torch.no_grad():
        for _ in range(MC_PASSES):
            logits = model(img_tensor)                          # [1, C]
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            passes.append(probs)
            logit_passes.append(logits.cpu().numpy())

    passes       = np.array(passes)       # [T, 1, C]
    logit_passes = np.array(logit_passes) # [T, 1, C]

    mean_probs  = passes.mean(axis=0)[0]       # [C]  — mean softmax over T passes
    mean_logits = logit_passes.mean(axis=0)    # [1, C] — mean raw logits over T passes
    mc_std      = float(passes.std(axis=0)[0].mean())  # scalar — mean std across classes

    # ── 5. Temperature scaling ─────────────────────────────────
    # Divide mean logits by optimal_T before softmax to produce calibrated probs.
    # T > 1 softens (widens) the distribution; T < 1 sharpens it.
    calibrated_probs = apply_temperature(mean_logits, optimal_T)[0]  # [C]

    # ── 6. Uncertainty signals ─────────────────────────────────
    eps     = 1e-8
    entropy = float(-np.sum(calibrated_probs * np.log(calibrated_probs + eps)))

    sorted_p = np.sort(calibrated_probs)[::-1]
    margin   = float(sorted_p[0] - sorted_p[1])

    # ── 7. Prediction ──────────────────────────────────────────
    pred_idx   = int(np.argmax(calibrated_probs))
    pred_label = DR_CLASSES[pred_idx]
    confidence = float(calibrated_probs[pred_idx])

    # ── 8. Triage routing ──────────────────────────────────────
    triage = triage_sample(
        pred          = pred_idx,
        entropy       = entropy,
        margin        = margin,
        mc_std        = mc_std,
        entropy_thresh = UNCERTAINTY_ENTROPY_THRESHOLD,
        margin_thresh  = UNCERTAINTY_MARGIN_THRESHOLD,
        mc_std_thresh  = UNCERTAINTY_MC_STD_THRESHOLD,
    )

    return PredictResponse(
        prediction    = pred_label,
        confidence    = round(confidence, 4),
        triage        = triage,
        entropy       = round(entropy, 4),
        margin        = round(margin, 4),
        mc_std        = round(mc_std, 6),
        model_version = MODEL_VERSION,
    )
