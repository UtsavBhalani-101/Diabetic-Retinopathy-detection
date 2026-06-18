import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pipeline.data.dataset import RetinopathyDataset, val_transformer, CLAHEPreprocess
from pipeline.training_loop_setup.model import EfficientNetMC

logger = logging.getLogger("pipeline.gradcam_analysis")

def get_rgb_img(img_path):
    """Load image, apply CLAHE, resize to 224x224, return as float32 in [0, 1]."""
    img = Image.open(img_path).convert("RGB")
    clahe = CLAHEPreprocess(clip_limit=2.0, tile_grid_size=(8, 8))
    img = clahe(img)
    img = img.resize((224, 224))
    img_np = np.array(img).astype(np.float32) / 255.0
    return img_np

def run_batch_gradcam(model, dataset, loader, device, dataset_name, img_dir, img_col, extension, output_dir="results/gradcam_analysis"):
    """
    Run Batch Grad-CAM on a provided dataset and dataloader.
    Extracts features, clusters correct/incorrect predictions, and visualizes the nearest 5.
    """
    all_features = []
    all_preds = []
    all_labels = []
    
    logger.info(f"[{dataset_name}] Extracting features and predictions for Grad-CAM...")
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.base(images) # [B, 1280]
            logits = model.classifier(features)
            preds = logits.argmax(dim=1)
            
            all_features.append(features.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_features = np.vstack(all_features)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    correct_mask = (all_preds == all_labels)
    wrong_mask = (all_preds != all_labels)

    correct_indices = np.where(correct_mask)[0]
    wrong_indices = np.where(wrong_mask)[0]

    logger.info(f"[{dataset_name}] Found {len(correct_indices)} correct and {len(wrong_indices)} wrong predictions for Grad-CAM.")

    def get_representative_samples(indices, k=5):
        if len(indices) < k:
            logger.warning(f"Only {len(indices)} samples available, picking all.")
            return indices
        
        feats = all_features[indices]
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(feats)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feats)
        return indices[closest]

    logger.info(f"[{dataset_name}] Clustering {len(correct_indices)} correct features into 5 clusters...")
    selected_correct = get_representative_samples(correct_indices, k=5)
    
    logger.info(f"[{dataset_name}] Clustering {len(wrong_indices)} wrong features into 5 clusters...")
    selected_wrong = get_representative_samples(wrong_indices, k=5)

    target_layers = [model.base.blocks[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    def process_and_save(indices, category):
        out_dir = os.path.join(output_dir, dataset_name, category)
        os.makedirs(out_dir, exist_ok=True)
        
        for idx in indices:
            row = dataset.df.iloc[idx]
            image_id = row[img_col]
            true_label = int(all_labels[idx])
            pred_label = int(all_preds[idx])
            
            img_path = os.path.join(img_dir, f"{image_id}{extension}")
            
            rgb_img = get_rgb_img(img_path)
            
            pil_img = Image.open(img_path).convert("RGB")
            input_tensor = val_transformer(pil_img).unsqueeze(0).to(device)
            
            targets = [ClassifierOutputTarget(pred_label)]
            model.eval()
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            out_name = f"{image_id}_True{true_label}_Pred{pred_label}.jpg"
            out_path = os.path.join(out_dir, out_name)
            Image.fromarray(visualization).save(out_path)
            logger.info(f"[{dataset_name}] Saved {category} Grad-CAM: {out_path}")

    logger.info(f"[{dataset_name}] Generating Grad-CAM for representative successes...")
    process_and_save(selected_correct, "successes")
    
    logger.info(f"[{dataset_name}] Generating Grad-CAM for representative failures...")
    process_and_save(selected_wrong, "failures")
    
    logger.info(f"[{dataset_name}] Finished processing Grad-CAM. Results saved to {os.path.join(output_dir, dataset_name)}")

def analyze_dataset(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading EfficientNetMC from {args.model_path}")
    model = EfficientNetMC(num_classes=5, dropout_rate=0.3, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    logger.info(f"Loading dataset from {args.csv_path}")
    dataset = RetinopathyDataset(
        img_path=args.img_dir,
        img_col=args.img_col,
        label_col=args.label_col,
        transforms=val_transformer,
        extension=args.extension,
        target_path=args.csv_path
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    run_batch_gradcam(
        model=model,
        dataset=dataset,
        loader=loader,
        device=device,
        dataset_name=args.dataset_name,
        img_dir=args.img_dir,
        img_col=args.img_col,
        extension=args.extension,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., IDRiD)")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to labels CSV")
    parser.add_argument("--model_path", type=str, default="artifacts/weights/aptos_efficientnet.pth")
    parser.add_argument("--output_dir", type=str, default="results/gradcam_analysis")
    parser.add_argument("--img_col", type=str, default="id_code")
    parser.add_argument("--label_col", type=str, default="diagnosis")
    parser.add_argument("--extension", type=str)
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parser.parse_args()
    
    analyze_dataset(args)
