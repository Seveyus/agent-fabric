"""
Evaluation utilities for the Delivery World Model.

Key metrics:
    representation_quality  — how well the latent space captures delivery dynamics
    prediction_accuracy     — how well the predictor forecasts future state
    downstream_auc          — AUC-ROC on regression/velocity_drop labels

Quick eval:
    python -m training.evaluate --checkpoint checkpoints/best.pt --pairs data/pairs_14d.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report

from models.world_model import DeliveryWorldModel, WorldModelConfig
from training.dataset import DeliveryPairDataset, collate_fn

log = logging.getLogger(__name__)


def evaluate(checkpoint_path: str, pairs_path: str, device: str = "cpu") -> dict:
    """
    Load a checkpoint and evaluate on a pairs dataset.

    Returns dict with all computed metrics.
    """
    device_obj = torch.device(device)

    state = torch.load(checkpoint_path, map_location=device_obj)
    model = DeliveryWorldModel(WorldModelConfig()).to(device_obj)
    model.load_state_dict(state["model"])
    model.eval()

    dataset = DeliveryPairDataset(pairs_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    all_s_x, all_s_y, all_s_y_hat = [], [], []
    all_labels, all_logits = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device_obj) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            s_x = model.online_encoder(batch["graph_t"])
            s_y = model.target_encoder(batch["graph_t_k"])
            s_y_hat = model.predictor(s_x, batch["events"])
            logits = model.risk_head(s_x)

            all_s_x.append(s_x.cpu())
            all_s_y.append(s_y.cpu())
            all_s_y_hat.append(s_y_hat.cpu())
            all_labels.append(batch["labels"].cpu())
            all_logits.append(logits.cpu())

    s_x = torch.cat(all_s_x)
    s_y = torch.cat(all_s_y)
    s_y_hat = torch.cat(all_s_y_hat)
    labels = torch.cat(all_labels).numpy()
    logits = torch.cat(all_logits).numpy()

    # JEPA prediction quality
    pred_mse = torch.nn.functional.mse_loss(s_y_hat, s_y).item()
    pred_cos = torch.nn.functional.cosine_similarity(s_y_hat, s_y).mean().item()

    # Representation collapse check: std of representations should be > 0.1
    repr_std = s_x.std(dim=0).mean().item()

    # Downstream classification
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    label_names = ["regression", "velocity_drop", "release_in_t_k"]
    aucs = {}
    for i, name in enumerate(label_names):
        y_true = labels[:, i]
        if y_true.sum() > 0 and (1 - y_true).sum() > 0:
            aucs[f"auc_{name}"] = roc_auc_score(y_true, probs[:, i])
        else:
            aucs[f"auc_{name}"] = float("nan")

    metrics = {
        "prediction_mse": pred_mse,
        "prediction_cosine_sim": pred_cos,
        "representation_std": repr_std,
        "n_samples": len(labels),
        **aucs,
    }

    log.info("Evaluation results:")
    for k, v in metrics.items():
        log.info("  %-35s %.4f", k, v if not np.isnan(v) else float("nan"))

    # Sanity check: are representations non-collapsed?
    if repr_std < 0.05:
        log.warning("⚠️  Representation std = %.4f — possible collapse! Check EMA tau.", repr_std)
    else:
        log.info("✓  Representations look healthy (std=%.4f)", repr_std)

    # AUC sanity
    mean_auc = np.nanmean([aucs[k] for k in aucs])
    if mean_auc > 0.65:
        log.info("✓  Mean downstream AUC = %.3f — signal is present", mean_auc)
    else:
        log.warning("⚠️  Mean AUC = %.3f — weak signal. Collect more data or tune features.", mean_auc)

    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    evaluate(args.checkpoint, args.pairs, args.device)
