import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_feature_norm_stats_from_shards(
    shard_index: List[Dict],
    continuous_feature_idxs: List[int],
) -> Dict[str, np.ndarray]:
    if len(shard_index) == 0:
        raise ValueError("shard_index is empty")

    first_shard = shard_index[0]["shard_path"]
    with np.load(first_shard, allow_pickle=True) as z:
        K = int(z["X"].shape[-1])

    mean = np.zeros((K,), dtype=np.float64)
    std = np.ones((K,), dtype=np.float64)
    sums = np.zeros((K,), dtype=np.float64)
    sums_sq = np.zeros((K,), dtype=np.float64)
    counts = np.zeros((K,), dtype=np.float64)

    cont_set = set(int(i) for i in continuous_feature_idxs)

    for row in shard_index:
        shard_path = row["shard_path"]
        with np.load(shard_path, allow_pickle=True) as z:
            X = np.asarray(z["X"], dtype=np.float32)

        for feat_idx in cont_set:
            vals = X[:, :, :, feat_idx].astype(np.float64).reshape(-1)
            sums[feat_idx] += vals.sum()
            sums_sq[feat_idx] += np.square(vals).sum()
            counts[feat_idx] += vals.size

    for feat_idx in cont_set:
        if counts[feat_idx] == 0:
            mean[feat_idx] = 0.0
            std[feat_idx] = 1.0
            continue

        mu = sums[feat_idx] / counts[feat_idx]
        var = (sums_sq[feat_idx] / counts[feat_idx]) - (mu * mu)
        var = max(var, 0.0)
        sigma = np.sqrt(var)

        mean[feat_idx] = mu
        std[feat_idx] = sigma if sigma > 1e-8 else 1.0

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def masked_bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    weighted = loss * valid_mask
    denom = valid_mask.sum().clamp_min(1.0)
    return weighted.sum() / denom


def update_binary_confusion_counts_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float,
    counts: Dict[str, int],
) -> None:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()
    t = targets.long()
    v = valid_mask > 0.5

    counts["tp"] += int(((preds == 1) & (t == 1) & v).sum().item())
    counts["tn"] += int(((preds == 0) & (t == 0) & v).sum().item())
    counts["fp"] += int(((preds == 1) & (t == 0) & v).sum().item())
    counts["fn"] += int(((preds == 0) & (t == 1) & v).sum().item())


def compute_binary_metrics_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    tp = int(counts["tp"])
    tn = int(counts["tn"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    static_false_positive_rate = fp / max(fp + tn, 1)
    dynamic_false_negative_rate = fn / max(fn + tp, 1)

    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "static_false_positive_rate": float(static_false_positive_rate),
        "dynamic_false_negative_rate": float(dynamic_false_negative_rate),
    }


def compute_binary_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    update_binary_confusion_counts_from_logits(
        logits=logits,
        targets=targets,
        valid_mask=valid_mask,
        threshold=threshold,
        counts=counts,
    )
    return compute_binary_metrics_from_counts(counts)


def find_best_threshold_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    candidate_thresholds,
    metric_name: str = "f1",
) -> Dict[str, float]:
    best_threshold = None
    best_metric_value = -float("inf")
    best_metrics = None

    for thr in candidate_thresholds:
        metrics = compute_binary_metrics_from_logits(
            logits=logits,
            targets=targets,
            valid_mask=valid_mask,
            threshold=float(thr),
        )
        value = float(metrics[metric_name])

        if value > best_metric_value:
            best_metric_value = value
            best_threshold = float(thr)
            best_metrics = metrics

    out = dict(best_metrics)
    out["best_threshold"] = float(best_threshold)
    out["best_metric_value"] = float(best_metric_value)
    return out


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )