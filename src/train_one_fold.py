import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader

mp.set_sharing_strategy("file_system")

from config import (
    DATA_ROOT,
    EXPERIMENTS_ROOT,
    CONTINUOUS_FEATURE_IDXS,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    PREFETCH_FACTOR,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SEED,
    OPTIMIZER_NAME,
    LOSS_NAME,
    DECISION_THRESHOLD,
    MODEL_SELECTION_METRIC,
    THRESHOLD_SELECTION_METRIC,
    THRESHOLD_CANDIDATES,
    SAVE_BEST_MODEL,
    SAVE_LAST_MODEL,
    SAVE_NORM_STATS,
    PRINT_EVERY_N_STEPS,
    SHARD_CACHE_SIZE,
    TRAIN_SHUFFLE,
    TRAIN_DROP_LAST,
)
from make_folds import get_fold_by_id
from shard_index import (
    build_shard_index,
    filter_shard_index_by_bags,
    build_shard_dataset_map,
)
from dataset import ShardSequenceDataset
from model import CastaMaskFullScanCNN
from utils import (
    set_seed,
    seed_worker,
    compute_feature_norm_stats_from_shards,
    masked_bce_with_logits_loss,
    update_binary_confusion_counts_from_logits,
    compute_binary_metrics_from_counts,
    compute_binary_metrics_from_logits,
    find_best_threshold_from_logits,
    save_json,
    save_checkpoint,
)


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    name = OPTIMIZER_NAME.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")


def move_batch_to_device(batch, device):
    x, y, valid_current = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    valid_current = valid_current.to(device, non_blocking=True)
    return x, y, valid_current


def build_loader_kwargs():
    kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "drop_last": False,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return kwargs


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()

    total_loss = 0.0
    total_sequences = 0
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for batch in loader:
        x, y, valid_current = move_batch_to_device(batch, device)
        logits = model(x)
        loss = masked_bce_with_logits_loss(logits, y, valid_current)

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_sequences += bs

        update_binary_confusion_counts_from_logits(
            logits=logits.detach().cpu(),
            targets=y.detach().cpu(),
            valid_mask=valid_current.detach().cpu(),
            threshold=DECISION_THRESHOLD,
            counts=counts,
        )

    metrics = compute_binary_metrics_from_counts(counts)
    metrics["loss"] = total_loss / max(total_sequences, 1)
    return metrics


@torch.no_grad()
def collect_logits_targets_masks(loader: DataLoader, model: nn.Module, device: torch.device):
    model.eval()

    all_logits = []
    all_targets = []
    all_masks = []

    for batch in loader:
        x, y, valid_current = move_batch_to_device(batch, device)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())
        all_masks.append(valid_current.detach().cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0), torch.cat(all_masks, dim=0)


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int):
    model.train()

    total_loss = 0.0
    total_sequences = 0
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for step, batch in enumerate(loader, start=1):
        x, y, valid_current = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = masked_bce_with_logits_loss(logits, y, valid_current)
        loss.backward()
        optimizer.step()

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_sequences += bs

        update_binary_confusion_counts_from_logits(
            logits=logits.detach().cpu(),
            targets=y.detach().cpu(),
            valid_mask=valid_current.detach().cpu(),
            threshold=DECISION_THRESHOLD,
            counts=counts,
        )

        if PRINT_EVERY_N_STEPS > 0 and (step % PRINT_EVERY_N_STEPS == 0):
            print(f"Epoch {epoch:03d} | step {step:05d} | loss {loss.item():.6f}")

    metrics = compute_binary_metrics_from_counts(counts)
    metrics["loss"] = total_loss / max(total_sequences, 1)
    return metrics


def is_better_metric(current_value: float, best_value: float, metric_name: str) -> bool:
    if metric_name.lower() == "loss":
        return current_value < best_value
    return current_value > best_value


def get_initial_best_value(metric_name: str) -> float:
    if metric_name.lower() == "loss":
        return float("inf")
    return -float("inf")


def pick_model_selection_value(metrics: dict) -> float:
    key = MODEL_SELECTION_METRIC.lower()
    if key not in metrics:
        raise KeyError(f"Model selection metric '{MODEL_SELECTION_METRIC}' not found in metrics.")
    return float(metrics[key])


def train_one_fold(fold_id: int, device: str = "cpu") -> dict:
    set_seed(SEED)

    device = torch.device(device)
    fold = get_fold_by_id(fold_id)

    print(f"\n=== Fold {fold_id} ===")
    print(f"Train families: {fold['train_families']}")
    print(f"Val family:     {fold['val_family']}")
    print(f"Test family:    {fold['test_family']}")
    print(f"Train bags: {fold['train_bags']}")
    print(f"Val bags:   {fold['val_bags']}")
    print(f"Test bags:  {fold['test_bags']}")

    all_shard_index = build_shard_index(DATA_ROOT)
    train_shard_index = filter_shard_index_by_bags(all_shard_index, fold["train_bags"])
    val_shard_index = filter_shard_index_by_bags(all_shard_index, fold["val_bags"])
    test_shard_index = filter_shard_index_by_bags(all_shard_index, fold["test_bags"])

    if len(train_shard_index) == 0 or len(val_shard_index) == 0 or len(test_shard_index) == 0:
        raise RuntimeError("One of train/val/test shard splits is empty.")

    train_dataset_map = build_shard_dataset_map(train_shard_index)
    val_dataset_map = build_shard_dataset_map(val_shard_index)
    test_dataset_map = build_shard_dataset_map(test_shard_index)

    print(
        f"Samples | train={train_dataset_map['total_samples']} "
        f"val={val_dataset_map['total_samples']} test={test_dataset_map['total_samples']}"
    )

    norm_stats = compute_feature_norm_stats_from_shards(
        shard_index=train_shard_index,
        continuous_feature_idxs=CONTINUOUS_FEATURE_IDXS,
    )

    train_ds = ShardSequenceDataset(train_dataset_map, norm_stats=norm_stats, return_meta=False, max_cached_shards=SHARD_CACHE_SIZE)
    val_ds = ShardSequenceDataset(val_dataset_map, norm_stats=norm_stats, return_meta=False, max_cached_shards=SHARD_CACHE_SIZE)
    test_ds = ShardSequenceDataset(test_dataset_map, norm_stats=norm_stats, return_meta=False, max_cached_shards=SHARD_CACHE_SIZE)

    data_loader_kwargs = build_loader_kwargs()
    generator = torch.Generator()
    generator.manual_seed(SEED + fold_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=TRAIN_SHUFFLE,
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        generator=generator,
        **data_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        generator=generator,
        **data_loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        generator=generator,
        **data_loader_kwargs,
    )

    model = CastaMaskFullScanCNN().to(device)
    optimizer = build_optimizer(model)

    fold_dir = EXPERIMENTS_ROOT / f"fold_{fold_id:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_NORM_STATS:
        norm_stats_to_save = {
            "mean": norm_stats["mean"].tolist(),
            "std": norm_stats["std"].tolist(),
            "continuous_feature_idxs": CONTINUOUS_FEATURE_IDXS,
        }
        save_json(norm_stats_to_save, fold_dir / "norm_stats.json")

    split_info = {
        "fold_id": fold_id,
        "train_families": fold["train_families"],
        "val_family": fold["val_family"],
        "test_family": fold["test_family"],
        "train_bags": fold["train_bags"],
        "val_bags": fold["val_bags"],
        "test_bags": fold["test_bags"],
        "num_train_shards": len(train_shard_index),
        "num_val_shards": len(val_shard_index),
        "num_test_shards": len(test_shard_index),
        "num_train_samples": int(train_dataset_map["total_samples"]),
        "num_val_samples": int(val_dataset_map["total_samples"]),
        "num_test_samples": int(test_dataset_map["total_samples"]),
    }
    save_json(split_info, fold_dir / "split_info.json")

    best_val_metric = get_initial_best_value(MODEL_SELECTION_METRIC)
    best_epoch = -1
    patience_counter = 0
    best_model_path = fold_dir / "best_model.pt"

    history = {"train": [], "val": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = evaluate(model, val_loader, device)

        history["train"].append({"epoch": epoch, **train_metrics})
        history["val"].append({"epoch": epoch, **val_metrics})

        current_val_metric = pick_model_selection_value(val_metrics)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.6f} f1={train_metrics['f1']:.4f} | "
            f"val loss={val_metrics['loss']:.6f} f1={val_metrics['f1']:.4f} "
            f"precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f}"
        )

        if is_better_metric(current_val_metric, best_val_metric, MODEL_SELECTION_METRIC):
            best_val_metric = current_val_metric
            best_epoch = epoch
            patience_counter = 0

            if SAVE_BEST_MODEL:
                save_checkpoint(model, optimizer, epoch, best_val_metric, best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    save_json(history, fold_dir / "history.json")

    if best_model_path.exists():
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    val_logits, val_targets, val_masks = collect_logits_targets_masks(val_loader, model, device)
    threshold_search = find_best_threshold_from_logits(
        logits=val_logits,
        targets=val_targets,
        valid_mask=val_masks,
        candidate_thresholds=THRESHOLD_CANDIDATES,
        metric_name=THRESHOLD_SELECTION_METRIC,
    )

    best_threshold = float(threshold_search["best_threshold"])
    print(
        f"Best validation threshold: {best_threshold:.4f} "
        f"({THRESHOLD_SELECTION_METRIC}={threshold_search['best_metric_value']:.6f})"
    )

    test_logits, test_targets, test_masks = collect_logits_targets_masks(test_loader, model, device)
    test_metrics = compute_binary_metrics_from_logits(
        logits=test_logits,
        targets=test_targets,
        valid_mask=test_masks,
        threshold=best_threshold,
    )
    test_loss = masked_bce_with_logits_loss(test_logits, test_targets, test_masks).item()
    test_metrics["loss"] = float(test_loss)
    test_metrics["decision_threshold"] = best_threshold

    results = {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
        "model_selection_metric": MODEL_SELECTION_METRIC,
        "threshold_selection_metric": THRESHOLD_SELECTION_METRIC,
        "threshold_candidates": THRESHOLD_CANDIDATES,
        "best_threshold": best_threshold,
        "val_threshold_search": threshold_search,
        "test_metrics": test_metrics,
    }
    save_json(results, fold_dir / "test_metrics.json")

    if SAVE_LAST_MODEL:
        torch.save(model.state_dict(), fold_dir / "last_model_state_dict.pt")

    print("\n=== Final test metrics ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    return results


if __name__ == "__main__":
    train_one_fold(fold_id=0, device="cpu")