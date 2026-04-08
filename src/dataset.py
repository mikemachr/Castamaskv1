from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from config import CONTINUOUS_FEATURE_IDXS, NORM_EPS, SHARD_CACHE_SIZE
from family_map import BAG_TO_FAMILY, BAG_TO_NAME
from shard_index import locate_global_index


class ShardSequenceDataset(Dataset):
    """
    Dataset backed by shard-level indexing for full-scan temporal sequences.

    Each sample is stored in the shard as:
        X[local_idx] with shape [T, 360, K]
        y[local_idx] with shape [360]
        valid_current[local_idx] with shape [360]

    The model expects:
        x -> [K, T, 360]
    """

    def __init__(
        self,
        dataset_map: Dict,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
        return_meta: bool = False,
        max_cached_shards: int = SHARD_CACHE_SIZE,
    ):
        self.dataset_map = dataset_map
        self.norm_stats = norm_stats
        self.return_meta = return_meta
        self.max_cached_shards = int(max_cached_shards)
        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

        total = int(self.dataset_map["total_samples"])
        if total <= 0:
            raise ValueError("ShardSequenceDataset received an empty dataset_map.")
        if self.max_cached_shards < 1:
            raise ValueError("max_cached_shards must be >= 1")

        self._validate_norm_stats()

    def _validate_norm_stats(self):
        if self.norm_stats is None:
            return

        if "mean" not in self.norm_stats or "std" not in self.norm_stats:
            raise KeyError("norm_stats must contain keys: 'mean' and 'std'")

        mean = np.asarray(self.norm_stats["mean"], dtype=np.float32)
        std = np.asarray(self.norm_stats["std"], dtype=np.float32)

        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError("norm_stats['mean'] and norm_stats['std'] must be 1D arrays")
        if mean.shape != std.shape:
            raise ValueError("norm_stats['mean'] and norm_stats['std'] must have the same shape")

    def __len__(self) -> int:
        return int(self.dataset_map["total_samples"])

    def _read_shard_arrays(self, shard_path: str) -> Dict[str, np.ndarray]:
        shard_path = str(Path(shard_path))
        with np.load(shard_path, allow_pickle=True) as z:
            arrays = {
                "X": z["X"].astype(np.float32),
                "y": z["y"].astype(np.uint8),
                "valid_current": z["valid_current"].astype(np.uint8),
            }
            if "bag_id" in z:
                arrays["bag_id"] = z["bag_id"].astype(np.int32)
            if "stamp_ns" in z:
                arrays["stamp_ns"] = z["stamp_ns"].astype(np.int64)
        return arrays

    def _get_shard_arrays(self, shard_path: str) -> Dict[str, np.ndarray]:
        shard_path = str(Path(shard_path))
        if shard_path in self._cache:
            self._cache.move_to_end(shard_path)
            return self._cache[shard_path]

        arrays = self._read_shard_arrays(shard_path)
        self._cache[shard_path] = arrays
        self._cache.move_to_end(shard_path)
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return arrays

    def _normalize_sample(self, x: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return x

        mean = np.asarray(self.norm_stats["mean"], dtype=np.float32)
        std = np.asarray(self.norm_stats["std"], dtype=np.float32)

        x = x.copy()
        for feat_idx in CONTINUOUS_FEATURE_IDXS:
            x[:, :, feat_idx] = (x[:, :, feat_idx] - mean[feat_idx]) / (std[feat_idx] + NORM_EPS)
        return x

    def __getitem__(self, idx: int):
        shard_row, local_idx = locate_global_index(self.dataset_map, idx)
        shard_path = shard_row["shard_path"]
        arrays = self._get_shard_arrays(shard_path)

        x = arrays["X"][local_idx]  # [T, 360, K]
        y = arrays["y"][local_idx]  # [360]
        valid_current = arrays["valid_current"][local_idx]  # [360]

        x = self._normalize_sample(x)
        x = np.transpose(x, (2, 0, 1)).astype(np.float32)  # [K, T, 360]

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y.astype(np.float32))
        valid_tensor = torch.from_numpy(valid_current.astype(np.float32))

        if not self.return_meta:
            return x_tensor, y_tensor, valid_tensor

        bag_id = int(arrays["bag_id"][local_idx]) if "bag_id" in arrays else None
        meta = {
            "bag_id": bag_id,
            "family_name": BAG_TO_FAMILY[bag_id] if bag_id is not None else None,
            "scenario_name": BAG_TO_NAME[bag_id] if bag_id is not None else None,
            "shard_path": shard_path,
            "local_idx": int(local_idx),
        }
        if "stamp_ns" in arrays:
            meta["stamp_ns"] = int(arrays["stamp_ns"][local_idx])

        return x_tensor, y_tensor, valid_tensor, meta