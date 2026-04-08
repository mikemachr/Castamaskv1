from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from family_map import BAG_TO_FAMILY, BAG_TO_NAME


def find_shard_files(data_root: Path) -> List[Path]:
    shard_files = sorted(data_root.glob("train_shard_*.npz"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {data_root}")
    return shard_files


def read_shard_header(shard_path: Path) -> Dict:
    with np.load(shard_path, allow_pickle=True) as z:
        if "X" not in z or "y" not in z or "bag_id" not in z:
            raise KeyError(f"Shard {shard_path} is missing one of: X, y, bag_id")

        num_samples = int(z["y"].shape[0])
        bag_ids = np.asarray(z["bag_id"], dtype=np.int64)

    unique_bag_ids = sorted(int(b) for b in np.unique(bag_ids))
    family_names = sorted({BAG_TO_FAMILY[b] for b in unique_bag_ids})
    scenario_names = [BAG_TO_NAME[b] for b in unique_bag_ids]

    return {
        "shard_path": str(shard_path),
        "num_samples": num_samples,
        "bag_ids_present": unique_bag_ids,
        "family_names_present": family_names,
        "scenario_names_present": scenario_names,
    }


def build_shard_index(data_root: Path) -> List[Dict]:
    shard_files = find_shard_files(data_root)
    return [read_shard_header(p) for p in shard_files]


def filter_shard_index_by_bags(shard_index: List[Dict], allowed_bag_ids: List[int]) -> List[Dict]:
    allowed = set(int(b) for b in allowed_bag_ids)
    out = []
    for row in shard_index:
        bag_ids_present = row["bag_ids_present"]
        if any(int(b) in allowed for b in bag_ids_present):
            out.append(row)
    return out


def build_shard_dataset_map(shard_index: List[Dict]) -> Dict:
    cumulative_sizes = []
    running = 0

    for row in shard_index:
        running += int(row["num_samples"])
        cumulative_sizes.append(running)

    return {
        "shard_index": shard_index,
        "cumulative_sizes": cumulative_sizes,
        "total_samples": running,
    }


def locate_global_index(dataset_map: Dict, global_idx: int) -> Tuple[Dict, int]:
    total = int(dataset_map["total_samples"])
    if global_idx < 0 or global_idx >= total:
        raise IndexError(f"global_idx {global_idx} out of range [0, {total - 1}]")

    cumulative_sizes = dataset_map["cumulative_sizes"]
    shard_pos = bisect_right(cumulative_sizes, global_idx)
    shard_row = dataset_map["shard_index"][shard_pos]

    prev_cum = 0 if shard_pos == 0 else cumulative_sizes[shard_pos - 1]
    local_idx = global_idx - prev_cum

    return shard_row, local_idx