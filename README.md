# CastaMask

Dynamic LiDAR beam filtering for autonomous navigation in industrial warehouses.
A modular scientific computing system built for TC6039.1 Applied Computing — Tec de Monterrey.

## What it does

A TurtleBot equipped with a 2D LiDAR sensor navigates a simulated warehouse. Moving objects (people, robots, pallets) introduce transient measurements that degrade SLAM localization. CastaMask filters those dynamic beams at the beam level before the scan reaches the localization module.

The core model is a temporal CNN that classifies each of the 360 LiDAR beams as static (0) or dynamic (1), using a sliding window of 7 consecutive frames to detect motion over time.

## Team

| Name | Component |
|------|-----------|
| Gerardo Andrés Castañón Sarmiento | C1 — Data loading & EDA |
| Priscila de los Ángeles Correa Miranda | C2 — Feature engineering · C3 — Hyperparameter optimization |
| Juan Angel Lucio Rojas | C4 — Classical ML baselines |
| Miguel Ángel Chávez Robles | C5 — Deep learning (CNN Temporal) |
| Ricardo Daniel Damián Cortez | C6 — Visualization · C7 — Orchestration |

## Project structure

```md
CastaMask/
├── main.py                  # Pipeline orchestrator (C7)
├── Makefile                 # All run targets
├── requirements.txt
├── data/                    # Shards (.npz) — not tracked in git
├── experiments_fullscan/    # Training outputs, checkpoints — not tracked
├── report/
│   └──                      # Output from running C6
├── src/
│   ├── data_loader.py       # C1
│   ├── feature_engineering.py  # C2
│   ├── optimizer.py         # C3
│   ├── ml_models.py         # C4
│   ├── dl_model.py          # C5 public interface
│   ├── viz.py               # C6
│   ├── model.py             # CNN architecture
│   ├── train_one_fold.py    # Training loop
│   ├── run_cv.py            # Cross-validation
│   ├── dataset.py           # PyTorch dataset (shard-backed)
│   ├── shard_index.py       # Shard indexing utilities
│   ├── make_folds.py        # Leave-one-family-out CV folds
│   ├── family_map.py        # Scenario metadata
│   ├── config.py            # All hyperparameters
│   └── utils.py             # Loss, metrics, checkpointing
└── tests/
    └── test_all_components.py
```

## Installation

```bash
git clone https://github.com/mikemachr/Castamaskv1.git
cd Castamask
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+. No GPU required (CPU training supported).

## Data

Data is not included in the repository. Place shard files in `data/`:
View readme in `data/` for download link.

```
data/
├── train_shard_00000.npz
├── train_shard_00001.npz
└── ...
```

Each shard contains 512 LiDAR scans from a single scenario bag. The dataset has 332 shards across 13 scenarios (bag_ids 0–12), totaling ~170k scans.

## Running the pipeline

All targets run from the project root.

```bash
make dry          # verify the code runs end-to-end (~1-2 min)
make fast         # real training, 1 shard/scenario, skip optimizer (~5-10 min)
make run          # full dataset, skip optimizer (~1-2 hrs)
make run-full     # full dataset + hyperparameter search (slow)
make test         # unit tests
make report       # generate figures + compile LaTeX
make help         # list all targets
```

### Fast mode targets

Fast mode loads one shard per bag_id (all 13 scenarios represented) instead of the full dataset, making iteration much quicker without sacrificing scenario diversity.

```bash
make fast                   # 1 shard/bag_id, no optimizer
make fast-n N=3             # 3 shards/bag_id, no optimizer
make fast-opt               # 1 shard/bag_id + hyperparameter search
make fast-opt-n N=3         # 3 shards/bag_id + hyperparameter search
```

### Direct usage

```bash
python main.py --skip-opt               # full run, no optimizer
python main.py --fast --max-shards 13   # fast run
python main.py --dry-run                # smoke test
python main.py --fold 0 --device cuda   # specific fold on GPU
```

## Architecture

### Input tensor

Each sample is a tensor `X ∈ ℝ^{T×360×6}` with T=7 temporal frames and 6 features per beam:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `range_norm` | Normalized range [0, 1] |
| 1 | `delta_r` | Range change vs previous frame |
| 2 | `static_residual` | Residual vs static map |
| 3 | `abs_static_residual` | Absolute residual |
| 4 | `spatial_grad` | Angular gradient between adjacent beams |
| 5 | `valid_mask` | Binary validity mask |

### Model (CastaMaskFullScanCNN)

```
Input  [B, 6, 7, 360]
  → Conv2d(6→16, 3×3) + ReLU
  → Conv2d(16→32, 3×3) + ReLU
  → Conv2d(32→32, 3×3) + ReLU
  → AdaptiveAvgPool2d → (1, 360)    # collapse temporal dim
  → Conv1d(32→1)
Output [B, 360]                      # one logit per beam
```

Conv2D over the (T, N) dimensions simultaneously captures temporal motion patterns and angular context — a dynamic object typically spans several adjacent beams. This design enables fully parallel inference, validated at <45ms per scan on CPU (22 Hz deployment target).

### Cross-validation

Leave-one-family-out grouped CV across 8 scenario families. Grouping prevents data leakage — `bag_id=2` (MIR robot vel 0.8, robot vel 0.3) and `bag_id=3` (same robot, different speeds) belong to the same family and are never split across train/test.

### Scientific hypothesis

> A temporal CNN using sliding windows can classify LiDAR beams as static or dynamic in real time, outperforming single-frame classifiers.

Primary metric: **recall on dynamic beams > 85%**. A missed dynamic beam (FN) contaminates the SLAM map — a more serious failure than a static beam incorrectly filtered (FP).

## Pipeline components

| Component | File | Responsibility |
|-----------|------|---------------|
| C1 | `data_loader.py` | Load shards, clean data, EDA (4 figures) |
| C2 | `feature_engineering.py` | Extract and normalize feature tensors |
| C3 | `optimizer.py` | Grid search over learning rate, window size, architecture |
| C4 | `ml_models.py` | Logistic Regression, Decision Tree, Random Forest baselines |
| C5 | `dl_model.py` | Train temporal CNN, cross-validation |
| C6 | `viz.py` | Training curves, comparison table, LaTeX report |
| C7 | `main.py` | Orchestrate C1→C6, timing, JSON summary |

## Output

After `make run` or `make fast`, outputs appear in:

```
experiments_fullscan/fold_00/
├── best_model.pt        # best checkpoint by val loss
├── history.json         # loss and F1 per epoch
├── norm_stats.json      # normalization statistics
├── split_info.json      # fold split details
└── test_metrics.json    # final test metrics

report/
├── figures/             # all generated plots
├── reporte_final.tex    # LaTeX report
└── pipeline_summary.json
```


