from pathlib import Path

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

DATA_ROOT = Path("data")
EXPERIMENTS_ROOT = Path("experiments_fullscan")

# -------------------------------------------------------------------
# Data / split settings
# -------------------------------------------------------------------

NUM_CLASSES = 1  # binary classification: static vs dynamic

# Current feature order:
# 0 range_norm
# 1 delta_r
# 2 static_residual
# 3 abs_static_residual
# 4 spatial_grad
# 5 valid_mask
CONTINUOUS_FEATURE_IDXS = [0, 1, 2, 3, 4]

NORM_EPS = 1e-6
NUM_BEAMS = 360
TIME_WINDOW = 7

# -------------------------------------------------------------------
# Training hyperparameters
# -------------------------------------------------------------------

BATCH_SIZE = 16
NUM_WORKERS = 0
PIN_MEMORY = False

PERSISTENT_WORKERS = False
PREFETCH_FACTOR = 2

MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

SEED = 42

# -------------------------------------------------------------------
# Optimizer / loss / scheduler
# -------------------------------------------------------------------

OPTIMIZER_NAME = "adamw"
LOSS_NAME = "bce_logits_masked"

USE_POS_WEIGHT = False
FIXED_POS_WEIGHT = 1.0

USE_LR_SCHEDULER = False
SCHEDULER_NAME = "reduce_on_plateau"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
SCHEDULER_MIN_LR = 1e-6

# -------------------------------------------------------------------
# Model architecture
# -------------------------------------------------------------------

IN_CHANNELS = 6

CONV1_OUT = 16
CONV2_OUT = 32
CONV3_OUT = 32

KERNEL_SIZE = 3
PADDING = 1
DROPOUT = 0.1

# -------------------------------------------------------------------
# Validation / metrics
# -------------------------------------------------------------------

DECISION_THRESHOLD = 0.5
MODEL_SELECTION_METRIC = "loss"
THRESHOLD_SELECTION_METRIC = "f1"
THRESHOLD_CANDIDATES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

# -------------------------------------------------------------------
# Saving / logging
# -------------------------------------------------------------------

SAVE_BEST_MODEL = True
SAVE_LAST_MODEL = False
SAVE_NORM_STATS = True
SAVE_FOLD_PREDICTIONS = False

PRINT_EVERY_N_STEPS = 50

# -------------------------------------------------------------------
# Data loading efficiency
# -------------------------------------------------------------------

SHARD_CACHE_SIZE = 2

TRAIN_SHUFFLE = True
TRAIN_DROP_LAST = False

# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------


def validate_config():
    assert DATA_ROOT is not None
    assert BATCH_SIZE > 0
    assert NUM_WORKERS >= 0
    assert MAX_EPOCHS > 0
    assert EARLY_STOPPING_PATIENCE > 0
    assert LEARNING_RATE > 0
    assert CONV1_OUT > 0
    assert CONV2_OUT > 0
    assert CONV3_OUT > 0
    assert SHARD_CACHE_SIZE >= 1
    assert NUM_BEAMS > 0
    assert TIME_WINDOW > 0
    assert MODEL_SELECTION_METRIC in {"f1", "precision", "recall", "loss", "accuracy"}
    assert THRESHOLD_SELECTION_METRIC in {"f1", "precision", "recall", "accuracy"}
    assert OPTIMIZER_NAME in {"adam", "adamw"}
    assert LOSS_NAME in {"bce_logits_masked"}


validate_config()