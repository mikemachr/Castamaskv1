"""
feature_engineering.py — Componente C2: Ingeniería de Características
======================================================================
Responsable: Priscila de los Ángeles Correa Miranda

Extrae y normaliza el tensor de características temporales a partir de
los shards .npz del dataset CastaMask. Produce el TENSOR_CARACTERISTICAS
[N, 360, K] consumido por C3 (optimización) y C4 (ML clásico), y el
TENSOR_TEMPORAL [M, T, 360, K] consumido por C5 (Deep Learning).

Las características ya están pre-computadas en los shards (generadas por
el pipeline de Gazebo + RayGtSensor). Este módulo las extrae, valida,
normaliza y expone con la interfaz acordada.

Features por haz (K=6):
    0  range_norm          — rango normalizado [0, 1]
    1  delta_r             — cambio de rango respecto al frame anterior [m]
    2  static_residual     — residuo respecto al mapa estático
    3  abs_static_residual — valor absoluto del residuo estático
    4  spatial_grad        — gradiente angular entre haces adyacentes
    5  valid_mask          — máscara binaria de haz válido {0, 1}

Interfaces de salida:
    extract_feature_tensor(shards)  -> TENSOR_CARACTERISTICAS [N, 360, K]
    extract_temporal_tensor(shards) -> TENSOR_TEMPORAL [M, T, 360, K]
    compute_norm_stats(X)           -> dict{mean, std}
    normalize_tensor(X, stats)      -> X normalizado

Referencias: Wilson et al. (2014) §5a (aserciones), §4b (modularización).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Root of the project (two levels up from src/)
_ROOT = Path(__file__).resolve().parent.parent
_FIGURES_DIR = _ROOT / "report" / "figures"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_BEAMS: int    = 360
NUM_FEATURES: int = 6
TIME_WINDOW: int  = 7    # debe coincidir con config.TIME_WINDOW

FEATURE_NAMES: List[str] = [
    "range_norm",
    "delta_r",
    "static_residual",
    "abs_static_residual",
    "spatial_grad",
    "valid_mask",
]

# Índices de features continuas (excluye valid_mask que es binaria)
CONTINUOUS_FEATURE_IDXS: List[int] = [0, 1, 2, 3, 4]

NORM_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# C2.1 — Extracción del tensor de características (frame actual)
# ---------------------------------------------------------------------------

def extract_feature_tensor(
    data_root: str | Path,
    pattern: str = "train_shard_*.npz",
    max_shards: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrae el tensor de características del frame actual de cada escaneo.

    Toma el frame temporal más reciente (índice T-1) de cada secuencia.
    Produce el TENSOR_CARACTERISTICAS usado por C4 (ML clásico).

    Args:
        data_root:  Directorio con shards .npz.
        pattern:    Patrón glob.
        max_shards: Límite opcional de shards a cargar.

    Returns:
        Tupla (X, y, valid_mask):
            X          — [N_total, 360, K] float32, features del frame actual
            y          — [N_total, 360]    uint8,   etiquetas GT binarias {0,1}
            valid_mask — [N_total, 360]    uint8,   máscara de haces válidos

    Raises:
        FileNotFoundError: Si no se encuentran shards.
        AssertionError:    Si los tensores contienen NaN o dimensiones incorrectas.
    """
    data_root = Path(data_root)
    shard_files = sorted(data_root.glob(pattern))

    if not shard_files:
        raise FileNotFoundError(f"No se encontraron shards en {data_root}")
    if max_shards:
        # Pick one shard per bag_id to guarantee all scenarios are covered.
        # Falls back to strided if numpy/npz reading fails.
        import numpy as _np
        seen_bags, selected = set(), []
        for _s in shard_files:
            try:
                _bid = int(_np.load(_s, allow_pickle=True)['bag_id'].flat[0])
            except Exception:
                _bid = -len(selected)  # unique fallback key
            if _bid not in seen_bags:
                seen_bags.add(_bid)
                selected.append(_s)
        shard_files = selected[:max_shards] if max_shards < len(selected) else selected

    X_list, y_list, v_list = [], [], []

    for shard_path in shard_files:
        with np.load(shard_path, allow_pickle=True) as z:
            X_shard = z["X"].astype(np.float32)           # [N, T, 360, K]
            y_shard = z["y"].astype(np.uint8)              # [N, 360]
            v_shard = z["valid_current"].astype(np.uint8)  # [N, 360]

        # PRECONDICIÓN: ventana temporal suficiente
        assert X_shard.shape[1] >= 2, \
            f"ventana_temporal={X_shard.shape[1]} debe ser >= 2"

        # Tomar solo el frame actual (último en la ventana)
        X_current = X_shard[:, -1, :, :]  # [N, 360, K]
        X_list.append(X_current)
        y_list.append(y_shard)
        v_list.append(v_shard)

    X     = np.concatenate(X_list, axis=0)  # [N_total, 360, K]
    y     = np.concatenate(y_list, axis=0)  # [N_total, 360]
    valid = np.concatenate(v_list, axis=0)  # [N_total, 360]

    # ASERCIÓN Wilson §5a: sin NaN en features continuas
    for feat_idx in CONTINUOUS_FEATURE_IDXS:
        feat_vals = X[:, :, feat_idx]
        assert not np.isnan(feat_vals).any(), \
            f"Feature {FEATURE_NAMES[feat_idx]} contiene valores NaN"

    assert set(np.unique(y)).issubset({0, 1}), \
        "y contiene valores distintos de {0, 1}"

    print(f"[feature_engineering] TENSOR_CARACTERISTICAS: {X.shape} "
          f"({X.shape[0]} escaneos × {NUM_BEAMS} haces × {NUM_FEATURES} features)")

    return X, y, valid


# ---------------------------------------------------------------------------
# C2.2 — Extracción del tensor temporal (para Deep Learning)
# ---------------------------------------------------------------------------

def extract_temporal_tensor(
    data_root: str | Path,
    pattern: str = "train_shard_*.npz",
    max_shards: Optional[int] = None,
    ventana_temporal: int = TIME_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrae el tensor temporal completo para la CNN Temporal (C5).

    Produce el TENSOR_TEMPORAL con la ventana deslizante completa.

    Args:
        data_root:        Directorio con shards .npz.
        pattern:          Patrón glob.
        max_shards:       Límite opcional.
        ventana_temporal: Tamaño de la ventana T. Rango válido: [2, 10].

    Returns:
        Tupla (X_temporal, y, valid_mask):
            X_temporal — [M, T, 360, K] float32
            y          — [M, 360]       uint8
            valid_mask — [M, 360]       uint8

    Raises:
        ValueError:    Si ventana_temporal está fuera de [2, 10].
        FileNotFoundError: Si no se encuentran shards.
    """
    # PRECONDICIÓN Wilson §5a
    if ventana_temporal < 2 or ventana_temporal > 10:
        raise ValueError(
            f"ventana_temporal={ventana_temporal} debe estar en [2, 10]"
        )

    data_root = Path(data_root)
    shard_files = sorted(data_root.glob(pattern))

    if not shard_files:
        raise FileNotFoundError(f"No se encontraron shards en {data_root}")
    if max_shards:
        # Pick one shard per bag_id to guarantee all scenarios are covered.
        # Falls back to strided if numpy/npz reading fails.
        import numpy as _np
        seen_bags, selected = set(), []
        for _s in shard_files:
            try:
                _bid = int(_np.load(_s, allow_pickle=True)['bag_id'].flat[0])
            except Exception:
                _bid = -len(selected)  # unique fallback key
            if _bid not in seen_bags:
                seen_bags.add(_bid)
                selected.append(_s)
        shard_files = selected[:max_shards] if max_shards < len(selected) else selected

    X_list, y_list, v_list = [], [], []

    for shard_path in shard_files:
        with np.load(shard_path, allow_pickle=True) as z:
            X_shard = z["X"].astype(np.float32)
            y_shard = z["y"].astype(np.uint8)
            v_shard = z["valid_current"].astype(np.uint8)

        # Ajustar ventana si el shard usa una T diferente
        T_shard = X_shard.shape[1]
        if T_shard < ventana_temporal:
            # usar la ventana disponible
            X_shard = X_shard
        else:
            # recortar a la ventana pedida (últimos T frames)
            X_shard = X_shard[:, -ventana_temporal:, :, :]

        X_list.append(X_shard)
        y_list.append(y_shard)
        v_list.append(v_shard)

    X     = np.concatenate(X_list, axis=0)
    y     = np.concatenate(y_list, axis=0)
    valid = np.concatenate(v_list, axis=0)

    # ASERCIÓN Wilson §5a: sin NaN
    for feat_idx in CONTINUOUS_FEATURE_IDXS:
        assert not np.isnan(X[:, :, :, feat_idx]).any(), \
            f"NaN en feature {FEATURE_NAMES[feat_idx]} del tensor temporal"

    print(f"[feature_engineering] TENSOR_TEMPORAL: {X.shape} "
          f"({X.shape[0]} muestras × T={X.shape[1]} × {NUM_BEAMS} haces × {NUM_FEATURES} features)")

    return X, y, valid


# ---------------------------------------------------------------------------
# C2.3 — Normalización
# ---------------------------------------------------------------------------

def compute_norm_stats(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Calcula media y desviación estándar por feature sobre features continuas.

    Args:
        X: Tensor de características [N, ..., K] o [N, K].

    Returns:
        dict con claves 'mean' y 'std', arrays float32 de longitud K.

    Raises:
        AssertionError: Si X contiene NaN.
    """
    assert not np.isnan(X).any(), "X contiene NaN — no se pueden calcular estadísticas"

    K = X.shape[-1]
    mean = np.zeros(K, dtype=np.float32)
    std  = np.ones(K,  dtype=np.float32)

    X_flat = X.reshape(-1, K).astype(np.float64)

    for feat_idx in CONTINUOUS_FEATURE_IDXS:
        vals = X_flat[:, feat_idx]
        mu   = vals.mean()
        sig  = vals.std()
        mean[feat_idx] = float(mu)
        std[feat_idx]  = float(sig) if sig > NORM_EPS else 1.0

    return {"mean": mean, "std": std}


def normalize_tensor(
    X: np.ndarray,
    stats: Dict[str, np.ndarray],
) -> np.ndarray:
    """Normaliza X usando las estadísticas pre-calculadas.

    Args:
        X:     Tensor [N, ..., K].
        stats: Dict con 'mean' y 'std' de longitud K.

    Returns:
        X normalizado, mismo shape, float32.
        Rango válido: aproximadamente [-3, 3] para features continuas.
    """
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std  = np.asarray(stats["std"],  dtype=np.float32)

    X_norm = X.copy().astype(np.float32)

    for feat_idx in CONTINUOUS_FEATURE_IDXS:
        X_norm[..., feat_idx] = (X[..., feat_idx] - mean[feat_idx]) / (std[feat_idx] + NORM_EPS)

    # POSTCONDICIÓN: sin NaN tras normalización
    assert not np.isnan(X_norm).any(), \
        "X_norm contiene NaN tras normalización — revisar stats"

    return X_norm


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import DATA_ROOT

    print("=== C2: Ingeniería de Características ===\n")

    print("-- TENSOR_CARACTERISTICAS (frame actual) --")
    X, y, valid = extract_feature_tensor(DATA_ROOT)
    stats = compute_norm_stats(X)
    X_norm = normalize_tensor(X, stats)
    print(f"   X normalizado: {X_norm.shape}, min={X_norm.min():.3f}, max={X_norm.max():.3f}")

    print("\n-- TENSOR_TEMPORAL (ventana completa) --")
    X_t, y_t, valid_t = extract_temporal_tensor(DATA_ROOT)
    print(f"   X_temporal: {X_t.shape}")

    print("\n-- Estadísticas de normalización --")
    for i, name in enumerate(FEATURE_NAMES):
        if i in CONTINUOUS_FEATURE_IDXS:
            print(f"   {name:25s}  mean={stats['mean'][i]:+.4f}  std={stats['std'][i]:.4f}")
