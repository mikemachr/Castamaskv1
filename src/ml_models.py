"""
ml_models.py — Componente C4: Machine Learning Clásico
=======================================================
Responsable: Juan Angel Lucio Rojas

Entrena tres clasificadores clásicos de sklearn como baseline interpretable
para la clasificación binaria de haces LiDAR (estático=0 / dinámico=1).
Produce una tabla comparativa de métricas y análisis de importancia
de características para comparar contra la CNN Temporal (C5).

Interfaces:
    train_classical_models(X, y) -> REPORTE_ML_CLASICO (dict)
    load_features_from_shards(data_root) -> X (np.ndarray), y (np.ndarray)

Referencias: Wilson et al. (2014) §5a (aserciones), §4b (modularización).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Root of the project (two levels up from src/)
_ROOT = Path(__file__).resolve().parent.parent
_FIGURES_DIR = _ROOT / "report" / "figures"

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_BEAMS: int = 360
FEATURE_NAMES: List[str] = [
    "range_norm",
    "delta_r",
    "static_residual",
    "abs_static_residual",
    "spatial_grad",
    "valid_mask",
]
RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_features_from_shards(
    data_root: str | Path,
    pattern: str = "train_shard_*.npz",
    max_shards: Optional[int] = None,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carga shards y devuelve features tabulares para ML clásico.

    Toma el frame temporal más reciente (T-1) de cada escaneo y aplana
    los haces a filas individuales: [N*360, K]. Solo incluye haces con
    valid_current=1.

    Args:
        data_root:   Directorio con archivos .npz.
        pattern:     Patrón glob para encontrar shards.
        max_shards:  Límite opcional de shards a cargar.
        train_ratio: Proporción de escaneos para entrenamiento.
        val_ratio:   Proporción de escaneos para validación.

    Returns:
        Tupla (X_train, X_val, X_test, y_train, y_val, y_test)
        donde X tiene forma [M, K] e y tiene forma [M].

    Raises:
        FileNotFoundError: Si no se encuentran shards.
        ValueError: Si los ratios de split no suman ≤ 1.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio debe ser < 1.0")

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

    all_X, all_y = [], []

    for shard_path in shard_files:
        with np.load(shard_path, allow_pickle=True) as z:
            X_shard     = z["X"].astype(np.float32)        # [N, T, 360, K]
            y_shard     = z["y"].astype(np.uint8)           # [N, 360]
            valid_shard = z["valid_current"].astype(np.uint8)  # [N, 360]

        N, T, H, K = X_shard.shape

        # Usar solo el frame actual (último en la ventana temporal)
        X_current = X_shard[:, -1, :, :]   # [N, 360, K]

        # Aplanar: cada haz es una muestra
        X_flat = X_current.reshape(N * H, K)      # [N*360, K]
        y_flat = y_shard.reshape(N * H)            # [N*360]
        v_flat = valid_shard.reshape(N * H).astype(bool)

        # Solo haces válidos
        all_X.append(X_flat[v_flat])
        all_y.append(y_flat[v_flat])

    X = np.vstack(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.int32)

    # ASERCIÓN Wilson §5a
    assert X.ndim == 2, f"X debe ser 2D, tiene shape {X.shape}"
    assert len(X) == len(y), "X e y deben tener el mismo número de muestras"
    assert not np.isnan(X).any(), "X contiene valores NaN"
    assert set(np.unique(y)).issubset({0, 1}), "y contiene valores fuera de {0,1}"

    # Split cronológico (sin mezclar para evitar data leakage)
    n = len(X)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    X_train, y_train = X[:n_train],          y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],    y[n_train+n_val:]

    print(f"[load_features] {n} haces válidos | "
          f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_pipelines() -> Dict[str, Pipeline]:
    """Construye los tres pipelines de sklearn."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                max_iter=500,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            )),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    DecisionTreeClassifier(
                max_depth=8,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ]),
    }


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calcula métricas de clasificación binaria."""
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = float("nan")

    # ASERCIÓN Wilson §5a: métricas en rango válido
    for k, v in metrics.items():
        if not np.isnan(v):
            assert 0.0 <= v <= 1.0, f"Métrica {k}={v} fuera de [0,1]"

    return metrics


def _extract_feature_importance(
    pipeline: Pipeline,
    model_name: str,
) -> np.ndarray:
    """Extrae importancia de características según el tipo de modelo."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return clf.feature_importances_
    if hasattr(clf, "coef_"):
        return np.abs(clf.coef_[0])
    return np.zeros(len(FEATURE_NAMES))


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

def train_classical_models(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_val:   np.ndarray,
    y_test:  np.ndarray,
    figures_dir: Optional[str | Path] = None,
) -> Dict:
    """Entrena y evalúa tres modelos clásicos de clasificación.

    Args:
        X_train, X_val, X_test: Features [M, K], rango adimensional.
        y_train, y_val, y_test: Etiquetas binarias {0, 1}.
        figures_dir: Directorio para guardar figuras. None = no guardar.

    Returns:
        REPORTE_ML_CLASICO: diccionario con:
            'comparison_table' (pd.DataFrame) — métricas de todos los modelos
            'best_model_name'  (str)
            'best_model'       (Pipeline sklearn entrenado)
            'feature_importance' (dict por modelo)
            'confusion_matrices' (dict por modelo)
            'figure_paths'     (list de str)

    Raises:
        ValueError: Si X o y tienen dimensiones incorrectas o contienen NaN.
        ValueError: Si el conjunto de entrenamiento tiene solo una clase.
    """
    # Validaciones de entrada
    if np.isnan(X_train).any():
        raise ValueError("X_train contiene valores NaN")
    if not set(np.unique(y_train)).issubset({0, 1}):
        raise ValueError("y_train contiene valores distintos de {0, 1}")
    if len(np.unique(y_train)) < 2:
        raise ValueError("El conjunto de entrenamiento contiene solo una clase")

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

    pipelines   = _build_pipelines()
    results     = {}
    figure_paths = []

    for name, pipeline in pipelines.items():
        print(f"[ml_models] Entrenando {name}...")
        pipeline.fit(X_train, y_train)

        y_pred_val  = pipeline.predict(X_val)
        y_pred_test = pipeline.predict(X_test)

        # Probabilidades para ROC-AUC (si el modelo las soporta)
        y_prob_val  = None
        y_prob_test = None
        if hasattr(pipeline.named_steps["clf"], "predict_proba"):
            y_prob_val  = pipeline.predict_proba(X_val)[:, 1]
            y_prob_test = pipeline.predict_proba(X_test)[:, 1]

        val_metrics  = _compute_metrics(y_val,  y_pred_val,  y_prob_val)
        test_metrics = _compute_metrics(y_test, y_pred_test, y_prob_test)

        results[name] = {
            "pipeline":           pipeline,
            "val_metrics":        val_metrics,
            "test_metrics":       test_metrics,
            "confusion_matrix":   confusion_matrix(y_test, y_pred_test),
            "feature_importance": _extract_feature_importance(pipeline, name),
        }

        print(f"  val  F1={val_metrics['f1']:.4f} "
              f"P={val_metrics['precision']:.4f} "
              f"R={val_metrics['recall']:.4f}")
        print(f"  test F1={test_metrics['f1']:.4f} "
              f"P={test_metrics['precision']:.4f} "
              f"R={test_metrics['recall']:.4f}")

    # Tabla comparativa
    rows = []
    for name, res in results.items():
        row = {"model": name}
        row.update({f"val_{k}": v for k, v in res["val_metrics"].items()})
        row.update({f"test_{k}": v for k, v in res["test_metrics"].items()})
        rows.append(row)
    comparison_table = pd.DataFrame(rows).set_index("model")

    # Mejor modelo por F1 en validación
    best_name = max(results, key=lambda n: results[n]["val_metrics"]["f1"])
    print(f"[ml_models] Mejor baseline: {best_name} "
          f"(val F1={results[best_name]['val_metrics']['f1']:.4f})")

    # --- Figura 1: Tabla de métricas comparativa ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    display_cols = ["val_f1", "val_precision", "val_recall",
                    "test_f1", "test_precision", "test_recall"]
    display_df = comparison_table[
        [c for c in display_cols if c in comparison_table.columns]
    ].round(4)
    tbl = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        rowLabels=display_df.index,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    ax.set_title("Comparativa de modelos clásicos — métricas val/test", pad=12)
    fig.tight_layout()
    if figures_dir:
        p = figures_dir / "ml_comparison_table.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        figure_paths.append(str(p))
    plt.close(fig)

    # --- Figura 2: Importancia de características ---
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        imp = res["feature_importance"]
        ax.barh(FEATURE_NAMES, imp, color="steelblue")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Importancia relativa [adim.]")
    fig.suptitle("Importancia de características por modelo", y=1.02)
    fig.tight_layout()
    if figures_dir:
        p = figures_dir / "ml_feature_importance.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        figure_paths.append(str(p))
    plt.close(fig)

    # --- Figura 3: Matrices de confusión ---
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        cm = res["confusion_matrix"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Estático", "Dinámico"])
        ax.set_yticklabels(["Estático", "Dinámico"])
        ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
        ax.set_title(name, fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle("Matrices de confusión — conjunto de prueba")
    fig.tight_layout()
    if figures_dir:
        p = figures_dir / "ml_confusion_matrices.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        figure_paths.append(str(p))
    plt.close(fig)

    return {
        "comparison_table":   comparison_table,
        "best_model_name":    best_name,
        "best_model":         results[best_name]["pipeline"],
        "feature_importance": {n: r["feature_importance"] for n, r in results.items()},
        "confusion_matrices": {n: r["confusion_matrix"]   for n, r in results.items()},
        "all_results":        results,
        "figure_paths":       figure_paths,
    }


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import DATA_ROOT

    X_train, X_val, X_test, y_train, y_val, y_test = load_features_from_shards(
        DATA_ROOT, figures_dir=None
    )

    report = train_classical_models(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        figures_dir=_FIGURES_DIR,
    )

    print("\n=== Tabla comparativa ===")
    print(report["comparison_table"].to_string())
    print(f"\nMejor modelo: {report['best_model_name']}")
