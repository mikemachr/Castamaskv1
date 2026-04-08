"""
optimizer.py — Componente C3: Optimización de Hiperparámetros
=============================================================
Responsable: Priscila de los Ángeles Correa Miranda

Realiza búsqueda sistemática de hiperparámetros para la CNN Temporal (C5)
usando búsqueda en cuadrícula (grid search) sobre tasa de aprendizaje,
tamaño de ventana temporal y configuración de capas convolucionales.

Produce MEJOR_CONFIGURACION: diccionario con los hiperparámetros óptimos
que maximizan el F1-score en el conjunto de validación.

Interfaz de salida:
    optimize_hyperparameters(data_root) -> MEJOR_CONFIGURACION (dict)
    plot_convergence(history)           -> figura de convergencia

Referencias: Wilson et al. (2014) §5a (aserciones), §6a (profiling).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Root of the project (two levels up from src/)
_ROOT = Path(__file__).resolve().parent.parent
_FIGURES_DIR = _ROOT / "report" / "figures"

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

SEARCH_SPACE: Dict[str, List[Any]] = {
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "time_window":   [5, 7],
    "conv1_out":     [16, 32],
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "learning_rate": 1e-3,
    "time_window":   7,
    "conv1_out":     16,
    "conv2_out":     32,
    "conv3_out":     32,
    "dropout":       0.1,
    "batch_size":    16,
    "max_epochs":    5,       # reducido para búsqueda rápida
    "weight_decay":  1e-4,
    "optimizer":     "adamw",
}


# ---------------------------------------------------------------------------
# Single-config evaluator
# ---------------------------------------------------------------------------

def _evaluate_config(
    config: Dict[str, Any],
    data_root: Path,
    fold_id: int = 0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Entrena un fold con la configuración dada y devuelve métricas de validación.

    Args:
        config:    Diccionario de hiperparámetros.
        data_root: Directorio de datos (usado para actualizar config global).
        fold_id:   Fold a usar para la evaluación.
        device:    'cpu' o 'cuda'.

    Returns:
        dict con 'val_f1', 'val_loss', 'config', 'train_time_s'.
    """
    import importlib
    import config as cfg_module

    # Parchear temporalmente los valores de config
    original_lr    = cfg_module.LEARNING_RATE
    original_tw    = cfg_module.TIME_WINDOW
    original_c1    = cfg_module.CONV1_OUT
    original_ep    = cfg_module.MAX_EPOCHS

    cfg_module.LEARNING_RATE = float(config["learning_rate"])
    cfg_module.TIME_WINDOW   = int(config["time_window"])
    cfg_module.CONV1_OUT     = int(config["conv1_out"])
    cfg_module.MAX_EPOCHS    = int(config.get("max_epochs", 5))

    try:
        # Re-importar módulos que dependen de config
        import train_one_fold
        importlib.reload(train_one_fold)

        t0 = time.time()
        result = train_one_fold.train_one_fold(fold_id=fold_id, device=device)
        elapsed = time.time() - t0

        # Extraer F1 de validación del historial
        history_path = Path(cfg_module.EXPERIMENTS_ROOT) / f"fold_{fold_id:02d}" / "history.json"
        val_f1 = 0.0
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            if history.get("val"):
                val_f1 = max(ep["f1"] for ep in history["val"])

        return {
            "config":       config,
            "val_f1":       float(val_f1),
            "val_loss":     float(result.get("best_val_metric", float("inf"))),
            "train_time_s": float(elapsed),
        }

    finally:
        # Restaurar config original
        cfg_module.LEARNING_RATE = original_lr
        cfg_module.TIME_WINDOW   = original_tw
        cfg_module.CONV1_OUT     = original_c1
        cfg_module.MAX_EPOCHS    = original_ep


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def optimize_hyperparameters(
    data_root: str | Path,
    search_space: Optional[Dict[str, List[Any]]] = None,
    base_config: Optional[Dict[str, Any]] = None,
    fold_id: int = 0,
    device: str = "cpu",
    figures_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Búsqueda en cuadrícula sobre el espacio de hiperparámetros.

    Itera sobre todas las combinaciones de learning_rate × time_window ×
    conv1_out, entrena un fold rápido (max_epochs=5) y selecciona la
    configuración que maximiza el F1-score de validación.

    Args:
        data_root:    Directorio de datos.
        search_space: Espacio de búsqueda. Si es None, usa SEARCH_SPACE.
        base_config:  Configuración base. Si es None, usa DEFAULT_CONFIG.
        fold_id:      Fold a usar para evaluación.
        device:       'cpu' o 'cuda'.
        figures_dir:  Directorio para guardar la curva de convergencia.

    Returns:
        MEJOR_CONFIGURACION: dict con los hiperparámetros óptimos:
            'learning_rate'  (float, rango [1e-5, 1e-1])
            'time_window'    (int,   rango [3, 10])
            'conv1_out'      (int)
            'conv2_out'      (int)
            'conv3_out'      (int)
            'dropout'        (float)
            'batch_size'     (int)
            'val_f1'         (float, rango [0, 1])
            'search_results' (list)

    Raises:
        ValueError: Si el espacio de búsqueda está vacío.
        RuntimeError: Si ninguna configuración converge (F1 < 0).
    """
    if search_space is None:
        search_space = SEARCH_SPACE
    if base_config is None:
        base_config = DEFAULT_CONFIG.copy()

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

    # Generar todas las combinaciones
    from itertools import product

    keys   = list(search_space.keys())
    values = list(search_space.values())
    combos = list(product(*values))

    if not combos:
        raise ValueError("El espacio de búsqueda está vacío")

    print(f"[optimizer] Iniciando grid search: {len(combos)} configuraciones")
    print(f"[optimizer] Parámetros: {keys}")

    search_results = []
    best_config    = None
    best_f1        = -float("inf")

    for i, combo in enumerate(combos):
        config = base_config.copy()
        config.update(dict(zip(keys, combo)))

        print(f"\n[optimizer] Config {i+1}/{len(combos)}: {dict(zip(keys, combo))}")

        try:
            result = _evaluate_config(config, Path(data_root), fold_id, device)
            val_f1 = result["val_f1"]

            # ASERCIÓN Wilson §5a: F1 en rango físico válido
            assert 0.0 <= val_f1 <= 1.0, \
                f"F1-score={val_f1} fuera del rango físico [0, 1]"

            search_results.append(result)
            print(f"[optimizer]   val_f1={val_f1:.4f}  tiempo={result['train_time_s']:.1f}s")

            if val_f1 > best_f1:
                best_f1     = val_f1
                best_config = config.copy()
                best_config["val_f1"] = val_f1

        except Exception as e:
            print(f"[optimizer]   FALLO: {e}")
            search_results.append({"config": config, "val_f1": 0.0, "error": str(e)})

    if best_config is None:
        raise RuntimeError("Ninguna configuración produjo resultados válidos")

    if best_f1 < 0.5:
        print(f"[optimizer] ALERTA: mejor F1={best_f1:.4f} < 0.5 — "
              "el modelo base puede no ser válido o los datos son muy ruidosos")

    best_config["search_results"] = search_results
    best_config["search_space"]   = search_space

    print(f"\n[optimizer] Mejor configuración: F1={best_f1:.4f}")
    for k, v in best_config.items():
        if k not in ("search_results", "search_space"):
            print(f"  {k}: {v}")

    # --- Figura: curva de F1 por configuración ---
    fig, ax = plt.subplots(figsize=(8, 4))
    f1_values = [r.get("val_f1", 0.0) for r in search_results]
    labels    = [
        f"lr={r['config']['learning_rate']:.0e}\nT={r['config']['time_window']}"
        for r in search_results
    ]
    colors = ["#2196F3" if v == best_f1 else "#90CAF9" for v in f1_values]
    ax.bar(range(len(f1_values)), f1_values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("F1-score en validación [adim.]")
    ax.set_ylim(0, 1)
    ax.set_title("Grid search — F1 por configuración de hiperparámetros")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="umbral mínimo")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if figures_dir:
        p = figures_dir / "optimizer_grid_search.png"
        fig.savefig(p, dpi=150)
        print(f"[optimizer] Figura guardada: {p}")
    plt.close(fig)

    return best_config


def plot_convergence(
    history_path: str | Path,
    figures_dir: Optional[str | Path] = None,
    show: bool = False,
) -> str | None:
    """Grafica curvas de pérdida y F1 de entrenamiento vs validación.

    Args:
        history_path: Ruta al archivo history.json generado por train_one_fold.
        figures_dir:  Directorio donde guardar la figura.
        show:         Si True, llama plt.show().

    Returns:
        Ruta de la figura guardada, o None si no se guardó.
    """
    history_path = Path(history_path)
    if not history_path.exists():
        print(f"[optimizer] history.json no encontrado: {history_path}")
        return None

    with open(history_path) as f:
        history = json.load(f)

    train_epochs = [ep["epoch"] for ep in history["train"]]
    train_loss   = [ep["loss"]  for ep in history["train"]]
    train_f1     = [ep["f1"]    for ep in history["train"]]
    val_loss     = [ep["loss"]  for ep in history["val"]]
    val_f1       = [ep["f1"]    for ep in history["val"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(train_epochs, train_loss, label="Train", color="#1565C0")
    ax1.plot(train_epochs, val_loss,   label="Val",   color="#E53935", linestyle="--")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida BCE [adim.]")
    ax1.set_title("Curva de pérdida")
    ax1.legend()

    ax2.plot(train_epochs, train_f1, label="Train", color="#1565C0")
    ax2.plot(train_epochs, val_f1,   label="Val",   color="#E53935", linestyle="--")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("F1-score [adim.]")
    ax2.set_ylim(0, 1)
    ax2.set_title("Curva de F1-score")
    ax2.legend()

    fig.suptitle("Curvas de entrenamiento — CNN Temporal (CastaMask)")
    fig.tight_layout()

    saved_path = None
    if figures_dir:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        p = figures_dir / "training_curves.png"
        fig.savefig(p, dpi=150)
        saved_path = str(p)

    if show:
        plt.show()
    plt.close(fig)

    return saved_path


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import DATA_ROOT

    # Búsqueda reducida para demo rápida
    small_space = {
        "learning_rate": [1e-3, 1e-4],
        "time_window":   [7],
        "conv1_out":     [16],
    }

    best = optimize_hyperparameters(
        data_root=DATA_ROOT,
        search_space=small_space,
        figures_dir=_FIGURES_DIR,
    )

    print("\nMEJOR_CONFIGURACION:")
    for k, v in best.items():
        if k not in ("search_results", "search_space"):
            print(f"  {k}: {v}")
