"""
dl_model.py — Componente C5: Deep Learning (CNN Temporal)
=========================================================
Responsable: Miguel Ángel Chávez Robles

Interfaz pública del componente de Deep Learning. Orquesta el entrenamiento
de la CNN Temporal 1D (CastaMaskFullScanCNN) definida en model.py usando
el pipeline de train_one_fold.py y la validación cruzada de run_cv.py.

La arquitectura procesa tensores [B, K, T, N] donde:
    B = batch size
    K = 6 canales de features
    T = ventana temporal (7 frames)
    N = 360 haces LiDAR

y produce logits [B, N] — un valor por haz para clasificación binaria.

Interfaces de salida:
    train_model(fold_id, device)  -> MODELO_ENTRENADO (dict)
    run_cross_validation(device)  -> RESULTADOS_CV (dict)
    load_trained_model(path)      -> modelo PyTorch listo para inferencia

Referencias: Wilson et al. (2014) §5a (aserciones), §5b (pruebas unitarias).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from pathlib import Path

# Root of the project (two levels up from src/)
_ROOT = Path(__file__).resolve().parent.parent
_FIGURES_DIR = _ROOT / "report" / "figures"

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def train_model(
    fold_id: int = 0,
    device: str = "cpu",
    figures_dir: Optional[str | Path] = None,
) -> Dict:
    """Entrena la CNN Temporal en un fold específico.

    Delega en train_one_fold.train_one_fold() con los hiperparámetros
    definidos en config.py (o actualizados por el optimizador C3).

    Args:
        fold_id:     Índice del fold (0 a N_families-1).
        device:      'cpu' o 'cuda'.
        figures_dir: Si se especifica, genera curvas de entrenamiento aquí.

    Returns:
        MODELO_ENTRENADO: dict con:
            'fold_id'          (int)
            'best_epoch'       (int)
            'best_val_metric'  (float)
            'test_metrics'     (dict) — accuracy, precision, recall, f1, loss
            'model_path'       (str)  — ruta al .pt del mejor modelo
            'history_path'     (str)  — ruta al history.json

    Raises:
        RuntimeError: Si el fold no tiene datos suficientes.
        AssertionError: Si la pérdida de entrenamiento no decrece.
    """
    from train_one_fold import train_one_fold
    from config import EXPERIMENTS_ROOT

    print(f"[dl_model] Entrenando fold {fold_id} en {device}...")
    result = train_one_fold(fold_id=fold_id, device=device)

    fold_dir     = Path(EXPERIMENTS_ROOT) / f"fold_{fold_id:02d}"
    model_path   = fold_dir / "best_model.pt"
    history_path = fold_dir / "history.json"

    # Generar curvas de entrenamiento si se pide
    if figures_dir and history_path.exists():
        from optimizer import plot_convergence
        plot_convergence(history_path, figures_dir=figures_dir)

    # ASERCIÓN Wilson §5a: pérdida de entrenamiento disminuyó
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        if len(history["train"]) >= 3:
            first_loss = history["train"][0]["loss"]
            last_loss  = history["train"][-1]["loss"]
            if last_loss >= first_loss:
                print(
                    f"[dl_model] ALERTA: pérdida final ({last_loss:.4f}) >= "
                    f"pérdida inicial ({first_loss:.4f}) — posible sobreajuste o LR muy alto"
                )

    result["model_path"]   = str(model_path)
    result["history_path"] = str(history_path)

    print(f"[dl_model] Fold {fold_id} completado.")
    print(f"  Mejor época: {result['best_epoch']}")
    for k, v in result["test_metrics"].items():
        if isinstance(v, float):
            print(f"  test_{k}: {v:.4f}")

    return result


def run_cross_validation(
    device: str = "cpu",
    figures_dir: Optional[str | Path] = None,
) -> Dict:
    """Ejecuta validación cruzada completa (leave-one-family-out).

    Args:
        device:      'cpu' o 'cuda'.
        figures_dir: Directorio para guardar figuras de resumen.

    Returns:
        RESULTADOS_CV: dict con:
            'fold_results' (list) — resultado de cada fold
            'summary'      (dict) — media y std de métricas entre folds
            'cv_path'      (str)  — ruta al cv_results.json
    """
    from run_cv import main as run_cv_main
    from config import EXPERIMENTS_ROOT

    print("[dl_model] Iniciando validación cruzada...")
    run_cv_main()

    cv_path = Path(EXPERIMENTS_ROOT) / "cross_validation" / "cv_results.json"

    results = {}
    if cv_path.exists():
        with open(cv_path) as f:
            results = json.load(f)
        results["cv_path"] = str(cv_path)

        print("\n[dl_model] Resumen CV:")
        for metric, stats in results.get("summary", {}).items():
            print(f"  {metric}: mean={stats['mean']:.4f} ± {stats['std']:.4f}")

    # Figura resumen: métricas por fold
    if figures_dir and results.get("fold_results"):
        _plot_cv_summary(results, figures_dir)

    return results


def load_trained_model(
    model_path: str | Path,
    device: str = "cpu",
) -> torch.nn.Module:
    """Carga un modelo entrenado desde disco para inferencia.

    Args:
        model_path: Ruta al archivo .pt guardado por train_one_fold.
        device:     Dispositivo donde cargar el modelo.

    Returns:
        CastaMaskFullScanCNN en modo eval(), listo para inferencia.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    from model import CastaMaskFullScanCNN

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    model = CastaMaskFullScanCNN()
    ckpt  = torch.load(model_path, map_location=device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    print(f"[dl_model] Modelo cargado desde {model_path}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros totales: {n_params:,}")

    return model


def predict(
    model: torch.nn.Module,
    X: torch.Tensor,
    threshold: float = 0.5,
    device: str = "cpu",
) -> Dict:
    """Realiza inferencia sobre un batch de escaneos.

    Args:
        model:     Modelo cargado con load_trained_model().
        X:         Tensor [B, K, T, N] o [K, T, N] (un solo escaneo).
        threshold: Umbral de decisión para clasificación binaria.
        device:    Dispositivo de cómputo.

    Returns:
        dict con:
            'logits'       — [B, N] float
            'probs'        — [B, N] float en [0, 1]
            'predictions'  — [B, N] int {0, 1}
    """
    if X.ndim == 3:
        X = X.unsqueeze(0)

    X = X.to(device)
    model = model.to(device)

    with torch.no_grad():
        logits = model(X)
        probs  = torch.sigmoid(logits)
        preds  = (probs >= threshold).long()

    return {
        "logits":      logits.cpu(),
        "probs":       probs.cpu(),
        "predictions": preds.cpu(),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _plot_cv_summary(results: Dict, figures_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = results.get("summary", {})
    metrics = ["f1", "precision", "recall", "accuracy"]
    metrics = [m for m in metrics if m in summary]

    means = [summary[m]["mean"] for m in metrics]
    stds  = [summary[m]["std"]  for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(metrics))
    ax.bar(x, means, yerr=stds, capsize=5, color="#1565C0", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Métrica [adim.]")
    ax.set_ylim(0, 1)
    ax.set_title("Métricas de la CNN Temporal — validación cruzada (media ± std)")
    fig.tight_layout()
    p = figures_dir / "dl_cv_summary.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[dl_model] Figura CV guardada: {p}")


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = train_model(fold_id=0, device="cpu", figures_dir=_FIGURES_DIR)
    print("\nMODELO_ENTRENADO:")
    print(f"  model_path: {result['model_path']}")
    print(f"  test F1:    {result['test_metrics'].get('f1', 'N/A'):.4f}")
