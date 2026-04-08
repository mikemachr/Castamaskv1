"""
main.py — Componente C7: Orquestación y Reproducibilidad
=========================================================
Responsable: Ricardo Daniel Damián Cortez

Ejecuta el pipeline completo de CastaMask de principio a fin:
C1 → C2 → C3 → C4 → C5 → C6

Uso:
    python main.py                  # pipeline completo
    python main.py --skip-opt       # omite búsqueda de hiperparámetros (lento)
    python main.py --dry-run        # ejecución rápida de verificación (~30s)
    python main.py --fold 0         # entrena solo el fold indicado
    python main.py --device cuda    # usa GPU si está disponible

Referencias: Wilson et al. (2014) §2c (build tool), §3b (VCS), §5b (pruebas).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — must happen before ANY local imports.
# ROOT is always the directory containing main.py, regardless of cwd.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# All output paths anchored to ROOT — always land next to main.py.
REPORT_DIR      = ROOT / "report"
FIGURES_DIR     = ROOT / "report" / "figures"
EXPERIMENTS_DIR = ROOT / "experiments_fullscan"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CastaMask — pipeline completo de filtrado dinamico LiDAR"
    )
    parser.add_argument("--fold",      type=int,  default=0,
                        help="Fold a entrenar (default: 0)")
    parser.add_argument("--device",    type=str,  default="cpu",
                        help="Dispositivo: cpu o cuda (default: cpu)")
    parser.add_argument("--skip-opt",  action="store_true",
                        help="Omitir busqueda de hiperparametros (usa config.py)")
    parser.add_argument("--skip-cv",   action="store_true",
                        help="Omitir validacion cruzada completa")
    parser.add_argument("--data-root", type=str,  default=None,
                        help="Ruta a los shards (override de config.DATA_ROOT)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Ejecucion rapida: 2 shards, 1 epoca, solo LogReg (~30 s)")
    parser.add_argument("--fast",       action="store_true",
                        help="Entrena con datos reales limitados (todas las familias, menos shards)")
    parser.add_argument("--max-shards", type=int, default=13,
                        help="Shards por split en --fast / shards totales en C1/C2/C4 (default: 13 = 1 por bag_id)")
    return parser.parse_args()


def _apply_dry_run_settings() -> None:
    """Reduce todos los parametros costosos al minimo funcional."""
    import config as cfg
    cfg.MAX_EPOCHS              = 1
    cfg.EARLY_STOPPING_PATIENCE = 1
    cfg.BATCH_SIZE              = 8
    cfg.PRINT_EVERY_N_STEPS     = 9999
    print("  [DRY-RUN] MAX_EPOCHS=1  BATCH_SIZE=8  max_shards=2")


# ---------------------------------------------------------------------------
# Pipeline steps  — all figures_dir / report_dir use ROOT-anchored constants
# ---------------------------------------------------------------------------

def step_eda(data_root: Path, dry_run: bool = False, max_shards: int = None) -> dict:
    print("\n" + "="*60)
    print("C1 — Carga de Datos y EDA")
    print("="*60)
    from data_loader import load_multiple_shards, clean_data, eda_summary

    shard_cap = 2 if dry_run else max_shards  # strided in load_multiple_shards
    df_raw   = load_multiple_shards(data_root, max_shards=shard_cap)
    df_clean = clean_data(df_raw)
    summary  = eda_summary(df_clean, figures_dir=FIGURES_DIR)

    print(f"  Escaneos cargados : {len(df_raw)}")
    print(f"  Escaneos limpios  : {len(df_clean)}")
    print(f"  Tasa dinamica     : {summary['class_balance']['overall_dynamic_rate']:.3f}")
    return {"df": df_clean, "summary": summary}


def step_features(data_root: Path, dry_run: bool = False, max_shards: int = None) -> dict:
    print("\n" + "="*60)
    print("C2 — Ingenieria de Caracteristicas")
    print("="*60)
    from feature_engineering import (
        extract_feature_tensor, extract_temporal_tensor, compute_norm_stats,
    )

    shard_cap = 2 if dry_run else max_shards  # strided in extract_*
    X, y, valid    = extract_feature_tensor(data_root, max_shards=shard_cap)
    X_t, y_t, v_t = extract_temporal_tensor(data_root, max_shards=shard_cap)
    norm_stats     = compute_norm_stats(X)

    return {
        "X": X, "y": y, "valid": valid,
        "X_temporal": X_t, "y_temporal": y_t,
        "norm_stats": norm_stats,
    }


def step_optimize(data_root: Path, skip: bool, dry_run: bool = False, fast: bool = False, max_shards: int = 4) -> dict:
    print("\n" + "="*60)
    print("C3 — Optimizacion de Hiperparametros")
    print("="*60)

    if skip or dry_run or fast:
        import config as cfg
        best_config = {
            "learning_rate": cfg.LEARNING_RATE,
            "time_window":   cfg.TIME_WINDOW,
            "conv1_out":     cfg.CONV1_OUT,
            "val_f1":        None,
        }
        reason = "[DRY-RUN]" if dry_run else "[FAST/OMITIDO]" if fast else "[OMITIDO]"
        print(f"  {reason} Usando hiperparametros de config.py")
        print(f"  learning_rate = {best_config['learning_rate']}")
        print(f"  time_window   = {best_config['time_window']}")
        return best_config

    from optimizer import optimize_hyperparameters

    if fast:
        _patch_shard_cap(max_shards)

    small_space = {
        "learning_rate": [1e-3, 5e-4],
        "time_window":   [7],
        "conv1_out":     [16],
    }
    best_config = optimize_hyperparameters(
        data_root=data_root,
        search_space=small_space,
        figures_dir=FIGURES_DIR,
    )
    return best_config


def step_classical_ml(data_root: Path, dry_run: bool = False, max_shards: int = None) -> dict:
    print("\n" + "="*60)
    print("C4 — Machine Learning Clasico")
    print("="*60)
    print("  Importando sklearn...", flush=True)
    from ml_models import load_features_from_shards, train_classical_models, _build_pipelines, _compute_metrics
    print("  Cargando features...", flush=True)

    # dry_run=2 shards; fast uses max_shards; full run uses None (all)
    # strided selection inside load_features_from_shards ensures all bag_ids covered
    shard_cap = 2 if dry_run else max_shards
    X_train, X_val, X_test, y_train, y_val, y_test = load_features_from_shards(
        data_root, max_shards=shard_cap,
    )
    import numpy as np
    import pandas as pd
    print("Train classes:", np.unique(y_train, return_counts=True))
    print("Val classes:", np.unique(y_val, return_counts=True))
    print("Test classes:", np.unique(y_test, return_counts=True))
    # Sub sample a lot, to have a fast runtime for dry run 
    if dry_run:

        # Sample randomly (not [:n]) so both classes are represented


        def _balanced_sample(X, y, n, seed):
            rng = np.random.default_rng(seed)
            n = min(n, len(X))
            idx = rng.choice(len(X), size=n, replace=False)
            return X[idx], y[idx]

        X_train, y_train = _balanced_sample(X_train, y_train, 2000, 42)
        X_val,   y_val   = _balanced_sample(X_val,   y_val,   500,  43)
        X_test,  y_test  = _balanced_sample(X_test,  y_test,  500,  44)
        print(f"  [DRY-RUN] Solo Logistic Regression — {len(X_train)} filas, clases: {np.unique(y_train).tolist()}")
        pipelines = {"Logistic Regression": _build_pipelines()["Logistic Regression"]}
        results = {}
        for name, pipeline in pipelines.items():
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            metrics = _compute_metrics(y_test, y_pred)
            results[name] = {
                "pipeline":           pipeline,
                "val_metrics":        metrics,
                "test_metrics":       metrics,
                "confusion_matrix":   np.zeros((2, 2), dtype=int),
                "feature_importance": np.zeros(6),
            }
        rows = [
            {"model": n, **{f"test_{k}": v for k, v in r["test_metrics"].items()}}
            for n, r in results.items()
        ]
        return {
            "comparison_table":  pd.DataFrame(rows).set_index("model"),
            "best_model_name":   "Logistic Regression",
            "best_model":        results["Logistic Regression"]["pipeline"],
            "feature_importance": {},
            "confusion_matrices": {},
            "all_results":       results,
            "figure_paths":      [],
        }

    ml_report = train_classical_models(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        figures_dir=FIGURES_DIR,
    )
    print(f"  Mejor baseline: {ml_report['best_model_name']}")
    return ml_report


def _patch_shard_cap(max_shards_per_split: int) -> None:
    """Caps shards per split in train_one_fold.
    Hooks into filter_shard_index_by_bags which is called AFTER family/bag
    filtering — so each split already contains only the right bag_ids.
    Strided selection is safe here because family membership is already guaranteed.
    """
    import shard_index as si
    _orig = si.filter_shard_index_by_bags

    def _capped(shard_index, allowed_bag_ids):
        result = _orig(shard_index, allowed_bag_ids)
        # strided so we sample across the full temporal range of each family
        step   = max(1, len(result) // max_shards_per_split)
        capped = result[::step][:max_shards_per_split]
        print(f"  [FAST/DRY] split: {len(result)} shards -> {len(capped)}")
        return capped

    si.filter_shard_index_by_bags = _capped


def step_deep_learning(fold_id: int, device: str, best_config: dict,
                       fast: bool = False, dry_run: bool = False,
                       max_shards: int = 4) -> dict:
    print("\n" + "="*60)
    print("C5 — Deep Learning (CNN Temporal)")
    print("="*60)

    import config as cfg
    if best_config.get("learning_rate"):
        cfg.LEARNING_RATE = float(best_config["learning_rate"])
    if best_config.get("time_window"):
        cfg.TIME_WINDOW = int(best_config["time_window"])
    if best_config.get("conv1_out"):
        cfg.CONV1_OUT = int(best_config["conv1_out"])

    # Point experiments dir to ROOT so checkpoints land in the right place
    cfg.EXPERIMENTS_ROOT = EXPERIMENTS_DIR

    # Cap shards per split: dry-run uses 2, fast uses max_shards, full uses None
    if dry_run:
        _patch_shard_cap(2)
    elif fast:
        _patch_shard_cap(max_shards)

    from dl_model import train_model
    dl_result = train_model(fold_id=fold_id, device=device, figures_dir=FIGURES_DIR)
    return dl_result


def step_visualize(ml_report: dict, dl_result: dict) -> dict:
    print("\n" + "="*60)
    print("C6 — Visualizacion y Reporte")
    print("="*60)
    from viz import generate_report

    history_path = dl_result.get(
        "history_path",
        str(EXPERIMENTS_DIR / "fold_00" / "history.json"),
    )
    results = {
        "ml_report":    ml_report,
        "dl_result":    dl_result,
        "history_path": history_path,
    }
    final = generate_report(
        results=results,
        figures_dir=FIGURES_DIR,
        report_dir=REPORT_DIR,
    )
    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Override DATA_ROOT before any import that reads config
    if args.data_root:
        import config as cfg
        cfg.DATA_ROOT = Path(args.data_root)

    # Apply dry-run caps before any component reads config
    if args.dry_run:
        _apply_dry_run_settings()

    from config import DATA_ROOT
    data_root = Path(DATA_ROOT)

    if not data_root.exists():
        print(f"ERROR: DATA_ROOT no existe: {data_root}")
        sys.exit(1)

    mode = " [DRY-RUN]" if args.dry_run else ""
    print("\n" + "="*60)
    print(f"CastaMask — Pipeline de Filtrado Dinamico LiDAR 2D{mode}")
    print("TC6039.1 Applied Computing — Tec de Monterrey")
    print("="*60)
    print(f"  ROOT      : {ROOT}")
    print(f"  DATA_ROOT : {data_root}")
    print(f"  fold      : {args.fold}")
    print(f"  device    : {args.device}")

    t_total     = time.time()
    all_results = {}

    try:
        # shard_cap: None = full dataset; int = strided cap across all files
        shard_cap = None
        if args.dry_run: shard_cap = 2
        elif args.fast:  shard_cap = args.max_shards

        t0 = time.time()
        step_eda(data_root, dry_run=args.dry_run, max_shards=shard_cap)
        all_results["eda"] = {"time_s": round(time.time() - t0, 2)}

        t0 = time.time()
        step_features(data_root, dry_run=args.dry_run, max_shards=shard_cap)
        all_results["features"] = {"time_s": round(time.time() - t0, 2)}

        t0 = time.time()
        best_config = step_optimize(data_root, skip=args.skip_opt, dry_run=args.dry_run, fast=args.fast, max_shards=args.max_shards)
        all_results["optimizer"] = {"time_s": round(time.time() - t0, 2)}

        t0 = time.time()
        ml_report = step_classical_ml(data_root, dry_run=args.dry_run, max_shards=shard_cap)
        all_results["ml"] = {
            "time_s":     round(time.time() - t0, 2),
            "best_model": ml_report["best_model_name"],
        }

        t0 = time.time()
        dl_result = step_deep_learning(args.fold, args.device, best_config,
                                       fast=args.fast, dry_run=args.dry_run,
                                       max_shards=args.max_shards)
        all_results["dl"] = {
            "time_s":       round(time.time() - t0, 2),
            "test_metrics": dl_result.get("test_metrics", {}),
        }

        t0 = time.time()
        final_results = step_visualize(ml_report, dl_result)
        all_results["viz"] = {"time_s": round(time.time() - t0, 2)}

    except Exception as e:
        print(f"\nERROR en el pipeline: {e}")
        raise

    total_time = time.time() - t_total
    print("\n" + "="*60)
    print(f"Pipeline completado{mode}  ({total_time:.1f}s)")
    print("="*60)
    for step, info in all_results.items():
        print(f"  {step:12s}  {info['time_s']}s")

    if "ml" in all_results:
        print(f"\n  Mejor modelo clasico : {all_results['ml'].get('best_model')}")
    if "dl" in all_results:
        tm = all_results["dl"].get("test_metrics", {})
        print(f"  CNN F1               : {tm.get('f1', 0):.4f}")
        print(f"  CNN Recall           : {tm.get('recall', 0):.4f}")

    print(f"\n  Reporte LaTeX : {REPORT_DIR / 'reporte_final.tex'}")
    print(f"  Figuras       : {FIGURES_DIR}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORT_DIR / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        safe = {
            k: {kk: str(vv) if not isinstance(vv, (int, float, str, dict, list)) else vv
                for kk, vv in v.items()}
            for k, v in all_results.items()
        }
        json.dump(safe, f, indent=2)
    print(f"  Resumen JSON  : {summary_path}")


if __name__ == "__main__":
    main()
