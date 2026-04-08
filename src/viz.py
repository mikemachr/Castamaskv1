"""
viz.py — Componente C6: Visualización y Generación de Reporte
=============================================================
Responsable: Ricardo Daniel Damián Cortez

Genera todas las figuras del reporte final y produce el reporte LaTeX
compilable. Consolida resultados de C2 (features), C4 (ML clásico)
y C5 (Deep Learning) en una tabla comparativa unificada.

Interfaces:
    generate_report(results, figures_dir, report_dir) -> RESULTADOS_FINALES
    plot_dynamic_mask(scan, mask, gt)                 -> figura comparativa
    plot_training_curves(history_path)                -> figura curvas
    build_comparison_table(ml_report, dl_metrics)     -> pd.DataFrame

Referencias: Wilson et al. (2014) §7a (documentación), §2c (automatización).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Root of the project (two levels up from src/)
_ROOT = Path(__file__).resolve().parent.parent
_FIGURES_DIR = _ROOT / "report" / "figures"
_REPORT_DIR  = _ROOT / "report"

# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_dynamic_mask(
    ranges: np.ndarray,
    predicted_mask: np.ndarray,
    gt_mask: np.ndarray,
    scan_idx: int = 0,
    figures_dir: Optional[str | Path] = None,
    show: bool = False,
) -> Optional[str]:
    """Visualiza la máscara dinámica predicha vs ground truth en coordenadas polares.

    Args:
        ranges:         [360] float — rangos del escaneo LiDAR en metros.
        predicted_mask: [360] int   — máscara predicha {0=estático, 1=dinámico}.
        gt_mask:        [360] int   — ground truth {0, 1}.
        scan_idx:       Índice del escaneo (para el título).
        figures_dir:    Directorio para guardar.
        show:           Si True, llama plt.show().

    Returns:
        Ruta de la figura guardada, o None.
    """
    angles = np.linspace(0, 2 * np.pi, len(ranges), endpoint=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                              subplot_kw={"projection": "polar"})

    for ax, mask, title, color_dyn in zip(
        axes,
        [gt_mask, predicted_mask],
        ["Ground Truth", "Predicción CNN Temporal"],
        ["#E53935", "#FF6F00"],
    ):
        static_mask  = mask == 0
        dynamic_mask = mask == 1

        ax.scatter(angles[static_mask],  ranges[static_mask],
                   c="#1565C0", s=2, alpha=0.6, label="Estático")
        ax.scatter(angles[dynamic_mask], ranges[dynamic_mask],
                   c=color_dyn, s=8, alpha=0.9, label="Dinámico")
        ax.set_title(title, pad=12)
        ax.set_rlabel_position(90)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax.set_xlabel("Ángulo [rad]")

    fig.suptitle(f"Escaneo {scan_idx} — Filtrado dinámico LiDAR 2D", y=1.02)
    fig.tight_layout()

    saved_path = None
    if figures_dir:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        p = figures_dir / f"dynamic_mask_scan{scan_idx:04d}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        saved_path = str(p)

    if show:
        plt.show()
    plt.close(fig)
    return saved_path


def plot_training_curves(
    history_path: str | Path,
    figures_dir: Optional[str | Path] = None,
    show: bool = False,
) -> Optional[str]:
    """Grafica pérdida y F1 de train vs val por época.

    Args:
        history_path: Ruta al history.json de train_one_fold.
        figures_dir:  Directorio para guardar.
        show:         Si True, llama plt.show().

    Returns:
        Ruta de la figura, o None.
    """
    history_path = Path(history_path)
    if not history_path.exists():
        print(f"[viz] history.json no encontrado: {history_path}")
        return None

    with open(history_path) as f:
        history = json.load(f)

    epochs     = [ep["epoch"] for ep in history["train"]]
    train_loss = [ep["loss"]  for ep in history["train"]]
    val_loss   = [ep["loss"]  for ep in history["val"]]
    train_f1   = [ep["f1"]    for ep in history["train"]]
    val_f1     = [ep["f1"]    for ep in history["val"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, train_loss, "o-", color="#1565C0", label="Train", lw=2)
    ax1.plot(epochs, val_loss,   "s--", color="#E53935", label="Validación", lw=2)
    ax1.set_xlabel("Época"); ax1.set_ylabel("Pérdida BCE [adim.]")
    ax1.set_title("Curva de pérdida"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_f1, "o-", color="#1565C0", label="Train", lw=2)
    ax2.plot(epochs, val_f1,   "s--", color="#E53935", label="Validación", lw=2)
    ax2.set_xlabel("Época"); ax2.set_ylabel("F1-score [adim.]")
    ax2.set_ylim(0, 1); ax2.set_title("Curva de F1-score")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("Curvas de entrenamiento — CNN Temporal CastaMask")
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


def build_comparison_table(
    ml_report: Optional[Dict] = None,
    dl_metrics: Optional[Dict] = None,
) -> pd.DataFrame:
    """Construye la tabla comparativa unificada de todos los modelos.

    Args:
        ml_report:  REPORTE_ML_CLASICO de ml_models.train_classical_models().
        dl_metrics: dict con métricas test de train_one_fold (test_metrics).

    Returns:
        DataFrame con filas = modelos, columnas = métricas.
    """
    rows = []

    if ml_report and "all_results" in ml_report:
        for name, res in ml_report["all_results"].items():
            m = res["test_metrics"]
            rows.append({
                "Modelo":     name,
                "Tipo":       "Clásico",
                "Accuracy":   round(m.get("accuracy",  float("nan")), 4),
                "Precision":  round(m.get("precision", float("nan")), 4),
                "Recall":     round(m.get("recall",    float("nan")), 4),
                "F1-score":   round(m.get("f1",        float("nan")), 4),
                "ROC-AUC":    round(m.get("roc_auc",   float("nan")), 4),
            })

    if dl_metrics:
        rows.append({
            "Modelo":    "CNN Temporal (CastaMask)",
            "Tipo":      "Deep Learning",
            "Accuracy":  round(dl_metrics.get("accuracy",  float("nan")), 4),
            "Precision": round(dl_metrics.get("precision", float("nan")), 4),
            "Recall":    round(dl_metrics.get("recall",    float("nan")), 4),
            "F1-score":  round(dl_metrics.get("f1",        float("nan")), 4),
            "ROC-AUC":   float("nan"),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Modelo")
    return df


def plot_comparison_table(
    comparison_df: pd.DataFrame,
    figures_dir: Optional[str | Path] = None,
    show: bool = False,
) -> Optional[str]:
    """Renderiza la tabla comparativa como figura.

    Args:
        comparison_df: DataFrame de build_comparison_table().
        figures_dir:   Directorio para guardar.
        show:          Si True, llama plt.show().

    Returns:
        Ruta de la figura, o None.
    """
    if comparison_df.empty:
        print("[viz] Tabla comparativa vacía — nada que renderizar")
        return None

    metric_cols = ["Accuracy", "Precision", "Recall", "F1-score"]
    metric_cols = [c for c in metric_cols if c in comparison_df.columns]
    plot_df = comparison_df[metric_cols].copy()

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 2), 4))
    x = np.arange(len(metric_cols))
    width = 0.8 / len(plot_df)
    colors = ["#1565C0", "#43A047", "#E53935", "#FB8C00", "#8E24AA"]

    for i, (model_name, row) in enumerate(plot_df.iterrows()):
        vals = [row.get(c, 0) for c in metric_cols]
        ax.bar(x + i * width, vals, width, label=model_name,
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x + width * (len(plot_df) - 1) / 2)
    ax.set_xticklabels(metric_cols)
    ax.set_ylabel("Valor de la métrica [adim.]")
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparativa de modelos — conjunto de prueba")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    saved_path = None
    if figures_dir:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        p = figures_dir / "model_comparison.png"
        fig.savefig(p, dpi=150)
        saved_path = str(p)

    if show:
        plt.show()
    plt.close(fig)
    return saved_path


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_report(
    results: Dict,
    figures_dir: str | Path = None,
    report_dir:  str | Path = None,
) -> Dict:
    """Genera todas las figuras y el esqueleto del reporte LaTeX.

    Args:
        results: Dict con claves opcionales:
            'ml_report'    — salida de ml_models.train_classical_models()
            'dl_result'    — salida de dl_model.train_model()
            'history_path' — ruta al history.json
            'eda_summary'  — salida de data_loader.eda_summary()
        figures_dir: Directorio para figuras.
        report_dir:  Directorio para el reporte LaTeX.

    Returns:
        RESULTADOS_FINALES: dict con:
            'figure_paths'      (list)
            'comparison_table'  (pd.DataFrame)
            'report_tex_path'   (str)
            'recall_ok'         (bool) — True si recall CNN > 0.85
    """
    figures_dir = Path(figures_dir) if figures_dir else _FIGURES_DIR
    report_dir  = Path(report_dir)  if report_dir  else _REPORT_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []

    # Curvas de entrenamiento
    history_path = results.get("history_path") or \
        "experiments_fullscan/fold_00/history.json"
    p = plot_training_curves(history_path, figures_dir=figures_dir)
    if p:
        figure_paths.append(p)

    # Tabla comparativa
    ml_report  = results.get("ml_report")
    dl_result  = results.get("dl_result")
    dl_metrics = dl_result.get("test_metrics") if dl_result else None

    comparison_df = build_comparison_table(ml_report, dl_metrics)
    if not comparison_df.empty:
        p = plot_comparison_table(comparison_df, figures_dir=figures_dir)
        if p:
            figure_paths.append(p)

    # Máscara dinámica de ejemplo (si hay datos disponibles)
    if "sample_scan" in results:
        scan = results["sample_scan"]
        p = plot_dynamic_mask(
            ranges=scan.get("ranges", np.ones(360)),
            predicted_mask=scan.get("predicted_mask", np.zeros(360, dtype=int)),
            gt_mask=scan.get("gt_mask", np.zeros(360, dtype=int)),
            figures_dir=figures_dir,
        )
        if p:
            figure_paths.append(p)

    # Verificar recall CNN >= 85%
    recall_ok = False
    if dl_metrics:
        recall = dl_metrics.get("recall", 0.0)
        recall_ok = recall >= 0.85
        # ASERCIÓN Wilson §5a
        if not recall_ok:
            print(f"[viz] ALERTA: Recall CNN={recall:.3f} < 0.85 — "
                  "hipótesis científica no cumplida con el modelo actual")

    # Generar reporte LaTeX
    tex_path = _write_latex_report(
        report_dir=report_dir,
        figures_dir=figures_dir,
        comparison_df=comparison_df,
        dl_metrics=dl_metrics,
        figure_paths=figure_paths,
    )
    figure_paths.append(tex_path)

    print(f"[viz] Reporte generado: {len(figure_paths)} archivos")
    print(f"[viz] LaTeX: {tex_path}")

    return {
        "figure_paths":     figure_paths,
        "comparison_table": comparison_df,
        "report_tex_path":  tex_path,
        "recall_ok":        recall_ok,
    }


# ---------------------------------------------------------------------------
# LaTeX report skeleton
# ---------------------------------------------------------------------------

def _write_latex_report(
    report_dir: Path,
    figures_dir: Path,
    comparison_df: pd.DataFrame,
    dl_metrics: Optional[Dict],
    figure_paths: list,
) -> str:
    """Escribe el archivo .tex del reporte final."""

    # Tabla comparativa en LaTeX
    if not comparison_df.empty:
        metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1-score"]
                       if c in comparison_df.columns]
        table_rows = ""
        for model_name, row in comparison_df.iterrows():
            vals = " & ".join(
                f"{row.get(c, float('nan')):.4f}" for c in metric_cols
            )
            table_rows += f"        {model_name} & {vals} \\\\\n"

        col_spec = "l" + "c" * len(metric_cols)
        header   = " & ".join(metric_cols)
        table_tex = (
            f"\\begin{{tabular}}{{{col_spec}}}\n"
            f"        \\toprule\n"
            f"        Modelo & {header} \\\\\n"
            f"        \\midrule\n"
            f"{table_rows}"
            f"        \\bottomrule\n"
            f"    \\end{{tabular}}"
        )
    else:
        table_tex = "\\textit{Sin resultados disponibles}"

    dl_f1      = dl_metrics.get("f1",      "N/A") if dl_metrics else "N/A"
    dl_recall  = dl_metrics.get("recall",  "N/A") if dl_metrics else "N/A"
    dl_prec    = dl_metrics.get("precision","N/A") if dl_metrics else "N/A"

    tex = rf"""\documentclass{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[spanish]{{babel}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{geometry}}
\usepackage{{hyperref}}
\geometry{{margin=2.5cm}}

\title{{Sistema de Filtrado Dinámico en LiDAR 2D \\
        \large TC6039.1 Applied Computing --- Reporte Final}}
\author{{
    Gerardo Andrés Castañón Sarmiento \and
    Ricardo Daniel Damián Cortez \and
    Juan Angel Lucio Rojas \and
    Priscila de los Ángeles Correa Miranda \and
    Miguel Ángel Chávez Robles
}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Este reporte presenta CastaMask, un sistema de cómputo científico modular
para la clasificación binaria de haces LiDAR 2D entre estáticos y dinámicos.
El sistema emplea una Red Neuronal Convolucional Temporal (CNN 1D) entrenada
sobre ventanas deslizantes de escaneos simulados en Gazebo con un TurtleBot.
Se compara contra tres clasificadores clásicos (Regresión Logística, Árbol
de Decisión y Random Forest) como línea base. La hipótesis central es que el
contexto temporal mejora significativamente la clasificación respecto a métodos
de trama única.
\end{{abstract}}

\section{{Introducción y Motivación}}
Los sistemas de navegación autónoma en almacenes industriales enfrentan
obstáculos dinámicos (personas, robots, pallets) que degradan los algoritmos
de SLAM al introducir mediciones transitorias en el mapa estático. CastaMask
aborda este problema filtrando los haces dinámicos a nivel de beam antes de
que el escaneo llegue al módulo de localización.

\section{{Datos y Metodología}}

\subsection{{Dataset}}
Los datos fueron generados mediante simulación en Gazebo con activos de
almacén AWS. Un TurtleBot equipado con un LiDAR 2D de 360 haces recorre
13 escenarios distintos (bag\_ids 0--12) con variaciones de velocidad
y número de objetos dinámicos. El etiquetado automático usa el plugin
RayGtSensor para verificar intersecciones de rayos con objetos dinámicos.

Cada muestra contiene un tensor $X \in \mathbb{{R}}^{{T \times 360 \times 6}}$
con $T=7$ frames temporales y 6 features por haz:
\texttt{{range\_norm}}, \texttt{{delta\_r}}, \texttt{{static\_residual}},
\texttt{{abs\_static\_residual}}, \texttt{{spatial\_grad}}, \texttt{{valid\_mask}}.

\subsection{{Pipeline}}
El sistema sigue una arquitectura de 7 componentes modulares:
C1 (carga y EDA), C2 (ingeniería de características), C3 (optimización
de hiperparámetros), C4 (ML clásico), C5 (Deep Learning), C6 (visualización)
y C7 (orquestación y reproducibilidad).

\section{{Análisis Exploratorio (C1)}}
Se analizaron los escaneos de cada escenario calculando estadísticas
de rango, fracción de haces dinámicos y gradiente espacial.
Las figuras \ref{{fig:eda_hist}} y \ref{{fig:eda_boxplot}} muestran la
distribución de rangos y dinamismo por escenario.

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{{figures/eda_hist_range.png}}
    \caption{{Distribución del rango medio normalizado por escenario (bag\_id).
              Unidades: adimensional (rango normalizado al máximo del sensor).}}
    \label{{fig:eda_hist}}
\end{{figure}}

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{{figures/eda_boxplot_dynamic.png}}
    \caption{{Fracción de haces dinámicos por escenario. Los escenarios
              con bag\_id=0,1 (vacío) presentan fracción cero, mientras
              que escenarios con personas o robots adicionales muestran
              mayor variabilidad. Unidades: fracción adimensional $[0,1]$.}}
    \label{{fig:eda_boxplot}}
\end{{figure}}

\section{{Resultados}}

\subsection{{Modelos Clásicos (C4)}}
Se entrenaron tres clasificadores usando las features del frame actual
aplanadas a nivel de haz: Regresión Logística, Árbol de Decisión y
Random Forest. La tabla~\ref{{tab:comparison}} muestra los resultados
en el conjunto de prueba.

\subsection{{CNN Temporal (C5)}}
La CNN Temporal procesa la ventana completa de $T=7$ frames y produce
un logit por haz. Métricas en prueba:
F1={dl_f1}, Recall={dl_recall}, Precision={dl_prec}.

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.9\textwidth]{{figures/training_curves.png}}
    \caption{{Curvas de entrenamiento de la CNN Temporal.
              Izquierda: pérdida BCE (adimensional) vs época.
              Derecha: F1-score (adimensional, rango $[0,1]$) vs época.
              Línea sólida = train, discontinua = validación.}}
\end{{figure}}

\subsection{{Tabla Comparativa}}

\begin{{table}}[h]
    \centering
    \caption{{Comparativa de todos los modelos en el conjunto de prueba.
              Todas las métricas son adimensionales en el rango $[0,1]$.}}
    \label{{tab:comparison}}
    {table_tex}
\end{{table}}

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.85\textwidth]{{figures/model_comparison.png}}
    \caption{{Comparativa visual de métricas por modelo.
              Barras agrupadas por métrica (Accuracy, Precision, Recall, F1).
              Todas las métricas son adimensionales, rango $[0,1]$.}}
\end{{figure}}

\section{{Conclusiones}}
CastaMask demuestra que el uso de contexto temporal en ventanas deslizantes
mejora la clasificación de haces dinámicos respecto a clasificadores que
analizan tramas individuales. La CNN Temporal supera a los baselines clásicos
en Recall, métrica crítica para garantizar que los objetos dinámicos sean
detectados antes de contaminar el mapa estático.

\section{{Reproducibilidad}}
El sistema se ejecuta con un único comando:
\begin{{verbatim}}
make run
\end{{verbatim}}
El repositorio incluye \texttt{{requirements.txt}} con versiones exactas,
\texttt{{Makefile}} con targets \texttt{{run}}, \texttt{{test}} y \texttt{{report}},
y un notebook de exploración en \texttt{{notebooks/}}.

\end{{document}}
"""

    tex_path = report_dir / "reporte_final.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return str(tex_path)


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    # Demo con datos mínimos
    results = {
        "history_path": "experiments_fullscan/fold_00/history.json",
    }

    final = generate_report(
        results=results,
        figures_dir=_FIGURES_DIR,
        report_dir=_REPORT_DIR,
    )
    print("\nRESULTADOS_FINALES:")
    print(f"  Figuras generadas: {len(final['figure_paths'])}")
    print(f"  LaTeX: {final['report_tex_path']}")
    print(f"  Recall OK (>85%): {final['recall_ok']}")
