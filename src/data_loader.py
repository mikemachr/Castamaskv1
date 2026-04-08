"""
data_loader.py — Componente C1: Carga de Datos, Limpieza y EDA
==============================================================
Responsable: Gerardo Andrés Castañón Sarmiento

Carga shards .npz del dataset LiDAR CastaMask, valida su integridad,
aplica limpieza básica y genera un reporte exploratorio (EDA).

Interfaces de salida:
    load_data(path)      -> pd.DataFrame  (resumen por escaneo)
    clean_data(df)       -> pd.DataFrame  (filtrado y validado)
    eda_summary(df)      -> dict           (estadísticas + figuras)

Referencias: Wilson et al. (2014) §5a (aserciones), §7a (docstrings).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Root of the project (two levels up from src/)
_ROOT = Path(__file__).resolve().parent.parent
_FIGURES_DIR = _ROOT / "report" / "figures"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_BEAMS: int = 360
RANGE_MIN: float = 0.05   # metros — mínimo físico del sensor
RANGE_MAX: float = 10.0   # metros — máximo configurado en el sensor
FEATURE_NAMES: List[str] = [
    "range_norm",
    "delta_r",
    "static_residual",
    "abs_static_residual",
    "spatial_grad",
    "valid_mask",
]


# ---------------------------------------------------------------------------
# C1.1 — Carga
# ---------------------------------------------------------------------------

def load_data(path: str | Path) -> pd.DataFrame:
    """Carga un shard .npz y devuelve un DataFrame resumen por escaneo.

    Cada fila del DataFrame corresponde a un escaneo LiDAR completo.
    Las columnas incluyen estadísticas agregadas de rangos, porcentaje
    de haces dinámicos y metadatos temporales.

    Args:
        path: Ruta al archivo .npz. Debe contener las claves
              'X', 'y', 'valid_current', 'bag_id' y 'stamp_ns'.

    Returns:
        DataFrame con columnas:
            scan_idx       (int)   — índice del escaneo en el shard
            bag_id         (int)   — identificador del escenario
            stamp_ns       (int)   — timestamp en nanosegundos
            range_mean     (float) — rango medio de haces válidos [m]
            range_std      (float) — desviación estándar de rangos [m]
            range_min      (float) — rango mínimo válido [m]
            range_max      (float) — rango máximo válido [m]
            pct_dynamic    (float) — fracción de haces dinámicos [0, 1]
            pct_valid      (float) — fracción de haces con medición válida
            delta_r_mean   (float) — cambio de rango medio (feature 1)
            spatial_grad_mean (float) — gradiente espacial medio (feature 4)

    Raises:
        FileNotFoundError: Si el archivo no existe.
        KeyError: Si el archivo .npz no contiene las claves requeridas.
        ValueError: Si las dimensiones de los arreglos son inconsistentes.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Shard no encontrado: {path}")

    with np.load(path, allow_pickle=True) as z:
        required_keys = {"X", "y", "valid_current", "bag_id", "stamp_ns"}
        missing = required_keys - set(z.keys())
        if missing:
            raise KeyError(f"Claves faltantes en {path.name}: {missing}")

        X             = z["X"].astype(np.float32)        # [N, T, 360, K]
        y             = z["y"].astype(np.uint8)           # [N, 360]
        valid_current = z["valid_current"].astype(np.uint8)  # [N, 360]
        bag_id_arr    = np.asarray(z["bag_id"], dtype=np.int32)   # [N]
        stamp_ns      = np.asarray(z["stamp_ns"], dtype=np.int64) # [N]

    N = y.shape[0]

    # ASERCIÓN Wilson §5a: dimensiones consistentes
    assert X.shape[0] == N, \
        f"Inconsistencia: X tiene {X.shape[0]} muestras pero y tiene {N}"
    assert X.shape[2] == NUM_BEAMS, \
        f"Se esperaban {NUM_BEAMS} haces, se encontraron {X.shape[2]}"
    assert y.shape[1] == NUM_BEAMS, \
        f"y debe tener {NUM_BEAMS} columnas, tiene {y.shape[1]}"

    # Extraer feature de rango normalizado (índice 0) del frame actual (T-1)
    range_norm   = X[:, -1, :, 0]   # [N, 360]
    delta_r      = X[:, -1, :, 1]   # [N, 360]
    spatial_grad = X[:, -1, :, 4]   # [N, 360]

    valid_bool = valid_current.astype(bool)  # [N, 360]

    rows = []
    for i in range(N):
        v = valid_bool[i]
        r = range_norm[i][v] if v.any() else np.array([np.nan])
        rows.append({
            "scan_idx":          i,
            "bag_id":            int(bag_id_arr[i] if bag_id_arr.ndim > 0 else bag_id_arr),
            "stamp_ns":          int(stamp_ns[i]),
            "range_mean":        float(np.nanmean(r)),
            "range_std":         float(np.nanstd(r)),
            "range_min":         float(np.nanmin(r)),
            "range_max":         float(np.nanmax(r)),
            "pct_dynamic":       float(y[i][v].mean()) if v.any() else 0.0,
            "pct_valid":         float(v.mean()),
            "delta_r_mean":      float(np.nanmean(delta_r[i][v])) if v.any() else 0.0,
            "spatial_grad_mean": float(np.nanmean(spatial_grad[i][v])) if v.any() else 0.0,
        })

    df = pd.DataFrame(rows)

    # ASERCIÓN Wilson §5a: DataFrame no vacío
    assert len(df) > 0, "El DataFrame resultante está vacío"

    return df


def load_multiple_shards(
    data_root: str | Path,
    pattern: str = "train_shard_*.npz",
    max_shards: Optional[int] = None,
) -> pd.DataFrame:
    """Carga múltiples shards y concatena sus DataFrames.

    Args:
        data_root: Directorio que contiene los shards.
        pattern:   Patrón glob para encontrar los archivos.
        max_shards: Si se especifica, limita el número de shards cargados.

    Returns:
        DataFrame concatenado con columna adicional 'shard_file'.

    Raises:
        FileNotFoundError: Si no se encuentran shards en data_root.
    """
    data_root = Path(data_root)
    shard_files = sorted(data_root.glob(pattern))

    if not shard_files:
        raise FileNotFoundError(f"No se encontraron shards en {data_root}")

    if max_shards is not None:
        # Pick one shard per bag_id to guarantee all scenarios are covered
        import numpy as _np
        seen_bags, selected = set(), []
        for _s in shard_files:
            try:
                _bid = int(_np.load(_s, allow_pickle=True)['bag_id'].flat[0])
            except Exception:
                _bid = -len(selected)
            if _bid not in seen_bags:
                seen_bags.add(_bid)
                selected.append(_s)
        shard_files = selected[:max_shards] if max_shards < len(selected) else selected

    dfs = []
    for shard_path in shard_files:
        df = load_data(shard_path)
        df["shard_file"] = shard_path.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# C1.2 — Limpieza
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el DataFrame eliminando escaneos degenerados.

    Estrategia de limpieza:
        1. Elimina filas con NaN en columnas numéricas clave.
        2. Elimina escaneos con pct_valid < 0.1 (menos del 10 % de haces válidos).
        3. Elimina outliers de range_mean usando el criterio IQR × 3.

    Args:
        df: DataFrame producido por load_data() o load_multiple_shards().

    Returns:
        DataFrame limpio con índice reiniciado.

    Raises:
        ValueError: Si el DataFrame de entrada está vacío.
    """
    if df.empty:
        raise ValueError("clean_data recibió un DataFrame vacío")

    n_original = len(df)
    numeric_cols = ["range_mean", "range_std", "pct_valid", "pct_dynamic"]

    # 1. Eliminar NaN
    df = df.dropna(subset=numeric_cols)

    # 2. Eliminar escaneos con muy pocos haces válidos
    df = df[df["pct_valid"] >= 0.10]

    # 3. Eliminar outliers de rango por IQR
    q1 = df["range_mean"].quantile(0.25)
    q3 = df["range_mean"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    df = df[(df["range_mean"] >= lower) & (df["range_mean"] <= upper)]

    df = df.reset_index(drop=True)

    # ASERCIÓN Wilson §5a: rango de pct_* en [0, 1]
    assert df["pct_dynamic"].between(0.0, 1.0).all(), \
        "pct_dynamic fuera del rango [0, 1] tras limpieza"
    assert df["pct_valid"].between(0.0, 1.0).all(), \
        "pct_valid fuera del rango [0, 1] tras limpieza"

    n_removed = n_original - len(df)
    print(f"[clean_data] {n_original} → {len(df)} escaneos "
          f"({n_removed} eliminados, {n_removed/n_original*100:.1f} %)")

    return df


# ---------------------------------------------------------------------------
# C1.3 — EDA
# ---------------------------------------------------------------------------

def eda_summary(
    df: pd.DataFrame,
    figures_dir: Optional[str | Path] = None,
    show: bool = False,
) -> Dict:
    """Genera estadísticas descriptivas y 4 figuras exploratorias.

    Figuras generadas:
        1. Histograma de range_mean por bag_id
        2. Boxplot de pct_dynamic por bag_id
        3. Matriz de correlación de features numéricas
        4. Scatter: range_mean vs pct_dynamic, coloreado por pct_valid

    Args:
        df:           DataFrame limpio producido por clean_data().
        figures_dir:  Directorio donde guardar las figuras (.png).
                      Si es None, las figuras no se guardan en disco.
        show:         Si True, llama plt.show() al final de cada figura.

    Returns:
        Diccionario con:
            'stats'         -> DataFrame de estadísticas descriptivas
            'class_balance' -> dict con conteo de escaneos dinámicos vs estáticos
            'figure_paths'  -> list de rutas a figuras guardadas (puede estar vacía)

    Raises:
        ValueError: Si el DataFrame está vacío.
    """
    if df.empty:
        raise ValueError("eda_summary recibió un DataFrame vacío")

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = [
        "range_mean", "range_std", "range_min", "range_max",
        "pct_dynamic", "pct_valid", "delta_r_mean", "spatial_grad_mean",
    ]
    stats = df[numeric_cols].describe()

    # Balance de clases (escaneos con al menos 1 haz dinámico)
    has_dynamic = df["pct_dynamic"] > 0
    class_balance = {
        "scans_with_dynamic": int(has_dynamic.sum()),
        "scans_fully_static": int((~has_dynamic).sum()),
        "overall_dynamic_rate": float(df["pct_dynamic"].mean()),
    }

    figure_paths = []

    # --- Figura 1: Histograma de range_mean por bag_id ---
    fig, ax = plt.subplots(figsize=(8, 4))
    for bid, grp in df.groupby("bag_id"):
        ax.hist(grp["range_mean"], bins=40, alpha=0.6, label=f"bag {bid}")
    ax.set_xlabel("Rango medio normalizado [adim.]")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de rango medio por escenario")
    ax.legend(fontsize=7)
    fig.tight_layout()
    if figures_dir:
        p = figures_dir / "eda_hist_range.png"
        fig.savefig(p, dpi=150)
        figure_paths.append(str(p))
    if show:
        plt.show()
    plt.close(fig)

    # --- Figura 2: Boxplot de pct_dynamic por bag_id ---
    fig, ax = plt.subplots(figsize=(8, 4))
    bag_ids = sorted(df["bag_id"].unique())
    data_to_plot = [df[df["bag_id"] == b]["pct_dynamic"].values for b in bag_ids]
    ax.boxplot(data_to_plot, labels=[f"bag {b}" for b in bag_ids], patch_artist=True)
    ax.set_xlabel("Escenario (bag_id)")
    ax.set_ylabel("Fracción de haces dinámicos [0, 1]")
    ax.set_title("Distribución de haces dinámicos por escenario")
    fig.tight_layout()
    if figures_dir:
        p = figures_dir / "eda_boxplot_dynamic.png"
        fig.savefig(p, dpi=150)
        figure_paths.append(str(p))
    if show:
        plt.show()
    plt.close(fig)

    # --- Figura 3: Matriz de correlación ---
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df[numeric_cols].corr()
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(numeric_cols, fontsize=8)
    plt.colorbar(im, ax=ax, label="Correlación de Pearson")
    ax.set_title("Matriz de correlación de features (nivel escaneo)")
    fig.tight_layout()
    if figures_dir:
        p = figures_dir / "eda_correlation_matrix.png"
        fig.savefig(p, dpi=150)
        figure_paths.append(str(p))
    if show:
        plt.show()
    plt.close(fig)

    # --- Figura 4: Scatter range_mean vs pct_dynamic ---
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        df["range_mean"], df["pct_dynamic"],
        c=df["pct_valid"], cmap="viridis", alpha=0.5, s=10,
    )
    plt.colorbar(sc, ax=ax, label="pct_valid [0, 1]")
    ax.set_xlabel("Rango medio normalizado [adim.]")
    ax.set_ylabel("Fracción de haces dinámicos [0, 1]")
    ax.set_title("Rango vs dinamismo (color = fracción de haces válidos)")
    fig.tight_layout()
    if figures_dir:
        p = figures_dir / "eda_scatter_range_dynamic.png"
        fig.savefig(p, dpi=150)
        figure_paths.append(str(p))
    if show:
        plt.show()
    plt.close(fig)

    print(f"[eda_summary] {len(df)} escaneos analizados | "
          f"{len(figure_paths)} figuras guardadas")
    print(f"[eda_summary] Tasa dinámica global: "
          f"{class_balance['overall_dynamic_rate']:.3f}")

    return {
        "stats":         stats,
        "class_balance": class_balance,
        "figure_paths":  figure_paths,
    }


# ---------------------------------------------------------------------------
# Ejecución directa — demo rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from config import DATA_ROOT

    shard_files = sorted(Path(DATA_ROOT).glob("train_shard_*.npz"))
    if not shard_files:
        print(f"No se encontraron shards en {DATA_ROOT}")
        sys.exit(1)

    print(f"Cargando {len(shard_files)} shards desde {DATA_ROOT}...")
    df_raw = load_multiple_shards(DATA_ROOT)
    print(f"  Escaneos cargados: {len(df_raw)}")

    df_clean = clean_data(df_raw)

    summary = eda_summary(df_clean, figures_dir=_FIGURES_DIR, show=False)
    print("\nEstadísticas descriptivas:")
    print(summary["stats"].to_string())
    print("\nBalance de clases:", summary["class_balance"])
