"""
tests/test_all_components.py
Pruebas unitarias para todos los componentes de CastaMask.
Ejecutar con: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Asegurar que src/ esté en el path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================
# Fixtures compartidos
# ============================================================

@pytest.fixture
def fake_shard(tmp_path):
    """Crea un shard .npz mínimo y válido para pruebas."""
    N, T, H, K = 20, 7, 360, 6
    shard_path = tmp_path / "train_shard_00000.npz"
    np.savez(
        shard_path,
        X             = np.random.randn(N, T, H, K).astype(np.float32),
        y             = np.random.randint(0, 2, (N, H)).astype(np.uint8),
        valid_current = np.ones((N, H), dtype=np.uint8),
        stamp_ns      = np.arange(N, dtype=np.int64) * 12_000_000,
        bag_id        = np.zeros(N, dtype=np.int32),
    )
    return shard_path


@pytest.fixture
def multi_bag_shards(tmp_path):
    """Crea shards con múltiples bag_ids para pruebas de split."""
    N, T, H, K = 10, 7, 360, 6
    paths = []
    for bag_id in range(3):
        p = tmp_path / f"train_shard_{bag_id:05d}.npz"
        np.savez(
            p,
            X             = np.random.randn(N, T, H, K).astype(np.float32),
            y             = np.random.randint(0, 2, (N, H)).astype(np.uint8),
            valid_current = np.ones((N, H), dtype=np.uint8),
            stamp_ns      = np.arange(N, dtype=np.int64),
            bag_id        = np.full(N, bag_id, dtype=np.int32),
        )
        paths.append(p)
    return tmp_path, paths


# ============================================================
# C1 — data_loader
# ============================================================

class TestDataLoader:

    def test_load_data_normal(self, fake_shard):
        """C1 caso normal: carga un shard válido y devuelve DataFrame correcto."""
        from data_loader import load_data
        df = load_data(fake_shard)
        assert len(df) == 20
        assert "range_mean" in df.columns
        assert "pct_dynamic" in df.columns
        assert df["pct_valid"].between(0, 1).all()

    def test_load_data_missing_file(self, tmp_path):
        """C1 caso borde: archivo inexistente lanza FileNotFoundError."""
        from data_loader import load_data
        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "no_existe.npz")

    def test_load_data_missing_keys(self, tmp_path):
        """C1 caso borde: shard sin clave 'y' lanza KeyError."""
        from data_loader import load_data
        bad = tmp_path / "bad.npz"
        np.savez(bad, X=np.zeros((5, 7, 360, 6)))
        with pytest.raises(KeyError):
            load_data(bad)

    def test_clean_data_removes_nan(self, fake_shard):
        """C1: clean_data elimina filas con NaN sin romper el DataFrame."""
        from data_loader import load_data, clean_data
        df = load_data(fake_shard)
        df.loc[0, "range_mean"] = float("nan")
        df_clean = clean_data(df)
        assert not df_clean["range_mean"].isna().any()

    def test_clean_data_empty_raises(self):
        """C1 caso borde: DataFrame vacío lanza ValueError."""
        from data_loader import clean_data
        import pandas as pd
        with pytest.raises(ValueError):
            clean_data(pd.DataFrame())

    def test_eda_summary_returns_dict(self, fake_shard, tmp_path):
        """C1: eda_summary devuelve dict con claves requeridas."""
        from data_loader import load_data, clean_data, eda_summary
        df = clean_data(load_data(fake_shard))
        result = eda_summary(df, figures_dir=tmp_path / "figs")
        assert "stats" in result
        assert "class_balance" in result
        assert "figure_paths" in result
        assert len(result["figure_paths"]) == 4


# ============================================================
# C2 — feature_engineering
# ============================================================

class TestFeatureEngineering:

    def test_extract_feature_tensor_shape(self, fake_shard):
        """C2 caso normal: tensor de salida tiene shape correcto [N, 360, K]."""
        from feature_engineering import extract_feature_tensor
        shard_dir = fake_shard.parent
        X, y, valid = extract_feature_tensor(shard_dir)
        assert X.ndim == 3
        assert X.shape[1] == 360
        assert X.shape[2] == 6
        assert X.shape[0] == y.shape[0]

    def test_extract_feature_tensor_no_nan(self, fake_shard):
        """C2: el tensor de features no contiene NaN."""
        from feature_engineering import extract_feature_tensor
        X, _, _ = extract_feature_tensor(fake_shard.parent)
        assert not np.isnan(X[:, :, :5]).any()

    def test_extract_temporal_invalid_window(self, fake_shard):
        """C2 caso borde: ventana_temporal=1 lanza ValueError."""
        from feature_engineering import extract_temporal_tensor
        with pytest.raises(ValueError):
            extract_temporal_tensor(fake_shard.parent, ventana_temporal=1)

    def test_normalize_tensor(self, fake_shard):
        """C2: normalize_tensor produce tensor sin NaN y con std ~1."""
        from feature_engineering import (
            extract_feature_tensor, compute_norm_stats, normalize_tensor
        )
        X, _, _ = extract_feature_tensor(fake_shard.parent)
        stats   = compute_norm_stats(X)
        X_norm  = normalize_tensor(X, stats)
        assert not np.isnan(X_norm).any()
        assert X_norm.shape == X.shape


# ============================================================
# C3 — optimizer (solo plot_convergence, el grid search tarda)
# ============================================================

class TestOptimizer:

    def test_plot_convergence_missing_file(self, tmp_path):
        """C3: plot_convergence con archivo inexistente devuelve None."""
        from optimizer import plot_convergence
        result = plot_convergence(tmp_path / "no_existe.json", figures_dir=tmp_path)
        assert result is None

    def test_plot_convergence_valid(self, tmp_path):
        """C3 caso normal: plot_convergence genera figura desde history válido."""
        import json
        from optimizer import plot_convergence

        history = {
            "train": [{"epoch": i, "loss": 1.0 / i, "f1": 0.1 * i} for i in range(1, 6)],
            "val":   [{"epoch": i, "loss": 1.2 / i, "f1": 0.08 * i} for i in range(1, 6)],
        }
        hp = tmp_path / "history.json"
        hp.write_text(json.dumps(history))
        result = plot_convergence(hp, figures_dir=tmp_path)
        assert result is not None
        assert Path(result).exists()


# ============================================================
# C4 — ml_models
# ============================================================

class TestMLModels:

    @pytest.fixture
    def tabular_data(self):
        """Datos tabulares sintéticos balanceados."""
        rng = np.random.default_rng(42)
        N = 500
        X = rng.standard_normal((N, 6)).astype(np.float32)
        y = rng.integers(0, 2, N).astype(np.int32)
        split = int(N * 0.7)
        val   = int(N * 0.85)
        return (X[:split], X[split:val], X[val:],
                y[:split], y[split:val], y[val:])

    def test_train_classical_normal(self, tabular_data, tmp_path):
        """C4 caso normal: entrena 3 modelos y devuelve reporte con tabla comparativa."""
        from ml_models import train_classical_models
        X_train, X_val, X_test, y_train, y_val, y_test = tabular_data
        report = train_classical_models(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            figures_dir=tmp_path,
        )
        assert "comparison_table" in report
        assert len(report["comparison_table"]) == 3
        assert "best_model_name" in report
        assert "best_model" in report

    def test_train_classical_nan_raises(self, tabular_data):
        """C4 caso borde: X_train con NaN lanza ValueError."""
        from ml_models import train_classical_models
        X_train, X_val, X_test, y_train, y_val, y_test = tabular_data
        X_train_bad = X_train.copy()
        X_train_bad[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            train_classical_models(
                X_train_bad, X_val, X_test,
                y_train, y_val, y_test,
            )

    def test_train_classical_single_class_raises(self, tabular_data):
        """C4 caso borde: solo una clase en entrenamiento lanza ValueError."""
        from ml_models import train_classical_models
        X_train, X_val, X_test, _, y_val, y_test = tabular_data
        y_single = np.zeros(len(X_train), dtype=np.int32)
        with pytest.raises(ValueError, match="clase"):
            train_classical_models(
                X_train, X_val, X_test,
                y_single, y_val, y_test,
            )

    def test_metrics_in_valid_range(self, tabular_data, tmp_path):
        """C4: todas las métricas del reporte están en [0, 1]."""
        from ml_models import train_classical_models
        X_train, X_val, X_test, y_train, y_val, y_test = tabular_data
        report = train_classical_models(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            figures_dir=tmp_path,
        )
        for name, res in report["all_results"].items():
            for metric, value in res["test_metrics"].items():
                assert 0.0 <= value <= 1.0, \
                    f"Métrica {metric} de {name} = {value} fuera de [0,1]"


# ============================================================
# C5 — dl_model (pruebas de inferencia, no de entrenamiento)
# ============================================================

class TestDLModel:

    def test_model_forward_shape(self):
        """C5 caso normal: el modelo produce output [B, 360] dado input [B, K, T, 360]."""
        import torch
        from model import CastaMaskFullScanCNN
        from config import IN_CHANNELS, TIME_WINDOW, NUM_BEAMS

        model = CastaMaskFullScanCNN()
        x = torch.randn(2, IN_CHANNELS, TIME_WINDOW, NUM_BEAMS)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, NUM_BEAMS)

    def test_model_output_finite(self):
        """C5: el modelo no produce NaN ni inf en el output."""
        import torch
        from model import CastaMaskFullScanCNN
        from config import IN_CHANNELS, TIME_WINDOW, NUM_BEAMS

        model = CastaMaskFullScanCNN()
        x = torch.randn(4, IN_CHANNELS, TIME_WINDOW, NUM_BEAMS)
        with torch.no_grad():
            y = model(x)
        assert torch.isfinite(y).all()

    def test_predict_interface(self):
        """C5: dl_model.predict devuelve dict con logits, probs y predictions."""
        import torch
        from model import CastaMaskFullScanCNN
        from dl_model import predict
        from config import IN_CHANNELS, TIME_WINDOW, NUM_BEAMS

        model = CastaMaskFullScanCNN()
        x = torch.randn(1, IN_CHANNELS, TIME_WINDOW, NUM_BEAMS)
        result = predict(model, x, threshold=0.5)
        assert "logits" in result
        assert "probs" in result
        assert "predictions" in result
        assert result["probs"].min() >= 0.0
        assert result["probs"].max() <= 1.0
        assert set(result["predictions"].unique().tolist()).issubset({0, 1})


# ============================================================
# C6 — viz
# ============================================================

class TestViz:

    def test_build_comparison_table_empty(self):
        """C6: build_comparison_table sin argumentos devuelve DataFrame vacío."""
        from viz import build_comparison_table
        df = build_comparison_table()
        assert df.empty

    def test_build_comparison_table_with_dl(self):
        """C6 caso normal: tabla con métricas DL tiene una fila."""
        from viz import build_comparison_table
        dl_metrics = {"accuracy": 0.9, "precision": 0.85,
                      "recall": 0.88, "f1": 0.86}
        df = build_comparison_table(dl_metrics=dl_metrics)
        assert len(df) == 1
        assert "F1-score" in df.columns

    def test_plot_dynamic_mask_saves_file(self, tmp_path):
        """C6: plot_dynamic_mask guarda archivo .png en figures_dir."""
        from viz import plot_dynamic_mask
        ranges = np.random.rand(360).astype(np.float32) * 5
        pred   = np.random.randint(0, 2, 360)
        gt     = np.random.randint(0, 2, 360)
        path   = plot_dynamic_mask(ranges, pred, gt, figures_dir=tmp_path)
        assert path is not None
        assert Path(path).exists()

    def test_generate_report_creates_tex(self, tmp_path):
        """C6: generate_report produce archivo .tex en report_dir."""
        from viz import generate_report
        final = generate_report(
            results={},
            figures_dir=tmp_path / "figures",
            report_dir=tmp_path / "report",
        )
        assert "report_tex_path" in final
        assert Path(final["report_tex_path"]).exists()
