# CastaMask — Makefile
# TC6039.1 Applied Computing — Tec de Monterrey
# Uso: make <target>

PYTHON = python3

.PHONY: all run run-full dry fast fast-n fast-opt fast-opt-n runfold0 \
        train cv test report eda features ml clean clean-all help

all: run

# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

## run: dataset completo, sin optimizacion de hiperparametros (~1-2 hrs)
run:
	@echo "==> Pipeline completo (sin optimizacion)..."
	$(PYTHON) main.py --skip-opt

## run-full: dataset completo + busqueda de hiperparametros (muy lento)
run-full:
	@echo "==> Pipeline completo con optimizacion..."
	$(PYTHON) main.py

## runfold0: dataset completo, fold 0, sin optimizacion
runfold0:
	@echo "==> Pipeline fold 0 (sin optimizacion)..."
	$(PYTHON) main.py --skip-opt --fold 0

# ---------------------------------------------------------------------------
# Modos rapidos — datos reales pero limitados
# ---------------------------------------------------------------------------
# Todos usan 1 shard por bag_id por defecto (13 shards = ~6.5k scans)
# C1/C2/C4 cargan 1 shard por bag_id (todos los escenarios representados)
# C5 usa max_shards shards por split del fold CV

## dry: verifica que el codigo corra de principio a fin (~1-2 min)
dry:
	@echo "==> Dry run (2 shards, 1 epoca, solo LogReg)..."
	$(PYTHON) main.py --dry-run --skip-opt

## fast: 1 shard/bag_id, sin optimizacion (~5-10 min en M4)
fast:
	@echo "==> Fast run (1 shard/bag_id, 13 escenarios)..."
	$(PYTHON) main.py --fast --skip-opt

## fast-n: N shards/bag_id, sin optimizacion. Uso: make fast-n N=3
fast-n:
	@echo "==> Fast run ($(N) shards/bag_id)..."
	$(PYTHON) main.py --fast --max-shards $(N) --skip-opt

## fast-opt: 1 shard/bag_id + busqueda de hiperparametros
fast-opt:
	@echo "==> Fast run con optimizacion..."
	$(PYTHON) main.py --fast

## fast-opt-n: N shards/bag_id + busqueda de hiperparametros. Uso: make fast-opt-n N=3
fast-opt-n:
	@echo "==> Fast run con optimizacion ($(N) shards/bag_id)..."
	$(PYTHON) main.py --fast --max-shards $(N)

# ---------------------------------------------------------------------------
# Componentes individuales
# ---------------------------------------------------------------------------

## train: entrena solo la CNN Temporal (fold 0, dataset completo)
train:
	@echo "==> Entrenando CNN Temporal fold 0..."
	cd src && $(PYTHON) train_one_fold.py

## cv: validacion cruzada leave-one-family-out completa
cv:
	@echo "==> Validacion cruzada..."
	cd src && $(PYTHON) run_cv.py

## eda: solo analisis exploratorio C1
eda:
	cd src && $(PYTHON) data_loader.py

## features: solo extraccion de caracteristicas C2
features:
	cd src && $(PYTHON) feature_engineering.py

## ml: solo modelos clasicos C4
ml:
	cd src && $(PYTHON) ml_models.py

# ---------------------------------------------------------------------------
# Calidad y reporte
# ---------------------------------------------------------------------------

## test: pruebas unitarias con pytest
test:
	@echo "==> Pruebas unitarias..."
	$(PYTHON) -m pytest tests/ -v --tb=short

## report: genera figuras y compila reporte LaTeX
report:
	@echo "==> Generando reporte..."
	cd src && $(PYTHON) viz.py
	@if command -v pdflatex > /dev/null 2>&1; then \
		echo "==> Compilando LaTeX..."; \
		cd report && pdflatex reporte_final.tex && pdflatex reporte_final.tex; \
	else \
		echo "==> pdflatex no encontrado — sube report/reporte_final.tex a Overleaf"; \
	fi

# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------

## clean: elimina caches y archivos temporales
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f report/*.aux report/*.log report/*.out 2>/dev/null || true

## clean-all: elimina TODO incluyendo experimentos y figuras (cuidado)
clean-all: clean
	rm -rf experiments_fullscan/
	rm -rf report/figures/
	rm -f report/reporte_final.pdf

## help: muestra esta ayuda
help:
	@grep -E '^## ' Makefile | sed 's/## /  make /'
