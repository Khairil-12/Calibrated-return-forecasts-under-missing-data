.PHONY: all setup data features experiment experiment-volatility figures report pdf clean clean-all test lint typecheck

# Full pipeline. PDF is a separate target: it needs pandoc and weasyprint,
# which are not Python dependencies, and `make pdf` fails if they are absent.
all: setup data features experiment figures report

# Environment setup
setup:
	pip install --upgrade pip
	pip install -r requirements.txt
	pre-commit install

# Data download and validation
data:
	python scripts/download_data.py

# Feature computation
features:
	python scripts/compute_features.py

# Run experimental grid
experiment:
	python scripts/run_grid.py

# Run the realized-volatility grid (separate target, separate outputs)
experiment-volatility:
	python scripts/run_grid.py --experiment-config experiment_volatility.yaml

# Figures for the volatility grid
figures-volatility:
	python scripts/generate_figures.py --metrics results/volatility/metrics.csv --output-dir results/volatility/figures

# Generate figures from results
figures:
	python scripts/generate_figures.py

# The written report is reports/report.md, produced by the experiment stage.
report:
	@test -f reports/report.md

# Render the PDF. Fails if pandoc or weasyprint is missing instead of
# reporting success for a PDF that was never produced.
pdf:
	pandoc reports/report.md -o reports/report.pdf --pdf-engine=weasyprint

# Run tests
test:
	pytest -v

# Lint
lint:
	ruff check src tests scripts

# Type check
typecheck:
	mypy src

# Clean generated files (keep raw data)
clean:
	rm -rf data/processed results/metrics.csv results/figures/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Clean everything including raw data
clean-all: clean
	rm -rf data/raw/*

# Run single config for debugging
run-single:
	python -m src.calibmiss.run_experiment \
		--model logistic_regression \
		--imputer median \
		--calibrator platt \
		--mechanism MCAR \
		--rate 0.1 \
		--seed 42