.PHONY: all setup data features experiment figures report clean clean-all test lint typecheck

# Full pipeline
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

# Generate figures from results
figures:
	python scripts/generate_figures.py

# Compile report
report:
	pandoc reports/report.md -o reports/report.pdf --pdf-engine=weasyprint || echo "pandoc/weasyprint not installed, skipping PDF"

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