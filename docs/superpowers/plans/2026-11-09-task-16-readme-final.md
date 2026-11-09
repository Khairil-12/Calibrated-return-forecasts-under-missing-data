# Task 16: README and Final Polish

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create comprehensive README with one-command reproduction and verify full pipeline works.

**Architecture:** Markdown file `README.md` at repo root.

**Tech Stack:** Markdown, make, pytest, ruff, mypy.

**Spec:** PRD Sections 12, 20.6, 20.7

---

### Files
- Create/Modify: `README.md`

### Content Structure

```markdown
# CalibMiss: Calibrated Return Forecasts Under Missing Data

Research pipeline for studying probabilistic classifier calibration under missing data in financial time series.

## Quick Start

```bash
# 1. Clone and enter
git clone <repo-url> calibmiss
cd calibmiss

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pre-commit install

# 4. Run full pipeline (downloads data, computes features, runs experiments, generates figures, builds report)
make all
```

## Pipeline Steps

```bash
make setup      # Install dependencies
make data       # Download OHLCV data (cached)
make features   # Compute features and target
make experiment # Run full experimental grid
make figures    # Generate all figures
make report     # Compile PDF report
```

## Project Structure

```
calibmiss/
├── config/           # YAML configuration files
├── data/
│   ├── raw/          # Cached raw data (gitignored)
│   └── processed/    # Features and target (gitignored)
├── src/calibmiss/    # Main package
├── scripts/          # Executable scripts
├── tests/            # Unit tests
├── notebooks/        # Exploration notebook
├── results/          # Experiment outputs
│   ├── metrics.csv   # All results
│   └── figures/      # Generated plots
└── reports/          # Written report
```

## Configuration

- `config/universe.yaml` - Ticker universe and date range
- `config/experiment.yaml` - Model, imputer, calibrator, missingness, split configs
- `config/missingness.yaml` - MAR dependence rules

## Reproducing Results

The full pipeline is reproducible with a single command:
```bash
make all
```

Expected runtime: < 2 hours on standard laptop.
Results saved to `results/metrics.csv` and `results/figures/`.
Report saved to `reports/report.pdf`.

## Testing

```bash
# Run all tests
pytest -v

# With coverage
pytest --cov=src/calibmiss

# Lint
make lint

# Type check
make typecheck
```

## Key Findings

[To be filled after experiments complete]

## License

MIT
```

---

### Step 1: Write README

Create the markdown file with all sections.

---

### Step 2: Verify `make all` works from clean clone

Test in a temporary directory:
```bash
cd /tmp
git clone <local-repo-path> test-clone
cd test-clone
make all
```

---

### Step 3: Run full test suite

**Run:** `pytest -v --cov=src/calibmiss`
**Expected:** All tests pass, ≥70% coverage on features and missingness modules

---

### Step 4: Run linting and type checking

**Run:** `make lint typecheck`
**Expected:** No errors

---

### Step 5: Commit

```bash
git add README.md
git commit -m "docs: add README and verify full pipeline"
```

---

## Dependencies

- **Requires:** All previous tasks (1-15)
- **Blocks:** None (final task)

---

## Audit Addendum

Confirmed gaps against PRD Sections 12, 14, 20.5-20.8:

- README example assumes `requirements.txt`, but dependency source is `pyproject.toml`; `pre-commit install` and `make setup` must be checked against actual Makefile/project config before documenting.
- `make all` claim needs a tested dependency order covering data download, feature generation, experiment, figures, report, and summary. Include network/data prerequisites and avoid claiming reproducibility until a clean offline/online path is verified.
- README must explain generated metadata (data date and dependency versions), baseline references, uncertainty/statistics, and where the one-page summary lives.
- Current verification criteria assert full-grid <2 hours without specifying representative hardware or recording measured runtime; define and report the environment and measured duration.
- PRD's required output paths include `reports/summary.md`; ensure reproduction command generates it alongside `reports/report.pdf`.
- Clean-clone test uses placeholder `<repo-url>` and `/tmp`, which is not portable to Windows. Use a local repository path and platform-neutral temporary directory handling.

## Success Criteria

- [ ] README has clear reproduction instructions
- [ ] `make all` works from clean clone
- [ ] All tests pass
- [ ] Coverage ≥70% on features and missingness modules
- [ ] Linting passes (ruff)
- [ ] Type checking passes (mypy)
- [ ] Pipeline completes in < 2 hours
- [ ] Results and figures generated
- [ ] Report PDF generated