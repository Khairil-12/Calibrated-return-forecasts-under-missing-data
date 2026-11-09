# Task 14: Exploration Notebook (`notebooks/exploration.ipynb`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create EDA notebook for data quality checks, feature distributions, missingness patterns, baseline model performance.

**Architecture:** Jupyter notebook `notebooks/exploration.ipynb`.

**Tech Stack:** Python 3.11+, pandas, matplotlib, seaborn, jupyter.

**Spec:** PRD Section 20.4 (notebooks/exploration.ipynb)

---

### Files
- Create: `notebooks/exploration.ipynb`

### Content Structure

```markdown
# CalibMiss - Exploratory Data Analysis

## 1. Data Quality Checks
- Load raw data
- Check for duplicates, negative prices, negative volume
- Native missing rates per column
- Date range coverage per ticker

## 2. Feature Distributions
- Load computed features
- Histograms of each feature
- Correlation matrix
- Target balance (up vs down days)

## 3. Missingness Patterns
- Simulate MCAR at different rates
- Simulate MAR with volume_ratio dependence
- Visualize missingness heatmaps
- Verify realized vs target rates

## 4. Baseline Model Performance
- Train logistic regression on complete data
- Train LightGBM on complete data
- Evaluate calibration (reliability diagrams)
- Compare discrimination (ROC AUC)

## 5. Walk-Forward Split Visualization
- Plot train/calibrate/test windows over time
- Verify purge gaps
- Show number of folds
```

---

### Step 1: Create notebook with all sections

Use Jupyter to create the notebook with markdown and code cells for each section.

---

### Step 2: Test notebook runs

**Run:** `jupyter nbconvert --execute notebooks/exploration.ipynb --to html`
**Expected:** Executes without errors

---

### Step 3: Commit

```bash
git add notebooks/exploration.ipynb
git commit -m "feat: add exploration notebook"
```

---

## Dependencies

- **Requires:** Tasks 1 (data.py), 2 (features.py), 3 (splits.py), 4 (missingness.py), 7 (models.py), 8 (metrics.py)

---

## Audit Addendum

Confirmed gaps against PRD Sections 10.1-10.4 and 14.1:

- Notebook tests only command execution; no outputs are saved or checked. Add deterministic synthetic/offline execution validation for required figures/tables and avoid requiring Yahoo Finance access during CI.
- Baseline section omits required naive training-base-rate predictor and complete-data oracle comparison. Include both references alongside model calibration metrics.
- Statistical design should expose variability across folds/seeds and paired-comparison method/effect-size/multiple-testing correction when results are available; keep notebook aligned with Task 10 outputs.
- Resolve repository data/results paths independently of notebook launch cwd and handle absent downloaded data with an actionable setup path.

## Success Criteria

- [ ] Notebook has all 5 sections
- [ ] Runs without errors
- [ ] Generates useful visualizations
- [ ] `jupyter nbconvert` executes cleanly