# Task 13: Figure Generation (`scripts/generate_figures.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate all report figures from results/metrics.csv.

**Architecture:** Script `scripts/generate_figures.py` using matplotlib/seaborn.

**Tech Stack:** Python 3.11+, pandas, matplotlib, seaborn.

**Spec:** PRD Sections 14.3, 20.5 (make figures), 20.8

---

### Files
- Create: `scripts/generate_figures.py`

### Interfaces
- **Consumes:** `results/metrics.csv`
- **Produces:** Figures in `results/figures/`

---

### Step 1: Implement figure generation

```python
#!/usr/bin/env python
"""Generate all report figures from results."""
import sys
sys.path.insert(0, "src")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
os.makedirs("results/figures", exist_ok=True)

def main():
    df = pd.read_csv("results/metrics.csv")
    
    # 1. Brier score vs missingness rate
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    mechanisms = ["MCAR", "MAR"]
    for i, mech in enumerate(mechanisms):
        for j, cal in enumerate(["none", "platt", "isotonic"]):
            ax = axes[i, j]
            subset = df[(df.mechanism == mech) & (df.calibrator == cal)]
            for imp in subset.imputer.unique():
                imp_data = subset[subset.imputer == imp]
                grouped = imp_data.groupby("rate")["brier"].mean()
                ax.plot(grouped.index, grouped.values, label=imp, marker="o")
            ax.set_title(f"{mech} - {cal}")
            ax.set_xlabel("Missing Rate")
            ax.set_ylabel("Brier Score")
            ax.legend()
    plt.tight_layout()
    plt.savefig("results/figures/brier_vs_missingness.png", dpi=150)
    plt.close()
    
    # 2. ECE vs missingness rate (same structure)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    for i, mech in enumerate(mechanisms):
        for j, cal in enumerate(["none", "platt", "isotonic"]):
            ax = axes[i, j]
            subset = df[(df.mechanism == mech) & (df.calibrator == cal)]
            for imp in subset.imputer.unique():
                imp_data = subset[subset.imputer == imp]
                grouped = imp_data.groupby("rate")["ece"].mean()
                ax.plot(grouped.index, grouped.values, label=imp, marker="o")
            ax.set_title(f"{mech} - {cal}")
            ax.set_xlabel("Missing Rate")
            ax.set_ylabel("ECE")
            ax.legend()
    plt.tight_layout()
    plt.savefig("results/figures/ece_vs_missingness.png", dpi=150)
    plt.close()
    
    # 3. Reliability diagrams at 0% and 30%
    for rate in [0.0, 0.3]:
        subset = df[(df.rate == rate) & (df.mechanism == "MCAR") & (df.imputer == "median") & (df.calibrator == "platt")]
        if len(subset) == 0:
            continue
        # Average reliability across folds/seeds
        rel_data = subset.iloc[0]["reliability"]  # Assuming stored as JSON string
        # Actually need to parse reliability dict from CSV
        # For now, placeholder
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Reliability Diagram ({int(rate*100)}% missing)")
        plt.tight_layout()
        plt.savefig(f"results/figures/reliability_{int(rate*100)}pct.png", dpi=150)
        plt.close()
    
    # 4. Heatmap of mean ECE across grid
    pivot = df.pivot_table(
        values="ece", 
        index=["model", "imputer", "calibrator"], 
        columns=["mechanism", "rate"], 
        aggfunc="mean"
    )
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn_r", center=pivot.median().median())
    plt.title("Mean ECE across Experimental Grid")
    plt.tight_layout()
    plt.savefig("results/figures/heatmap_calibration.png", dpi=150)
    plt.close()
    
    print("Figures saved to results/figures/")

if __name__ == "__main__":
    main()
```

---

### Step 2: Test manually after experiment runs

**Run:** `make figures`
**Expected:** PNG files in `results/figures/`

---

### Step 3: Commit

```bash
git add scripts/generate_figures.py
git commit -m "feat: add figure generation script"
```

---

## Dependencies

- **Requires:** Task 10 (run_experiment.py)
- **Blocks:** Task 15 (report.md)

---

## Audit Addendum

Confirmed gaps against PRD Sections 10.3, 14.3, and 20.8:

- The sample creates a 2×2 axes grid but indexes it as 2×3 (six panels), which raises an index error. Set layout dimensions to match two mechanisms by three calibrators.
- Reliability diagrams are placeholders: CSV serialization and bin aggregation are unspecified, and the example never plots reliability data. Define a tested JSON round-trip schema for bins, aggregate counts/accuracies/confidences across folds/seeds, and plot actual points plus the diagonal.
- Figure requirements specify reliability at 0% and 30%, Brier and ECE by imputer/calibrator/mechanism, and mean ECE heatmap. Test each expected output and include nonempty/sample input cases.
- Make paths repository-root-relative and create output directory inside `main`; current example performs filesystem effects at import time and assumes cwd.
- Handle one-class fold ROC AUC `NaN` in aggregate plotting without dropping valid Brier/ECE data.

## Success Criteria

- [ ] Script runs without errors
- [ ] Generates 4 required figures
- [ ] Figures saved to `results/figures/`
- [ ] `make figures` works