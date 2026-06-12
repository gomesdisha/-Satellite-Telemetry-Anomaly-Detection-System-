# Phase 2 — Anomaly Injection + Classical ML Models

## Overview
Build on the Phase 1 preprocessing pipeline to:
1. Inject realistic synthetic anomalies into the clean telemetry data
2. Train and evaluate two classical anomaly detection models
3. Deliver section notebooks (10, 11, 12, 13) + one combined Phase 2 notebook

---

## Proposed Changes

### Section 10 — Anomaly Injection (`10_anomaly_injection.ipynb`)

**Goal:** Create a labelled dataset with realistic anomalies for model training/evaluation.

Three types of anomalies will be injected across different parameters and subsystems:

#### Point Anomalies
- Sudden extreme spike or drop — a single reading far outside the normal range
- Example: `BATT_VOLTAGE_1` suddenly reads 18.5V (normally 11.5–13.0V)
- Injection: replace N random readings with `mean ± k×std` where k = 4–6

#### Contextual Anomalies
- Value is unusual given the *time context*, not just its magnitude
- Example: `SOLAR_POWER_TOTAL` reads 0W during the daytime window (should be ~100W)
- Injection: at specific `hour` values, force parameters to "wrong-context" values

#### Collective Anomalies
- A sequence of readings that collectively represent an abnormal pattern
- Example: `TANK_PRESSURE` slowly drifts down over 20 consecutive readings (propellant leak pattern)
- Injection: apply a monotonic drift or oscillation over a contiguous window

**Output:**
- `telemetry_with_anomalies.csv` — labelled dataset (columns: all original + `is_anomaly`, `anomaly_type`)
- Summary table: how many anomalies, which parameters, what types
- Plots: before/after overlay showing injected anomalies clearly visible

---

### Section 11 — Model: Isolation Forest (`11_isolation_forest.ipynb`)

**How it works:**
- Builds random trees by repeatedly splitting data on random features at random thresholds
- Normal points require many splits to isolate (deep in tree)
- Anomalies require very few splits (isolated quickly near root)
- `anomaly_score = average path length across all trees`
- Short path → anomaly | Long path → normal

**What we implement:**
- Load `telemetry_standard_scaled.csv` (wide format, StandardScaler applied)
- Train Isolation Forest on **clean normal data only** (unsupervised — no labels used)
- Score the anomaly-injected dataset
- Threshold tuning: try contamination = 0.01, 0.05, 0.10
- Evaluation metrics: Precision, Recall, F1, Confusion Matrix, ROC curve
- Feature importance proxy: mean anomaly score contribution per parameter
- Plots: anomaly score distribution, flagged anomalies overlaid on time-series

---

### Section 12 — Model: One-Class SVM (`12_one_class_svm.ipynb`)

**How it works:**
- Learns a decision boundary (hyperplane in kernel-projected space) around normal data
- Points outside the boundary = anomalies
- Uses RBF kernel by default — maps data to high-dimensional space where normal points form a tight cluster
- `nu` parameter controls the fraction of training data allowed to be outside the boundary

**What we implement:**
- Load `telemetry_standard_scaled.csv`
- Train OCSVM on clean normal data (same train/test split as Isolation Forest)
- Tune `nu` parameter (0.01, 0.05, 0.10)
- Same evaluation metrics as Isolation Forest for direct comparison
- Plots: decision boundary (PCA-projected to 2D), anomaly flags on time-series

---

### Section 13 — Model Comparison (`13_model_comparison.ipynb`)

**What we implement:**
- Side-by-side metrics table: Isolation Forest vs One-Class SVM
- Precision, Recall, F1, AUC-ROC for both models
- Confusion matrices side by side
- Per-anomaly-type breakdown: which model detects point/contextual/collective better?
- Computation time comparison
- Recommendations for next stage (when to use which model, limitations)

---

### Combined Notebook — `phase2_combined.ipynb`
All 4 sections above merged into one master notebook with a clean flow:
Injection → Isolation Forest → One-Class SVM → Comparison

---

## File Structure After Phase 2

```
sections/
  10_anomaly_injection.ipynb        ← NEW
  11_isolation_forest.ipynb         ← NEW
  12_one_class_svm.ipynb            ← NEW
  13_model_comparison.ipynb         ← NEW

phase2_combined.ipynb               ← NEW (project root)

data/
  telemetry_with_anomalies.csv      ← NEW (labelled dataset)

plots_v2/
  10_*.png   ← anomaly injection plots
  11_*.png   ← isolation forest plots
  12_*.png   ← one-class svm plots
  13_*.png   ← comparison plots
```

---

## Open Questions

> [!IMPORTANT]
> **Anomaly proportion:** How many anomalies should be injected? Typical spacecraft anomaly datasets use 2–5% anomaly rate. Default plan: **~3%** (≈300 anomalies across 10,000 rows). Is this okay?

> [!IMPORTANT]
> **Which parameters to inject into?** Plan is to inject into parameters from all 6 subsystems (Power, Thermal, ADCS, Comms, OBC, Propulsion) — 2–3 parameters per subsystem. Confirm?

> [!NOTE]
> **Model training strategy:** Both models will be trained on the **clean normal data only** (unsupervised), then scored against the anomaly-injected data. This is the realistic spacecraft scenario — you never have labelled anomalies in advance.

---

## Verification Plan

### Automated
- Validate injected anomaly labels match actual injection locations
- Confirm model outputs have correct shape and score range
- Run `python validate_phase2.py` — same validation pattern as Phase 1

### Manual (you review)
- Anomaly injection plots: injected points should be **visually obvious**
- Anomaly score distribution: anomalies should cluster at the **high-score tail**
- Confusion matrix: check whether false negatives (missed anomalies) are acceptable
