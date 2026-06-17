# Model Validation Metrics

![CI](https://github.com/mengden-quant/model-validation-metrics/actions/workflows/ci.yml/badge.svg)

Production-style Python toolkit for model validation metrics used in credit risk and model risk management.

## Status

Current version: `0.1.0`

This is an early-stage project. The API is usable for examples and validation prototypes, but may change before version `1.0.0`.

## Current Scope

### Discrimination

- AUC-ROC
- Standard Gini
- Conservative Gini
- Conservative tie correction
- Kolmogorov-Smirnov statistic

### Calibration

- Hosmer-Lemeshow calibration test
- Grouped Hosmer-Lemeshow calibration test
- Binomial calibration test
- Grouped binomial calibration test

### Stability

- Continuous Population Stability Index
- Categorical Population Stability Index

### Diagnostics

- Herfindahl-Hirschman Index
- Adjusted Herfindahl-Hirschman Index
- Highest Concentration Index

---

## Installation

Clone the repository and install dependencies with Poetry:

```bash
git clone https://github.com/mengden-quant/model-validation-metrics.git
cd model-validation-metrics
poetry install
```

---

## Quick Start

### Discrimination

```python
from valmetrics.discrimination import gini_conservative, gini_standard, ks_statistic
y_true = [1, 1, 1, 0, 0, 0]
y_score = [0.90, 0.80, 0.70, 0.30, 0.20, 0.10]
gini = gini_standard(y_true, y_score)
gini_cons = gini_conservative(y_true, y_score)
ks = ks_statistic(y_true, y_score)
```

### Calibration

```python
from valmetrics.calibration import hosmer_lemeshow
result = hosmer_lemeshow(
    y_true=[0, 0, 1, 1, 0, 1, 0, 1],
    y_prob=[0.10, 0.20, 0.80, 0.70, 0.20, 0.90, 0.30, 0.60],
    n_groups=4,
)
result.statistic
result.p_value
```

### Stability

```python
from valmetrics.stability import psi_continuous
psi = psi_continuous(
    expected=[0.01, 0.02, 0.03, 0.04, 0.05],
    actual=[0.03, 0.04, 0.05, 0.06, 0.07],
    bins=3,
)
```

### Diagnostics

```python
from valmetrics.diagnostics import hci, herfindahl_hirschman
groups = ["A", "A", "B", "C", "C", "C"]
hhi = herfindahl_hirschman(groups, normalized=True)
largest_groups = hci(groups)
```

---

## Status

Early development.
