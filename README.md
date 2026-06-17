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

```python
from valmetrics.discrimination import gini_standard, roc_auc
from valmetrics.calibration import hosmer_lemeshow
from valmetrics.stability import psi_continuous
from valmetrics.diagnostics import herfindahl_hirschman

y_true = [0, 0, 1, 1, 0, 1]
y_prob = [0.05, 0.10, 0.40, 0.60, 0.20, 0.80]

auc = roc_auc(y_true, y_prob)

gini = gini_standard(y_true, y_prob)

hl = hosmer_lemeshow(y_true, y_prob, n_groups=3)

psi = psi_continuous(
    expected=[0.01, 0.02, 0.03, 0.04],
    actual=[0.02, 0.03, 0.05, 0.06],
    bins=2,
)

hhi = herfindahl_hirschman(["A", "A", "B", "C"], normalized=True)
```

## Documentation

- [Methodology](docs/methodology.md)
- [PD model validation example notebook](examples/pd_model_validation.ipynb)
- [Synthetic data generator](examples/data_generator.py)

## Example Workflow

The repository includes an end-to-end synthetic PD model validation notebook.

The notebook demonstrates:
- synthetic credit-risk data generation;
- logistic regression model fitting;
- development, validation, and out-of-time sample scoring;
- rating grade construction;
- discrimination analysis;
- calibration testing;
- population stability analysis;
- concentration diagnostics;
- final validation conclusions.

## Methodological Notes

The metrics are designed for validation workflows, not for automatic model approval.

Important limitations:
- binomial calibration tests use a group-level average-PD approximation;
- PSI is sensitive to binning choices and sample size;
- HHI and HCI are concentration diagnostics, not model performance metrics.

See [docs/methodology.md](docs/methodology.md) for details.

## Roadmap

Potential future additions:
- confidence intervals for discrimination metrics;
- Poisson-binomial calibration test;
- calibration plots and summary helpers;
- migration matrix diagnostics;
- LGD validation metrics;
- EAD / CCF validation metrics;
- additional model monitoring utilities.

These items are not part of the current `0.1.0` scope.

## License

MIT License.
