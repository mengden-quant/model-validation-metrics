# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-06-17

### Added

- Added shared input validation utilities for binary targets, numeric scores, probabilities, and group labels.
- Added discrimination metrics:
  - ROC AUC;
  - standard Gini;
  - conservative Gini;
  - conservative tie correction;
  - Kolmogorov-Smirnov statistic.
- Added calibration metrics:
  - Hosmer-Lemeshow calibration test;
  - grouped Hosmer-Lemeshow calibration test;
  - binomial calibration test;
  - grouped binomial calibration test.
- Added stability metrics:
  - continuous Population Stability Index;
  - categorical Population Stability Index.
- Added concentration diagnostics:
  - Herfindahl-Hirschman Index;
  - adjusted Herfindahl-Hirschman Index;
  - Highest Concentration Index.
- Added methodology documentation for implemented validation metrics.
- Added synthetic PD validation data generator.
- Added end-to-end PD model validation example notebook.
