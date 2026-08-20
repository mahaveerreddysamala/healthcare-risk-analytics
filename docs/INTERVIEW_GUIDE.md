# Interview Guide

## Modeling
- Why this model? Start with an interpretable baseline, then compare nonlinear ensemble models.
- How is risk evaluated? Use ROC-AUC plus precision, recall, F1, calibration, and confusion-matrix analysis.
- Why threshold tuning? In healthcare risk screening, the cost of missed high-risk patients can differ substantially from false positives.

## Data quality
Validate ranges, missingness, duplicate records, leakage, and temporal availability of features before training.

## Responsible ML
Check subgroup performance, explainability, calibration, and potential bias. This project is for analytics demonstration and is not a clinical decision tool.

## Production extensions
- Add cross-validation and model registry/versioning.
- Monitor drift and calibration.
- Log model inputs and predictions with appropriate privacy controls.
