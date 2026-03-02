# ADR 001: ML Framework Choice

## Status
Accepted

## Context
Need to choose ML framework for model training.

## Decision
Use XGBoost with scikit-learn for:
- Excellent performance on tabular data
- Fast training
- Good interpretability
- Industry standard

## Consequences
- Learning curve for XGBoost tuning
- Need to handle feature preprocessing
