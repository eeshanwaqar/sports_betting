# Improvements Overview & Next Steps

This document summarizes the enhancements made to the EPL Betting Odds Predictor and where to find model results. Use it to assess progress and decide next steps.

---

## 1. What Was Done

### Phase 1: Test Suite
- **conftest.py** – Shared fixtures: `sample_matches_df`, `cutoff_date`, `late_cutoff_date` for reproducible tests.
- **Unit tests (features)** – `test_base.py`, `test_form.py`, `test_h2h.py`, `test_team_stats.py`, `test_elo.py`.
- **Unit tests (inference)** – `test_odds_calculator.py` (prob_to_odds, probs_to_odds, calc_value, is_value_bet).
- **Integration tests (API)** – `test_api.py`: root, health, predict (single/batch), teams, matches/recent, head-to-head; uses mocked predictor so no model files needed.
- **Result:** 92 tests, all passing. Run with: `pytest tests/ -v`.

### Phase 2: XGBoost & Prediction Script
- **XGBoost** installed and added to the trainer’s model set (with tuned defaults).
- **scripts/predict.py** rewritten to use the production pipeline:
  - `MatchPredictor` from `src.inference.predictor`
  - Same logic as the API (features → scale if LR → predict → odds).
  - Supports `--list-teams`, `--home`, `--away`, `--date`.
- **Training:** Full train (with MLflow) and no-MLflow train both run; XGBoost became best model (54.3% test accuracy, 0.444 F1 macro) and was registered in MLflow.

### Phase 3: Hyperparameter Tuning
- **src/training/hyperparameter_tuning.py** – Optuna studies per model (LR, RF, GB, XGBoost), TimeSeriesSplit CV, optional MLflow logging.
- **scripts/tune.py** – CLI: `--trials`, `--folds`, `--no-mlflow`, `--retrain`, `--data-path`.
- **Tuning run:** 30 trials per model, 5-fold CV; best CV: XGBoost (53.44%). Retrain with tuned params saved a new best model (current `model_info.json` shows Logistic Regression as best on that test run at 54.32%).

---

## 2. Where to See Model Results

| What | Where |
|------|--------|
| **Current best model & all-model summary** | `models/model_info.json` – `model_type`, `test_accuracy`, `test_f1_macro`, `all_models` (per-model acc/F1). |
| **Tuning best params & CV scores** | `models/tuning_results.json` – `best_params`, `best_scores`, `overall_best_model`. |
| **Feature list used by the model** | `models/features.txt` – one feature per line. |
| **Test set predictions** | `data/features/test_predictions.csv` – test rows plus `predicted`, `prob_H`, `prob_D`, `prob_A`. |
| **MLflow experiments & runs** | Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` then open http://localhost:5000 – experiments, runs, metrics, and registered model versions. |
| **Evaluation notebook (if run)** | `notebooks/05_evaluation.ipynb` – accuracy vs bookmaker, value bets, calibration, P/L simulation. |
| **Console summary after training** | Output of `python scripts/train.py` or `python scripts/tune.py --retrain` – best model name, test acc, F1, and per-model table. |

---

## 3. Progress Snapshot

- **Before (from prior work):** Best model was Logistic Regression ~54.2% test accuracy; bookmaker baseline ~55.2%.
- **After Phase 2:** XGBoost best at **54.3%** test accuracy, **0.444** F1 macro; still slightly below bookmaker.
- **After Phase 3 (tuning + retrain):** Tuned models retrained; saved best in the last run was **Logistic Regression at 54.32%** test accuracy (tuned LR won on that test split). Tuning improved CV (e.g. XGBoost CV 53.44%); test-set winner can vary by split and tuning randomness.

**Takeaway:** We added tests, productionized prediction (script + API alignment), brought in XGBoost, and introduced systematic tuning. Accuracy is in the 54–55% band; we have not yet consistently exceeded the bookmaker baseline (~55.2%). Next steps should focus on data, features, or evaluation (e.g. value betting, P/L) as much as on further tuning.

---

## 4. Recommended Next Steps

**Short term (quick wins)**  
1. **Run evaluation notebook** – Execute `05_evaluation.ipynb` on current `test_predictions.csv` (and raw odds if available) to get bookmaker comparison, value-bet rate, and P/L.  
2. **Promote a champion in MLflow** – Use `ModelRegistry.promote_to_champion(version)` so the API/serving always uses the chosen model version.  
3. **Add pytest to CI** – Run `pytest tests/` in GitHub Actions (or your CI) on every commit.

**Medium term (model & data)**  
4. **More data or fresher data** – Include recent seasons (e.g. up to 2023–24) if not already; ensure `model_ready.csv` is rebuilt after data updates.  
5. **Feature ablation** – Use the existing feature importance and ablation in the modeling notebook to drop weak/noisy features and retrain.  
6. **Calibration** – Fit Platt scaling or isotonic regression on validation probabilities to improve probability estimates and value-bet detection.

**Longer term (product & ops)**  
7. **Value-bet API or report** – Expose “value bets” (e.g. where model prob > implied odds prob by a threshold) via API or a small report.  
8. **Scheduled retrains** – Cron or pipeline to run `scripts/train.py` (or `tune.py --retrain`) when new data is available.  
9. **Monitoring** – Track prediction distribution and accuracy over time (e.g. Evidently or custom metrics in MLflow).

---

## 5. Quick Reference Commands

```bash
# Run all tests
pytest tests/ -v

# Train (with MLflow)
python scripts/train.py

# Train without MLflow
python scripts/train.py --no-mlflow

# Single prediction
python scripts/predict.py --home Arsenal --away Chelsea

# List teams
python scripts/predict.py --list-teams

# Hyperparameter tuning (then retrain with best params)
python scripts/tune.py --trials 50 --retrain

# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Use this overview to see where model results live and to choose the next improvements (e.g. evaluation, champion promotion, more data, or value-bet reporting).
