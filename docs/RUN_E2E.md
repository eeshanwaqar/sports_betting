# Running the Application End-to-End

This guide explains how to run the full EPL Betting Odds Predictor pipeline and then the API + frontend.

---

## Option A: One-time setup, then start the app

### 1. Pipeline (data → features → model)

Run these in order. You can use the **all-in-one script** or run each step yourself.

**All-in-one (recommended):**

```bash
# From project root, with your venv activated (e.g. dev_env\Scripts\activate)
python scripts/run_e2e.py
```

This will:

1. Load and clean raw data → `data/raw/matches.csv`
2. Build features → `data/features/features.csv`, `data/features/model_ready.csv`
3. Train the best model → `models/best_model.joblib` and related artifacts

**Or run step-by-step:**

```bash
# 1) Data: load season CSVs, clean, save to data/raw/
python -c "from src.data.loader import run_pipeline; from src.utils.config import load_config; run_pipeline(load_config())"

# 2) Features: build features from matches, save to data/features/
python -c "from src.features.builder import run_pipeline; from src.utils.config import load_config; run_pipeline(load_config())"

# 3) Train: train models and save best to models/
python scripts/train.py
```

### 2. Start the application (API + frontend)

You need **two terminals** (or run the API in the background).

**Terminal 1 – API (port 8000):**

```bash
python scripts/run_api.py
```

**Terminal 2 – Frontend (port 3000):**

```bash
python scripts/run_frontend.py
```

Then open: **http://localhost:3000** in your browser. The frontend talks to the API at http://localhost:8000.

---

## Option B: Run pipeline and start servers from one command

To run the full pipeline and then start both API and frontend in the background from one command:

```bash
python scripts/run_e2e.py --serve
```

This will:

1. Run the pipeline (data → features → train) as above.
2. Start the API and the frontend server in the background.
3. Print the URLs and wait; press **Ctrl+C** to stop both servers.

---

## Prerequisites

- **Python** with the project dependencies installed (e.g. `pip install -r requirements-ml.txt` or your project’s requirements).
- **Raw data**: Season CSV files under `archive/Datasets/` (or the path set in `config.yaml` under `source.path`). If missing, the data step will fail with a clear error.

---

## Quick reference

| Goal                    | Command / step |
|-------------------------|----------------|
| Full pipeline only      | `python scripts/run_e2e.py` |
| Pipeline + start app    | `python scripts/run_e2e.py --serve` |
| API only                | `python scripts/run_api.py` |
| Frontend only           | `python scripts/run_frontend.py` |
| Single prediction (CLI) | `python scripts/predict.py --home Arsenal --away Chelsea` |
| Run tests               | `pytest tests/ -v` |
