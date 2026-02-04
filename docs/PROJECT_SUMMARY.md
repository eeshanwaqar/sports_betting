# EPL Betting Odds Predictor - Project Summary

## Overview

A **production-grade sports betting prediction system** that predicts English Premier League match outcomes and generates betting odds using machine learning, real-time news analysis, and LLM-powered explanations.

**Input:** "Arsenal vs Chelsea"  
**Output:** Win probabilities (H/D/A) + Fair betting odds + AI-generated explanation

---

## What This Project Demonstrates

| Skill Category | Technologies & Techniques |
|----------------|---------------------------|
| **Machine Learning** | XGBoost, Scikit-learn, Feature Engineering, Model Evaluation |
| **MLOps** | MLflow (Experiment Tracking, Model Registry), DVC (Data Versioning) |
| **Data Engineering** | ETL Pipelines, Feature Store (Feast), Data Validation (Great Expectations) |
| **NLP/Text Processing** | spaCy NER, Sentiment Analysis, News Classification |
| **LLM/GenAI** | RAG (LangChain + ChromaDB), Prompt Engineering, Explainable AI |
| **Backend Development** | FastAPI, REST APIs, Async Python, Pydantic |
| **Real-Time Systems** | Event-Driven Architecture, Redis Streams, Caching |
| **DevOps/Infrastructure** | Docker, Docker Compose, GitHub Actions CI/CD |
| **Monitoring** | Evidently AI (Drift Detection), Performance Tracking |
| **Software Engineering** | Type Hints, Testing (Pytest), Pre-commit Hooks, Code Quality (Ruff, MyPy) |

---

## Core Features

### 1. Match Outcome Prediction
- Predict Home Win / Draw / Away Win probabilities
- Convert probabilities to decimal betting odds
- Confidence scoring based on data freshness and model certainty

### 2. Feature Engineering Pipeline
- **Team Form:** Last 5/10 match performance, win streaks
- **Goals Statistics:** Scoring/conceding averages, clean sheet rates
- **Head-to-Head:** Historical matchup records
- **League Position:** Current standings, points
- **Temporal:** Match scheduling, rest days, fixture congestion

### 3. News-Enhanced Predictions
- Real-time news collection from multiple sources
- Injury detection and impact scoring
- Sentiment analysis for team morale
- Automatic feature updates when news arrives

### 4. Conversational AI (RAG)
- Natural language queries: *"Why will Liverpool win?"*
- Context-aware explanations combining stats + news
- LLM-generated prediction reasoning

### 5. Real-Time Updates
- Event-driven architecture for instant feature updates
- Cache invalidation on breaking news
- Sliding window aggregations for time-based features

---

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                        TECHNOLOGY STACK                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DATA LAYER                                                     │
│  ├── Historical Data: 20 seasons EPL (7,600+ matches)          │
│  ├── Storage: PostgreSQL, CSV files                            │
│  ├── Versioning: DVC                                           │
│  └── Validation: Great Expectations                            │
│                                                                 │
│  ML LAYER                                                       │
│  ├── Models: XGBoost, Scikit-learn                             │
│  ├── Experiment Tracking: MLflow                               │
│  ├── Feature Store: Feast + Redis                              │
│  └── Hyperparameter Tuning: Optuna                             │
│                                                                 │
│  NLP LAYER                                                      │
│  ├── Entity Recognition: spaCy                                 │
│  ├── Sentiment: Transformers / VADER                           │
│  └── Classification: Custom models                             │
│                                                                 │
│  RAG LAYER                                                      │
│  ├── Framework: LangChain                                      │
│  ├── Vector DB: ChromaDB                                       │
│  ├── Embeddings: Sentence Transformers                         │
│  └── LLM: Ollama (local) / OpenAI                              │
│                                                                 │
│  API LAYER                                                      │
│  ├── Framework: FastAPI                                        │
│  ├── Cache: Redis                                              │
│  └── Auth: API Key (optional)                                  │
│                                                                 │
│  INFRASTRUCTURE                                                 │
│  ├── Containers: Docker, Docker Compose                        │
│  ├── CI/CD: GitHub Actions                                     │
│  ├── IaC: Terraform (optional)                                 │
│  └── Monitoring: Evidently AI                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   DATA SOURCES                                                              │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                          │
│   │Historical│ │  News   │ │  RSS    │ │ Twitter │                          │
│   │   CSV   │ │   API   │ │  Feeds  │ │   /X    │                          │
│   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘                          │
│        │           │           │           │                                │
│        └───────────┴───────────┴───────────┘                                │
│                         │                                                   │
│                         ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      DATA PIPELINE                                   │  │
│   │  loader.py → cleaner.py → validator.py → feature engineering        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                         │                                                   │
│          ┌──────────────┴──────────────┐                                   │
│          ▼                             ▼                                   │
│   ┌─────────────────┐           ┌─────────────────┐                        │
│   │  FEATURE STORE  │           │   NEWS PIPELINE │                        │
│   │  (Feast/Redis)  │           │   (NLP/Sentiment)│                       │
│   └────────┬────────┘           └────────┬────────┘                        │
│            │                             │                                  │
│            └──────────────┬──────────────┘                                  │
│                           ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                       ML MODEL (XGBoost)                             │  │
│   │                  Trained with MLflow tracking                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                           │                                                 │
│            ┌──────────────┴──────────────┐                                  │
│            ▼                             ▼                                  │
│   ┌─────────────────┐           ┌─────────────────┐                        │
│   │   PREDICTION    │           │   RAG ENGINE    │                        │
│   │   (Odds/Probs)  │           │   (LangChain)   │                        │
│   └────────┬────────┘           └────────┬────────┘                        │
│            │                             │                                  │
│            └──────────────┬──────────────┘                                  │
│                           ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      FastAPI REST API                                │  │
│   │              /predict  /chat  /teams  /health                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                           │                                                 │
│                           ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    Dashboard (Streamlit)                             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
EPL-Predictor/
│
├── .github/workflows/            # CI/CD pipelines
│   ├── ci.yml                    # Lint, test, type check
│   ├── train.yml                 # Model training pipeline
│   ├── build.yml                 # Docker build
│   └── deploy.yml                # Deployment
│
├── src/                          # Source Code
│   ├── data/                     # Data pipeline
│   │   ├── loader.py             # Load CSV data
│   │   ├── cleaner.py            # Clean and standardize
│   │   ├── validator.py          # Great Expectations
│   │   ├── splitter.py           # Train/test splits
│   │   └── schemas.py            # Pydantic models
│   │
│   ├── features/                 # Feature engineering
│   │   ├── form.py               # Recent form features
│   │   ├── team_stats.py         # Goals, shots averages
│   │   ├── h2h.py                # Head-to-head features
│   │   ├── league_position.py    # Standing features
│   │   ├── temporal.py           # Time-based features
│   │   ├── builder.py            # Combine all features
│   │   └── store.py              # Feast integration
│   │
│   ├── news/                     # News processing
│   │   ├── collectors/           # NewsAPI, RSS, Twitter
│   │   ├── processors/           # NER, sentiment, classifier
│   │   ├── features/             # News-derived features
│   │   └── scheduler.py          # Periodic collection
│   │
│   ├── training/                 # Model training
│   │   ├── trainer.py            # Core training logic
│   │   ├── mlflow_trainer.py     # MLflow integration
│   │   ├── hyperparameter_tuning.py  # Optuna
│   │   ├── evaluator.py          # Metrics and evaluation
│   │   └── model_registry.py     # MLflow registry
│   │
│   ├── inference/                # Prediction pipeline
│   │   ├── predictor.py          # Main prediction logic
│   │   ├── model_loader.py       # Load from MLflow
│   │   ├── feature_assembler.py  # Assemble features
│   │   ├── odds_calculator.py    # Probability to odds
│   │   └── confidence_scorer.py  # Confidence rating
│   │
│   ├── rag/                      # RAG pipeline
│   │   ├── embeddings.py         # Create embeddings
│   │   ├── vector_store.py       # ChromaDB
│   │   ├── retriever.py          # Context retrieval
│   │   ├── llm_provider.py       # LLM abstraction
│   │   ├── chains/               # LangChain chains
│   │   └── prompts/              # Prompt templates
│   │
│   ├── realtime/                 # Real-time processing
│   │   ├── event_handlers/       # Injury, lineup, news handlers
│   │   ├── stream_processor.py   # Redis Streams consumer
│   │   ├── feature_updater.py    # Update hot features
│   │   ├── cache_manager.py      # Cache invalidation
│   │   └── aggregators.py        # Sliding windows
│   │
│   ├── api/                      # FastAPI application
│   │   ├── app.py                # Main application
│   │   ├── routes/               # API endpoints
│   │   ├── middleware/           # Logging, rate limiting
│   │   ├── dependencies.py       # DI container
│   │   └── schemas.py            # Request/response models
│   │
│   ├── monitoring/               # Model monitoring
│   │   ├── drift_detector.py     # Evidently integration
│   │   ├── performance_tracker.py
│   │   ├── metrics_collector.py
│   │   └── alerting.py
│   │
│   ├── database/                 # Database layer
│   │   ├── connection.py
│   │   ├── models.py             # SQLAlchemy models
│   │   ├── repositories/         # Data access layer
│   │   └── migrations/           # Alembic migrations
│   │
│   └── utils/                    # Utilities
│       ├── config.py
│       ├── logger.py
│       ├── helpers.py
│       └── constants.py
│
├── scripts/                      # Executable scripts
│   ├── train.py
│   ├── predict.py
│   ├── run_api.py
│   └── ...
│
├── tests/                        # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── notebooks/                    # Jupyter notebooks
├── infra/                        # Docker & Terraform
├── configs/                      # YAML configurations
├── dashboard/                    # Streamlit app
├── web/                          # HTML frontend
├── docs/                         # Documentation
└── archive/Datasets/             # Historical EPL data
```

---

## Development Phases

| Phase | Focus | Key Deliverables |
|-------|-------|------------------|
| **0** | Setup | Environment, tooling, code quality |
| **1** | Data Pipeline | Load, clean, validate, version data |
| **2** | Feature Engineering | 25-35 predictive features |
| **3** | Model Training | MLflow tracking, hyperparameter tuning |
| **4** | Inference | Prediction API, odds calculation |
| **5** | Docker | Containerize all services |
| **6** | API | FastAPI endpoints |
| **7** | CI/CD | GitHub Actions automation |
| **8** | News Integration | NLP pipeline, real-time features |
| **9** | RAG | LangChain, explainable predictions |
| **10** | Real-Time | Event-driven updates |
| **11** | Monitoring | Drift detection, alerting |
| **12** | Dashboard | Streamlit UI, documentation |

---

## Data Sources

| Source | Type | Purpose |
|--------|------|---------|
| Historical CSV (2000-2020) | Static | 20 seasons of EPL match data |
| EPL Standings | Static | League position data |
| NewsAPI | Dynamic | Real-time sports news |
| RSS Feeds (BBC, Sky, ESPN) | Dynamic | Breaking news |
| Twitter/X (optional) | Dynamic | Official club announcements |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict` | POST | Single match prediction |
| `/api/v1/predict/batch` | POST | Multiple match predictions |
| `/api/v1/predict/upcoming` | GET | Next gameweek predictions |
| `/api/v1/teams` | GET | List all teams |
| `/api/v1/teams/{name}` | GET | Team details & form |
| `/api/v1/matches/head-to-head` | GET | H2H record |
| `/api/v1/chat` | POST | RAG chat interface |
| `/health` | GET | Health check |

---

## Sample Prediction Output

```json
{
  "match": "Liverpool vs Manchester City",
  "date": "2024-01-20",
  "prediction": "H",
  "probabilities": {
    "home_win": 0.42,
    "draw": 0.28,
    "away_win": 0.30
  },
  "odds": {
    "home_win": 2.38,
    "draw": 3.57,
    "away_win": 3.33
  },
  "confidence": 0.78,
  "factors": {
    "home_form": "WWDWL (10 pts)",
    "away_form": "WDWWL (10 pts)",
    "injuries": "Liverpool: Salah (FIT), City: Haaland (DOUBT)",
    "h2h_last_5": "Liverpool 2W, 1D, 2L"
  },
  "explanation": "Liverpool are slight favorites at home due to their strong Anfield record (4W in last 5). City's uncertainty around Haaland's fitness reduces their away threat..."
}
```

---

## Expected Model Performance

| Metric | Value |
|--------|-------|
| Random Baseline | ~33% (3 classes) |
| Target Accuracy | 53-57% |
| Excellent Model | 57-60% |

---

## Quick Start Commands

```bash
# Setup
make install            # Install dependencies

# Data Pipeline
make data-load          # Load all seasons
make data-clean         # Clean and validate

# Training
make train              # Train with MLflow

# API
make api                # Start FastAPI server

# Docker
make docker-up          # Start all services
make docker-down        # Stop all services

# Quality
make lint               # Run linting
make test               # Run tests
make format             # Format code
```

---

## Key Configuration Files

| File | Purpose |
|------|---------|
| `config.yaml` | Main application configuration |
| `configs/mlflow_config.yaml` | MLflow settings |
| `configs/feast_config.yaml` | Feature store settings |
| `pyproject.toml` | Project & tool configuration |
| `Makefile` | Common commands |
| `dvc.yaml` | Data pipeline definition |
| `docker-compose.yml` | Service orchestration |

---

## Future Enhancements

- Live in-play predictions
- Betting value detection (edge over bookmakers)
- Mobile app interface
- Cloud deployment (AWS/GCP)
- A/B testing for model versions
- Multi-league support (La Liga, Bundesliga, etc.)

---

## Repository

**GitHub:** [github.com/eeshanwaqar/sports_betting](https://github.com/eeshanwaqar/sports_betting)

---

*This project serves as a comprehensive portfolio piece demonstrating the full spectrum of ML Engineering and MLOps skills required in modern AI/ML roles.*
