# EPL Sports Betting Odds Predictor

A **production-grade machine learning system** for predicting English Premier League match outcomes and generating betting odds.

[![CI](https://github.com/eeshanwaqar/sports_betting/actions/workflows/ci.yml/badge.svg)](https://github.com/eeshanwaqar/sports_betting/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What This Does

**Input:** `Arsenal vs Chelsea`

**Output:**
```
Prediction: Home Win
Probabilities: H=45% | D=30% | A=25%
Odds:         H=2.22 | D=3.33 | A=4.00
Confidence:   78%

"Arsenal are favored due to their strong home form (4W in last 5)
and Chelsea's injury concerns in midfield..."
```

---

## Key Features

- **ML Predictions** - XGBoost model trained on 20 years of EPL data (7,600+ matches)
- **Real-Time News** - Incorporates injuries, transfers, team news via NLP
- **Explainable AI** - RAG-powered natural language explanations
- **Production Ready** - Docker, CI/CD, MLflow, monitoring

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML/MLOps** | XGBoost, Scikit-learn, MLflow, DVC, Feast |
| **NLP/LLM** | spaCy, LangChain, ChromaDB, Ollama |
| **Backend** | FastAPI, PostgreSQL, Redis |
| **DevOps** | Docker, GitHub Actions, Terraform |
| **Monitoring** | Evidently AI, Great Expectations |

---

## Project Structure

```
EPL-Predictor/
├── src/
│   ├── data/           # Data pipeline (load, clean, validate)
│   ├── features/       # Feature engineering (form, stats, h2h)
│   ├── news/           # News collection & NLP processing
│   ├── training/       # MLflow-integrated model training
│   ├── inference/      # Prediction pipeline
│   ├── rag/            # LangChain RAG implementation
│   ├── realtime/       # Event-driven updates
│   ├── api/            # FastAPI application
│   └── monitoring/     # Drift detection & alerting
├── scripts/            # Executable scripts
├── tests/              # Unit, integration, e2e tests
├── infra/              # Docker & Terraform
├── configs/            # Configuration files
├── dashboard/          # Streamlit analytics
└── docs/               # Documentation
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose

### Setup

```bash
# Clone repository
git clone https://github.com/eeshanwaqar/sports_betting.git
cd sports_betting

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run Data Pipeline

```bash
# Load historical data
python src/data/loader.py

# Clean and validate
python src/data/cleaner.py
```

### Train Model

```bash
python scripts/train.py
```

### Make Predictions

```bash
python scripts/predict.py --home "Arsenal" --away "Chelsea"
```

### Start API

```bash
python scripts/run_api.py
# Access: http://localhost:8000/docs
```

### Docker (Full Stack)

```bash
docker-compose -f infra/docker-compose.yml up
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict` | POST | Match prediction |
| `/api/v1/chat` | POST | RAG chat interface |
| `/api/v1/teams` | GET | List all teams |
| `/api/v1/teams/{name}` | GET | Team details |
| `/health` | GET | Health check |

---

## Development

```bash
# Run linting
make lint

# Run tests
make test

# Format code
make format

# Start all services
make docker-up
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Project Summary](docs/PROJECT_SUMMARY.md) | Comprehensive project overview |
| [System Design](docs/architecture/system_design.md) | Architecture documentation |
| [API Design](docs/architecture/api_design.md) | API specifications |
| [Setup Guide](docs/guides/setup.md) | Installation instructions |
| [Development Guide](docs/guides/development.md) | Developer documentation |

---

## Skills Demonstrated

This project showcases:

- **Machine Learning** - Feature engineering, model training, evaluation
- **MLOps** - Experiment tracking, model registry, data versioning
- **NLP/GenAI** - Entity extraction, sentiment analysis, RAG
- **Backend** - REST APIs, async programming, caching
- **DevOps** - Containerization, CI/CD, infrastructure as code
- **Software Engineering** - Testing, type hints, code quality

---

## Data

- **Source:** [Football-Data.co.uk](http://www.football-data.co.uk/)
- **Coverage:** 20 EPL seasons (2000-2020)
- **Matches:** 7,600+
- **Features:** Goals, shots, corners, cards, betting odds

---

## Model Performance

| Metric | Value |
|--------|-------|
| Baseline (random) | 33% |
| Target accuracy | 53-57% |

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Author

**Eeshan Waqar**

- GitHub: [@eeshanwaqar](https://github.com/eeshanwaqar)
