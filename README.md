# EPL Sports Betting Odds Predictor

A production-grade ML system that predicts English Premier League match outcomes and generates fair betting odds.

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
```

---

## Key Features

- **59 Engineered Features** - Elo ratings, multi-window form, H2H records, shooting stats, venue metrics, and more
- **4 Calibrated Models** - XGBoost, Random Forest, Gradient Boosting, Logistic Regression with time-series CV
- **MLflow Integration** - Experiment tracking, model registry, champion/challenger promotion
- **REST API** - Single/batch predictions, team analytics, head-to-head history
- **Cloud Ready** - Docker, Terraform (AWS), GitHub Actions CI/CD

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML/MLOps** | XGBoost, Scikit-learn, MLflow |
| **Backend** | FastAPI, Pydantic, Uvicorn |
| **DevOps** | Docker, Docker Compose, GitHub Actions |
| **Infrastructure** | Terraform, AWS (ECS Fargate, ALB, RDS, S3, ECR, CloudWatch) |
| **Quality** | pytest, Ruff, MyPy |

---

## Project Structure

```
├── src/
│   ├── data/           # Data pipeline (load, clean, validate)
│   ├── features/       # 11 feature engineering modules
│   ├── training/       # Model training, evaluation, MLflow tracking
│   ├── inference/      # Prediction, odds calculation, confidence scoring
│   └── api/            # FastAPI application (routes, schemas, middleware)
├── tests/              # Unit, integration, e2e tests
├── scripts/            # Training, prediction, API launch scripts
├── infra/
│   ├── docker/         # Dockerfiles (API, training, MLflow)
│   ├── terraform/      # AWS infrastructure as code
│   └── docker-compose.yml
└── .github/workflows/  # CI, training, data validation pipelines
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
python src/data/loader.py
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

### Start API + Frontend

```bash
python scripts/run_api.py
# Frontend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Docker (Full Stack)

```bash
docker-compose -f infra/docker-compose.yml up
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single match prediction |
| `/predict/batch` | POST | Batch predictions (up to 20) |
| `/teams` | GET | List all available teams |
| `/teams/{name}` | GET | Team record and recent form |
| `/matches/recent` | GET | Recent match results |
| `/matches/head-to-head` | GET | H2H record between two teams |
| `/health` | GET | Liveness check |
| `/health/ready` | GET | Readiness check (model loaded) |

---

## Feature Engineering

59 features across 11 modules, all computed using only pre-match data (zero leakage):

| Module | Features |
|--------|----------|
| Elo Ratings | Home/away Elo, Elo difference |
| Betting Odds | Implied probabilities, favourite indicator |
| Form (3/5-match) | Points, win rate, goal difference |
| Exponential Form | Decay-weighted recent performance |
| Venue Stats | Home/away win rate, goals, clean sheets |
| Season Stats | Win rate, PPG, goals for, clean sheets |
| Streaks | Win streak, unbeaten streak |
| Shooting | Shot accuracy, shot conversion |
| Head-to-Head | H2H win rate, average goals (10-match window) |
| Differentials | 12 mismatch features (form, attack vs defense, etc.) |
| Temporal | Weekend flag, month, matchweek |

---

## Deploy to AWS

### Prerequisites
- AWS CLI configured with credentials
- Terraform >= 1.5.0
- Docker

### 1. Bootstrap Terraform State

Create an S3 bucket and DynamoDB table for remote state (one-time):

```bash
aws s3 mb s3://epl-predictor-terraform-state --region us-east-1
aws dynamodb create-table \
  --table-name epl-predictor-terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

### 2. Provision Infrastructure

```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

This creates: VPC, ECS Fargate cluster (with Spot), ALB, RDS (PostgreSQL for MLflow), S3 (artifacts), ECR (container registries), CloudWatch (logging), and auto-scaling (1-3 tasks).

### 3. Build and Push Docker Images

```bash
# Get ECR URLs from Terraform output
API_REPO=$(terraform output -raw ecr_api_url)
TRAINING_REPO=$(terraform output -raw ecr_training_url)

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $API_REPO

# Build and push API image
docker build -f infra/docker/api.Dockerfile -t $API_REPO:latest .
docker push $API_REPO:latest

# Build and push training image
docker build -f infra/docker/training.Dockerfile -t $TRAINING_REPO:latest .
docker push $TRAINING_REPO:latest
```

### 4. Train the Model

Trigger training via GitHub Actions (manual workflow dispatch) or run the ECS training task:

```bash
aws ecs run-task \
  --cluster $(terraform output -raw ecs_cluster_name) \
  --task-definition epl-predictor-prod-training \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-id>],securityGroups=[<sg-id>],assignPublicIp=ENABLED}"
```

### 5. Access the Application

```bash
# Get the application URL
terraform output api_url

# Frontend + API are served from the same URL
curl http://<alb-dns>/health
curl -X POST http://<alb-dns>/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Chelsea"}'
```

The frontend is served from the same ALB URL. MLflow UI is at the `/mlflow` path.

### 6. Tear Down

```bash
terraform destroy
```

---

## CI/CD Pipelines

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| **pipeline.yml** | Push to `main`, manual dispatch | Full pipeline: CI → Build → Deploy MLflow → Train → Deploy API + Frontend |
| **ci.yml** | Push to `develop`, PR to `main` | Lint (Ruff), type check (MyPy), test (pytest + coverage) |
| **deploy-manual.yml** | Manual dispatch | Redeploy API + Frontend without retraining |
| **train-manual.yml** | Manual dispatch | Trains model on ECS, logs to MLflow, promotes champion if better |
| **data-validation.yml** | Weekly + manual | Validates data schema, target distribution, nulls, duplicates |

---

## Development

```bash
make lint      # Ruff + MyPy
make test      # pytest with coverage
make format    # Auto-format code
make docker-up # Start all services
```

---

## Data

- **Source:** [Football-Data.co.uk](http://www.football-data.co.uk/)
- **Coverage:** 20 EPL seasons (2000-2020)
- **Matches:** 7,600+
- **Raw features:** Goals, shots, corners, cards, betting odds

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Author

**Eeshan Waqar** - [@eeshanwaqar](https://github.com/eeshanwaqar)
