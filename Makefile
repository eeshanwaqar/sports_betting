# Makefile for EPL Predictor

.PHONY: install lint test format clean docker-up docker-down

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Run linting
lint:
	ruff check src/ tests/
	mypy src/

# Run tests
test:
	pytest tests/ -v

# Format code
format:
	ruff format src/ tests/

# Clean cache files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Docker commands
docker-build:
	docker-compose -f infra/docker-compose.yml build

docker-up:
	docker-compose -f infra/docker-compose.yml up -d

docker-down:
	docker-compose -f infra/docker-compose.yml down

docker-logs:
	docker-compose -f infra/docker-compose.yml logs -f

# Data pipeline
data-load:
	python src/data/loader.py

data-clean:
	python src/data/cleaner.py

# Training
train:
	python scripts/train.py

# API
api:
	python scripts/run_api.py
