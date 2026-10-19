# Makefile for credit default prediction project
# Lệnh tắt: make setup / train / test / serve / app

.PHONY: setup train figures dvc-repro dvc-dag test serve app lint format docker-up docker-down docker-logs docker-build docker-clean

# Setup: install dependencies
setup:
	uv sync

# Install pre-commit hooks (run separately if needed)
hooks:
	pre-commit install

# Train: run the full training pipeline
train:
	uv run scripts/run_pipeline.py

# Figures: generate visual evaluation and SHAP figures
figures:
	uv run python scripts/generate_figures.py

# DVC: reproduce pipeline and view DAG
dvc-repro:
	dvc repro

dvc-dag:
	dvc dag

# Test: run pytest
test:
	uv run pytest tests/ -v

# Serve: run FastAPI server
serve:
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# App: run Streamlit frontend
app:
	uv run streamlit run app/streamlit_app.py --server.port 8501

# Lint: run ruff
lint:
	ruff check .

# Format: format code with ruff
format:
	ruff check --fix .

# Docker compose up (build and start all services)
docker-up:
	docker-compose up --build -d

# Docker compose up (foreground, see logs)
docker-up-logs:
	docker-compose up --build

# Docker compose down (stop and remove containers)
docker-down:
	docker-compose down

# Docker compose down with volumes (clean data)
docker-down-volumes:
	docker-compose down -v

# View docker logs
docker-logs:
	docker-compose logs -f

# View specific service logs
docker-logs-backend:
	docker-compose logs -f backend

docker-logs-frontend:
	docker-compose logs -f frontend

docker-logs-mlflow:
	docker-compose logs -f mlflow

# Build docker images
docker-build:
	docker-compose build

# Build specific service
docker-build-backend:
	docker-compose build backend

docker-build-frontend:
	docker-compose build frontend

# Clean docker (remove containers, networks, images)
docker-clean:
	docker-compose down -v --rmi all --remove-orphans

# Run tests in docker
docker-test:
	docker-compose run --rm backend uv run pytest tests/ -v

# Open MLflow UI in browser
mlflow-ui:
	@echo "Opening MLflow UI at http://localhost:5000"
	@start http://localhost:5000 2>/dev/null || xdg-open http://localhost:5000 2>/dev/null || open http://localhost:5000

# Open Streamlit app in browser
app-ui:
	@echo "Opening Streamlit app at http://localhost:8501"
	@start http://localhost:8501 2>/dev/null || xdg-open http://localhost:8501 2>/dev/null || open http://localhost:8501

# Open API docs in browser
api-docs:
	@echo "Opening API docs at http://localhost:8000/docs"
	@start http://localhost:8000/docs 2>/dev/null || xdg-open http://localhost:8000/docs 2>/dev/null || open http://localhost:8000/docs