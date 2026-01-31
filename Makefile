.PHONY: help install test test-integration lint format db-up-qdrant db-up-milvus db-down clean

help:
	@echo "Available targets:"
	@echo "  install          - Install dependencies with uv"
	@echo "  test             - Run unit tests"
	@echo "  test-integration - Run integration tests (requires Docker)"
	@echo "  lint             - Run linting (ruff + mypy)"
	@echo "  format           - Auto-format code with ruff"
	@echo "  db-up-qdrant     - Start Qdrant service"
	@echo "  db-up-milvus     - Start Milvus service (with etcd + minio)"
	@echo "  db-down          - Stop all database services"
	@echo "  clean            - Remove cache and build artifacts"

install:
	uv sync --all-extras

test:
	uv run pytest tests/ -m "not integration" -v

test-integration:
	uv run pytest tests/ -m integration -v

lint:
	uv run ruff check src/ tests/
	uv run mypy src/ragmark/

format:
	uv run ruff format src/ tests/

db-up-qdrant:
	cd docker && docker compose --profile qdrant up -d
	@echo "Waiting for Qdrant to be healthy..."
	@sleep 3
	@echo "Qdrant available at http://localhost:6333"

db-up-milvus:
	cd docker && docker compose --profile milvus up -d
	@echo "Waiting for Milvus to be healthy..."
	@sleep 10
	@echo "Milvus available at localhost:19530"

db-down:
	cd docker && docker compose --profile qdrant --profile milvus down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
