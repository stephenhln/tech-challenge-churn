.PHONY: help install data train test api lint clean docker-build docker-run

help:
	@echo "install      - instala dependências de desenvolvimento"
	@echo "data         - baixa o dataset bruto"
	@echo "train        - treina os modelos e salva o campeão"
	@echo "test         - roda a suíte de testes com pytest"
	@echo "api          - sobe a API localmente com reload"
	@echo "lint         - checa o estilo do código com ruff"
	@echo "clean        - remove caches e artefatos temporários"

install:
	pip install -e ".[dev]"

data:
	python -m churn.data --download

train:
	python -m churn.train

test:
	pytest

api:
	uvicorn churn.api.main:app --reload --host 0.0.0.0 --port 8000

lint:
	ruff check src tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov

docker-build:
	docker build -t churn-api:latest .

docker-run:
	docker run --rm -p 8000:8000 churn-api:latest
