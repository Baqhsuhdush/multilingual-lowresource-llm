# ============================================
# МНОГОЯЗЫЧНАЯ LLM - Makefile
# ============================================

# Конфигурация
# ============================================
PYTHON := python3
PIP := pip3
PROJECT_NAME := multilingual-lowresource-llm
PYTHONPATH := $(shell pwd)
SRC_DIR := src
TESTS_DIR := tests
EXPERIMENTS_DIR := experiments
DEMO_DIR := demo
DOCKER_COMPOSE := docker-compose
DOCKER := docker
VENV := venv

# Цвета для вывода
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m # No Color

# Утилиты
ECHO := echo -e
MKDIR := mkdir -p
RM := rm -rf
CP := cp
FIND := find

# Помощь
# ============================================
.PHONY: help
help: ## Показать эту справку
	@$(ECHO) "${CYAN}Доступные команды:${NC}"
	@$(ECHO) ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "${YELLOW}%-30s${NC} %s\n", $$1, $$2}'

# Установка и настройка
# ============================================
.PHONY: install setup download-data download-models

install: ## Установить зависимости
	@$(ECHO) "${GREEN}Установка зависимостей...${NC}"
	$(PIP) install -e ".[dev]"
	@$(ECHO) "${GREEN}Зависимости установлены!${NC}"

setup: ## Настроить проект
	@$(ECHO) "${GREEN}Настройка проекта...${NC}"
	$(MKDIR) data/{raw/{kk,ru},processed,embeddings,faiss_indexes,cache}
	$(MKDIR) models/{pretrained,finetuned,adapters}
	$(MKDIR) logs mlruns
	$(CP) .env.example .env
	@$(ECHO) "${YELLOW}Отредактируйте .env файл перед использованием${NC}"
	@$(ECHO) "${GREEN}Проект настроен!${NC}"

download-data: ## Скачать примеры данных
	@$(ECHO) "${GREEN}Скачивание примеров данных...${NC}"
	$(PYTHON) scripts/download_sample_data.py
	@$(ECHO) "${GREEN}Данные скачаны!${NC}"

download-models: ## Скачать предобученные модели
	@$(ECHO) "${GREEN}Скачивание моделей...${NC}"
	$(PYTHON) scripts/download_models.py
	@$(ECHO) "${GREEN}Модели скачаны!${NC}"

# Тренировка
# ============================================
.PHONY: train qlora-train rag-train

train: ## Полная тренировка (данные + модель + RAG)
	@$(ECHO) "${GREEN}Запуск полного пайплайна тренировки...${NC}"
	$(MAKE) preprocess-data
	$(MAKE) qlora-train
	$(MAKE) build-index

qlora-train: ## QLoRA fine-tuning
	@$(ECHO) "${GREEN}QLoRA fine-tuning...${NC}"
	$(PYTHON) -m src.models.qlora_trainer \
		--config config/training_config.yaml \
		--output_dir models/finetuned

rag-train: ## Тренировка RAG системы
	@$(ECHO) "${GREEN}Тренировка RAG системы...${NC}"
	$(PYTHON) -m src.rag.trainer \
		--documents data/processed/documents \
		--output data/faiss_indexes

# Обработка данных
# ============================================
.PHONY: preprocess-data clean-data augment-data

preprocess-data: ## Предобработка данных
	@$(ECHO) "${GREEN}Предобработка данных...${NC}"
	$(PYTHON) -m src.data_processing.collector
	$(PYTHON) -m src.data_processing.cleaner
	@$(ECHO) "${GREEN}Данные обработаны!${NC}"

clean-data: ## Очистка данных
	@$(ECHO) "${GREEN}Очистка данных...${NC}"
	$(PYTHON) -m src.data_processing.cleaner --mode deep
	@$(ECHO) "${GREEN}Данные очищены!${NC}"

augment-data: ## Аугментация данных
	@$(ECHO) "${GREEN}Аугментация данных...${NC}"
	$(PYTHON) -m src.data_processing.augmenter \
		--input data/processed/train \
		--output data/processed/train_augmented \
		--factor 3
	@$(ECHO) "${GREEN}Данные аугментированы!${NC}"

# RAG система
# ============================================
.PHONY: build-index test-retrieval update-index

build-index: ## Построить FAISS индекс
	@$(ECHO) "${GREEN}Построение FAISS индекса...${NC}"
	$(PYTHON) -m src.rag.vector_store \
		--documents data/processed/documents \
		--index_path data/faiss_indexes \
		--embedding_model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
	@$(ECHO) "${GREEN}Индекс построен!${NC}"

test-retrieval: ## Тестирование поиска
	@$(ECHO) "${GREEN}Тестирование поиска...${NC}"
	$(PYTHON) scripts/test_retrieval.py \
		--index data/faiss_indexes \
		--queries data/processed/test_queries.json

update-index: ## Обновить индекс
	@$(ECHO) "${GREEN}Обновление индекса...${NC}"
	$(PYTHON) -m src.rag.vector_store \
		--documents data/processed/new_documents \
		--index_path data/faiss_indexes \
		--mode update

# Тестирование
# ============================================
.PHONY: test test-unit test-integration test-coverage lint format

test: ## Запустить все тесты
	@$(ECHO) "${GREEN}Запуск всех тестов...${NC}"
	pytest $(TESTS_DIR) -v --tb=short

test-unit: ## Запустить юнит-тесты
	@$(ECHO) "${GREEN}Запуск юнит-тестов...${NC}"
	pytest $(TESTS_DIR)/unit -v

test-integration: ## Запустить интеграционные тесты
	@$(ECHO) "${GREEN}Запуск интеграционных тестов...${NC}"
	pytest $(TESTS_DIR)/integration -v

test-coverage: ## Запустить тесты с покрытием
	@$(ECHO) "${GREEN}Запуск тестов с покрытием...${NC}"
	pytest $(TESTS_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term

lint: ## Проверить код стиля
	@$(ECHO) "${GREEN}Проверка стиля кода...${NC}"
	black --check $(SRC_DIR) $(TESTS_DIR) $(DEMO_DIR)
	flake8 $(SRC_DIR) $(TESTS_DIR) $(DEMO_DIR)
	mypy $(SRC_DIR)

format: ## Форматировать код
	@$(ECHO) "${GREEN}Форматирование кода...${NC}"
	black $(SRC_DIR) $(TESTS_DIR) $(DEMO_DIR)
	isort $(SRC_DIR) $(TESTS_DIR) $(DEMO_DIR)

# Запуск сервисов
# ============================================
.PHONY: api demo docker-up docker-down docker-logs

api: ## Запустить API сервер
	@$(ECHO) "${GREEN}Запуск API сервера...${NC}"
	uvicorn src.api.server:app \
		--host $(shell grep -E '^API_HOST=' .env | cut -d'=' -f2) \
		--port $(shell grep -E '^API_PORT=' .env | cut -d'=' -f2) \
		--reload \
		--log-level info

demo: ## Запустить демо интерфейс
	@$(ECHO) "${GREEN}Запуск демо интерфейса...${NC}"
	cd $(DEMO_DIR) && streamlit run app.py \
		--server.port $(shell grep -E '^DEMO_PORT=' ../.env | cut -d'=' -f2) \
		--server.address 0.0.0.0

docker-up: ## Запустить все сервисы через Docker
	@$(ECHO) "${GREEN}Запуск Docker сервисов...${NC}"
	$(DOCKER_COMPOSE) up -d --build
	@$(ECHO) "${GREEN}Сервисы запущены!${NC}"
	@$(ECHO) "${CYAN}API: http://localhost:8000${NC}"
	@$(ECHO) "${CYAN}Демо: http://localhost:8501${NC}"
	@$(ECHO) "${CYAN}MLflow: http://localhost:5000${NC}"
	@$(ECHO) "${CYAN}Grafana: http://localhost:3000${NC}"

docker-down: ## Остановить Docker сервисы
	@$(ECHO) "${GREEN}Остановка Docker сервисов...${NC}"
	$(DOCKER_COMPOSE) down --remove-orphans

docker-logs: ## Показать логи Docker
	@$(ECHO) "${GREEN}Логи Docker сервисов...${NC}"
	$(DOCKER_COMPOSE) logs -f --tail=100

# Модели и инференс
# ============================================
.PHONY: infer evaluate benchmark

infer: ## Запустить инференс
	@$(ECHO) "${GREEN}Запуск инференса...${NC}"
	$(PYTHON) -m src.models.inference \
		--model models/finetuned \
		--prompt "Қазақстан туралы" \
		--language kk

evaluate: ## Оценка модели
	@$(ECHO) "${GREEN}Оценка модели...${NC}"
	$(PYTHON) -m src.models.evaluator \
		--model models/finetuned \
		--dataset data/processed/test \
		--metrics bleu rouge perplexity

benchmark: ## Бенчмарк производительности
	@$(ECHO) "${GREEN}Запуск бенчмарков...${NC}"
	$(PYTHON) -m src.utils.benchmark \
		--model models/finetuned \
		--batch_sizes 1 4 8 16 \
		--sequence_lengths 128 256 512

# Мониторинг и логирование
# ============================================
.PHONY: monitor logs metrics

monitor: ## Открыть панель мониторинга
	@$(ECHO) "${GREEN}Открытие панели мониторинга...${NC}"
	open http://localhost:3000 || xdg-open http://localhost:3000 || start http://localhost:3000

logs: ## Показать логи
	@$(ECHO) "${GREEN}Просмотр логов...${NC}"
	tail -f logs/multilingual_llm.log

metrics: ## Показать метрики Prometheus
	@$(ECHO) "${GREEN}Открытие метрик Prometheus...${NC}"
	open http://localhost:9090 || xdg-open http://localhost:9090 || start http://localhost:9090

# Очистка
# ============================================
.PHONY: clean clean-pyc clean-build clean-cache clean-all

clean: ## Очистить временные файлы
	@$(ECHO) "${GREEN}Очистка временных файлов...${NC}"
	$(FIND) . -type f -name "*.pyc" -delete
	$(FIND) . -type d -name "__pycache__" -delete
	$(FIND) . -type d -name "*.egg-info" -exec $(RM) {} +
	$(FIND) . -type d -name "*.pytest_cache" -exec $(RM) {} +
	$(FIND) . -type d -name ".mypy_cache" -exec $(RM) {} +
	$(FIND) . -type d -name ".coverage" -delete
	$(FIND) . -type f -name ".coverage.*" -delete

clean-build: ## Очистить build файлы
	@$(ECHO) "${GREEN}Очистка build файлов...${NC}"
	$(RM) build/
	$(RM) dist/
	$(RM) *.egg-info/

clean-cache: ## Очистить кэш
	@$(ECHO) "${GREEN}Очистка кэша...${NC}"
	$(RM) .cache/
	$(RM) .pytest_cache/
	$(RM) .mypy_cache/
	$(RM) mlruns/
	$(RM) wandb/

clean-all: ## Полная очистка
	@$(ECHO) "${RED}Полная очистка проекта...${NC}"
	$(MAKE) clean
	$(MAKE) clean-build
	$(MAKE) clean-cache
	$(RM) models/finetuned/*
	$(RM) data/processed/*
	$(RM) data/embeddings/*
	$(RM) data/faiss_indexes/*
	$(RM) logs/*
	@$(ECHO) "${GREEN}Проект полностью очищен!${NC}"

# Утилиты
# ============================================
.PHONY: notebook shell export-model

notebook: ## Запустить Jupyter notebook
	@$(ECHO) "${GREEN}Запуск Jupyter notebook...${NC}"
	jupyter notebook $(EXPERIMENTS_DIR)/notebooks \
		--ip=0.0.0.0 \
		--port=8888 \
		--no-browser \
		--allow-root

shell: ## Открыть Python shell с окружением проекта
	@$(ECHO) "${GREEN}Открытие Python shell...${NC}"
	$(PYTHON) -m IPython --no-banner

export-model: ## Экспортировать модель для production
	@$(ECHO) "${GREEN}Экспорт модели...${NC}"
	$(PYTHON) -m src.models.export \
		--input models/finetuned \
		--output models/exported \
		--format onnx

# Документация
# ============================================
.PHONY: docs docs-build docs-serve

docs: ## Создать документацию
	@$(ECHO) "${GREEN}Создание документации...${NC}"
	cd docs && make html

docs-build: ## Сборка документации
	@$(ECHO) "${GREEN}Сборка документации...${NC}"
	sphinx-build -b html docs/source docs/build

docs-serve: ## Запустить сервер документации
	@$(ECHO) "${GREEN}Запуск сервера документации...${NC}"
	python -m http.server 8001 --directory docs/build

# Git и deployment
# ============================================
.PHONY: git-status git-commit git-push deploy

git-status: ## Статус Git
	@$(ECHO) "${GREEN}Статус Git...${NC}"
	git status

git-commit: ## Сделать коммит (использовать: make git-commit m="Сообщение")
	@$(ECHO) "${GREEN}Создание коммита...${NC}"
	git add .
	git commit -m "$(m)"
	@$(ECHO) "${GREEN}Коммит создан!${NC}"

git-push: ## Отправить изменения
	@$(ECHO) "${GREEN}Отправка изменений...${NC}"
	git push origin main

deploy: ## Деплой на Hugging Face Spaces
	@$(ECHO) "${GREEN}Деплой на Hugging Face Spaces...${NC}"
	$(PYTHON) scripts/deploy_to_hf.py \
		--model models/finetuned \
		--repo_id your-username/multilingual-llm

# Специальные цели
# ============================================
.PHONY: all check pre-commit

all: ## Выполнить все проверки и тесты
	@$(ECHO) "${GREEN}Выполнение всех проверок...${NC}"
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) format

check: ## Проверить готовность проекта к коммиту
	@$(ECHO) "${GREEN}Проверка готовности к коммиту...${NC}"
	$(MAKE) lint
	$(MAKE) test
	@$(ECHO) "${GREEN}Проект готов к коммиту!${NC}"

pre-commit: ## Запустить pre-commit хуки
	@$(ECHO) "${GREEN}Запуск pre-commit хуков...${NC}"
	pre-commit run --all-files

# ============================================
# Конец Makefile
# ============================================