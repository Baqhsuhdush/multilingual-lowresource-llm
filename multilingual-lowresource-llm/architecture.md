# Архитектура проекта

## Обзор

Проект представляет собой многоязычную языковую модель для низкоресурсных языков (казахский и русский) с системой RAG (Retrieval-Augmented Generation).

## Компоненты

### 1. Сбор и обработка данных (`src/data_processing/`)
- **collector.py**: Сбор данных из различных источников (Hugging Face, Wikipedia, локальные файлы)
- **cleaner.py**: Очистка и фильтрация текстовых данных
- **augmenter.py**: Аугментация данных для увеличения размера датасета
- **tokenizer.py**: Многоязычная токенизация

### 2. Модели (`src/models/`)
- **base_model.py**: Базовый класс для работы с языковыми моделями
- **qlora_trainer.py**: QLoRA fine-tuning для эффективного обучения
- **model_selector.py**: Выбор и загрузка моделей
- **evaluator.py**: Оценка моделей

### 3. RAG система (`src/rag/`)
- **vector_store.py**: Векторное хранилище на основе FAISS
- **retriever.py**: Многоязычный ретривер с гибридным поиском
- **reranker.py**: Реранкинг результатов поиска
- **query_processor.py**: Обработка поисковых запросов

### 4. Утилиты (`src/utils/`)
- **logger.py**: Настройка логирования
- **metrics.py**: Метрики для оценки моделей
- **multilingual.py**: Утилиты для работы с многоязычным текстом

### 5. API (`src/api/`)
- **server.py**: FastAPI сервер
- **schemas.py**: Pydantic схемы для API
- **endpoints.py**: Эндпоинты API

## Поток данных

1. **Сбор данных** → Очистка → Аугментация → Токенизация
2. **Тренировка модели** → QLoRA fine-tuning → Оценка → Сохранение
3. **Индексация документов** → Создание FAISS индекса → Сохранение метаданных
4. **Поиск и генерация** → Обработка запроса → RAG поиск → Генерация ответа

## Технологии

- **Языковые модели**: Qwen2.5, Transformer-based models
- **Fine-tuning**: QLoRA, PEFT, Hugging Face Transformers
- **Векторный поиск**: FAISS, Sentence Transformers
- **API**: FastAPI, Uvicorn
- **Демо интерфейс**: Streamlit
- **Контейнеризация**: Docker, Docker Compose
- **Мониторинг**: MLflow, WandB, Prometheus, Grafana

## Структура данных
