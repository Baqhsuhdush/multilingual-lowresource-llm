# Документация API

## Базовый URL

## Эндпоинты

### 1. Проверка здоровья

```http
GET /health{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "0.1.0"
}GET /info{
  "name": "Multilingual LLM API",
  "version": "0.1.0",
  "languages": ["kk", "ru", "en"],
  "models": {
    "llm": "Qwen/Qwen2.5-1.5B",
    "embedding": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  },
  "rag_enabled": true,
  "timestamp": "2024-01-01T12:00:00Z"
}
POST /search

{
  "query": "Қазақстан туралы",
  "language": "kk",
  "top_k": 5,
  "filter": {
    "source": "wikipedia"
  }
}
{
  "query": "Қазақстан туралы",
  "language": "kk",
  "results": [
    {
      "score": 0.876,
      "text": "Қазақстан - әлемдегі ең үлкен мемлекеттердің бірі...",
      "metadata": {
        "source": "wikipedia",
        "language": "kk"
      }
    }
  ],
  "total": 1,
  "search_time": 0.123
}
POST /generate

{
  "prompt": "Қазақстан туралы не білесіз?",
  "language": "kk",
  "use_rag": true,
  "rag_top_k": 3,
  "max_length": 200,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50
}
{
  "prompt": "Қазақстан туралы не білесіз?",
  "response": "Қазақстан - әлемдегі ең үлкен мемлекеттердің бірі...",
  "language": "kk",
  "context_used": [
    {
      "text": "Қазақстан - Орталық Азиядағы мемлекет...",
      "score": 0.876
    }
  ],
  "generation_time": 1.234,
  "tokens_generated": 45
}
POST /generate/stream

POST /documents

{
  "text": "Қазақстан туралы мәтін...",
  "language": "kk",
  "metadata": {
    "source": "custom",
    "category": "geography"
  },
  "chunk_size": 512,
  "chunk_overlap": 50
}

{
  "doc_id": "abc123def456",
  "chunks": 3,
  "message": "Документ добавлен, создано 3 чанков"
}
DELETE /documents/{doc_id}

GET /stats

POST /batch/generate

{
  "requests": [
    {
      "prompt": "Қазақстан туралы",
      "language": "kk"
    },
    {
      "prompt": "О Казахстане",
      "language": "ru"
    }
  ],
  "parallel": false
}

GET /export/{format}


## Финальная структура проекта

Вот полная структура проекта, который  создал:

multilingual-lowresource-llm/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── Dockerfile
├── Dockerfile.demo
├── docker-compose.yml
├── Makefile
├── run.py
├── setup.sh
├── setup.bat
├── config/
│ ├── init.py
│ ├── model_config.yaml
│ ├── training_config.yaml
│ └── rag_config.yaml
├── data/
│ ├── raw/
│ │ ├── kk/
│ │ └── ru/
│ ├── processed/
│ ├── embeddings/
│ └── faiss_indexes/
├── src/
│ ├── init.py
│ ├── cli.py
│ ├── data_processing/
│ │ ├── init.py
│ │ ├── collector.py
│ │ ├── cleaner.py
│ │ ├── augmenter.py
│ │ └── tokenizer.py
│ ├── models/
│ │ ├── init.py
│ │ ├── base_model.py
│ │ ├── qlora_trainer.py
│ │ ├── model_selector.py
│ │ └── evaluator.py
│ ├── rag/
│ │ ├── init.py
│ │ ├── vector_store.py
│ │ ├── retriever.py
│ │ ├── reranker.py
│ │ └── query_processor.py
│ ├── utils/
│ │ ├── init.py
│ │ ├── logger.py
│ │ ├── metrics.py
│ │ └── multilingual.py
│ └── api/
│ ├── init.py
│ ├── server.py
│ ├── schemas.py
│ └── endpoints.py
├── experiments/
│ ├── notebooks/
│ │ ├── 01_data_exploration.ipynb
│ │ ├── 02_model_training.ipynb
│ │ └── 03_rag_evaluation.ipynb
│ └── scripts/
│ ├── train_qlora.sh
│ └── build_index.sh
├── tests/
│ ├── init.py
│ ├── test_multilingual.py
│ ├── test_data_processing.py
│ ├── test_models.py
│ └── test_rag.py
├── demo/
│ ├── app.py
│ ├── components/
│ │ ├── init.py
│ │ └── chat_interface.py
│ └── static/
│ └── styles.css
├── docs/
│ ├── architecture.md
│ ├── api_documentation.md
│ └── deployment_guide.md
├── scripts/
│ ├── setup_project.py
│ ├── download_sample_data.py
│ ├── quick_test.py
│ └── deploy_to_hf.py
├── monitoring/
│ ├── prometheus.yml
│ └── grafana/
├── nginx/
│ ├── nginx.conf
│ └── conf.d/
├── ssl/
└── logs/


## Как использовать проект

1. **Клонировать и настроить:**
   ```bash
   git clone <repository-url>
   cd multilingual-lowresource-llm
   
   # Linux/Mac
   ./setup.sh
   
   # Windows
   setup.bat

   nano .env  # или любой текстовый редактор

   python run.py setup
python -m src.data_processing.collector

python run.py train --config config/training_config.yaml

python -m src.rag.vector_store --documents data/processed/documents

python run.py serve --host 0.0.0.0 --port 8000

python run.py demo --port 8501

docker-compose up -d
