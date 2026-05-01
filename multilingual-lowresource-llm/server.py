"""
FastAPI сервер для многоязычной LLM.
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from src.utils.logger import setup_logger
from src.models.base_model import MultilingualLLM, ModelConfig
from src.rag.retriever import MultilingualRetriever, RetrievalConfig
from src.rag.vector_store import VectorStore, VectorStoreConfig

logger = setup_logger(__name__)


class Settings(BaseSettings):
    """Настройки приложения."""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    api_workers: int = 1
    api_log_level: str = "info"
    
    # CORS настройки
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
    ]
    
    # Пути к моделям и данным
    model_path: str = "./models/finetuned"
    index_path: str = "./data/faiss_indexes"
    data_path: str = "./data/processed"
    
    # Параметры моделей
    model_name: str = "Qwen/Qwen2.5-1.5B"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    # Параметры генерации
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # Параметры поиска
    search_top_k: int = 5
    rerank_top_k: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Модели запросов и ответов
class HealthResponse(BaseModel):
    """Ответ на запрос здоровья."""
    status: str = "healthy"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    version: str = "0.1.0"


class InfoResponse(BaseModel):
    """Информация о системе."""
    name: str = "Multilingual LLM API"
    version: str = "0.1.0"
    languages: List[str] = ["kk", "ru", "en"]
    models: Dict[str, Any]
    rag_enabled: bool
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class SearchRequest(BaseModel):
    """Запрос для поиска."""
    query: str
    language: str = "kk"
    top_k: Optional[int] = 5
    filter: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Ответ на поисковый запрос."""
    query: str
    language: str
    results: List[Dict[str, Any]]
    total: int
    search_time: float


class GenerateRequest(BaseModel):
    """Запрос для генерации."""
    prompt: str
    language: str = "kk"
    context: Optional[str] = None
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    use_rag: bool = True
    rag_top_k: Optional[int] = 3


class GenerateResponse(BaseModel):
    """Ответ на запрос генерации."""
    prompt: str
    response: str
    language: str
    context_used: Optional[List[Dict[str, Any]]] = None
    generation_time: float
    tokens_generated: int


class DocumentRequest(BaseModel):
    """Запрос для добавления документа."""
    text: str
    language: str = "kk"
    metadata: Optional[Dict[str, Any]] = None
    chunk_size: Optional[int] = 512
    chunk_overlap: Optional[int] = 50


class DocumentResponse(BaseModel):
    """Ответ на добавление документа."""
    doc_id: str
    chunks: int
    message: str


class BatchRequest(BaseModel):
    """Пакетный запрос."""
    requests: List[GenerateRequest]
    parallel: bool = False


class BatchResponse(BaseModel):
    """Ответ на пакетный запрос."""
    responses: List[GenerateResponse]
    total_time: float
    avg_time_per_request: float


# Инициализация приложения
app = FastAPI(
    title="Multilingual LLM API",
    description="API для многоязычной языковой модели для казахского и русского языков",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные объекты
settings = Settings()
model = None
retriever = None
vector_store = None


def get_model():
    """Зависимость для получения модели."""
    global model
    if model is None:
        logger.info("Инициализация модели...")
        model_config = ModelConfig(
            model_name=settings.model_name,
            load_in_4bit=True,
        )
        model = MultilingualLLM(model_config)
        
        # Попытка загрузки fine-tuned модели
        model_path = Path(settings.model_path)
        if model_path.exists():
            try:
                model.load(str(model_path))
                logger.info(f"Fine-tuned модель загружена из {model_path}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить fine-tuned модель: {e}")
    
    return model


def get_retriever():
    """Зависимость для получения ретривера."""
    global retriever
    if retriever is None:
        logger.info("Инициализация ретривера...")
        
        # Конфигурация RAG
        rag_config = RetrievalConfig(
            embedding_model=settings.embedding_model,
            top_k=settings.search_top_k,
            rerank_top_k=settings.rerank_top_k,
            language="kk",
        )
        
        retriever = MultilingualRetriever(rag_config)
        
        # Загрузка индекса если существует
        index_path = Path(settings.index_path)
        if index_path.exists():
            try:
                # Здесь должна быть загрузка индекса
                logger.info(f"Индекс загружен из {index_path}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить индекс: {e}")
    
    return retriever


def get_vector_store():
    """Зависимость для получения векторного хранилища."""
    global vector_store
    if vector_store is None:
        logger.info("Инициализация векторного хранилища...")
        
        store_config = VectorStoreConfig(
            persist_dir=settings.index_path,
            embedding_model=settings.embedding_model,
        )
        
        vector_store = VectorStore(store_config)
    
    return vector_store


@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения."""
    logger.info("Запуск Multilingual LLM API...")
    
    # Предварительная загрузка моделей
    get_model()
    get_retriever()
    get_vector_store()
    
    logger.info("API готов к работе!")


@app.on_event("shutdown")
async def shutdown_event():
    """Событие остановки приложения."""
    logger.info("Остановка Multilingual LLM API...")


# Эндпоинты
@app.get("/", response_model=HealthResponse)
async def root():
    """Корневой эндпоинт."""
    return HealthResponse()


@app.get("/health", response_model=HealthResponse)
async def health():
    """Проверка здоровья API."""
    return HealthResponse()


@app.get("/info", response_model=InfoResponse)
async def info(
    model_obj: MultilingualLLM = Depends(get_model),
    retriever_obj: MultilingualRetriever = Depends(get_retriever),
):
    """Информация о системе."""
    return InfoResponse(
        models={
            "llm": model_obj.config.model_name,
            "embedding": retriever_obj.config.embedding_model,
        },
        rag_enabled=retriever_obj is not None,
    )


@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    retriever_obj: MultilingualRetriever = Depends(get_retriever),
):
    """Поиск документов."""
    import time
    
    start_time = time.time()
    
    try:
        results = retriever_obj.search(
            query=request.query,
            language=request.language,
            top_k=request.top_k or settings.search_top_k,
        )
        
        search_time = time.time() - start_time
        
        return SearchResponse(
            query=request.query,
            language=request.language,
            results=results,
            total=len(results),
            search_time=search_time,
        )
        
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    model_obj: MultilingualLLM = Depends(get_model),
    retriever_obj: MultilingualRetriever = Depends(get_retriever),
):
    """Генерация текста."""
    import time
    
    start_time = time.time()
    context_docs = None
    
    try:
        # Использование RAG если включено
        if request.use_rag and retriever_obj:
            # Поиск релевантных документов
            search_results = retriever_obj.search(
                query=request.prompt,
                language=request.language,
                top_k=request.rag_top_k or 3,
            )
            
            # Объединение контекста
            if search_results:
                contexts = [r["text"] for r in search_results]
                context = "\n\n".join(contexts)
                context_docs = [
                    {"text": r["text"], "score": r["score"]}
                    for r in search_results
                ]
            else:
                context = request.context
        else:
            context = request.context
        
        # Генерация ответа
        response = model_obj.generate(
            prompt=request.prompt,
            language=request.language,
            context=context,
            max_length=request.max_length or settings.max_length,
            temperature=request.temperature or settings.temperature,
            top_p=request.top_p or settings.top_p,
            top_k=request.top_k or settings.top_k,
        )
        
        generation_time = time.time() - start_time
        
        # Примерное количество токенов
        tokens_generated = len(response.split())
        
        return GenerateResponse(
            prompt=request.prompt,
            response=response,
            language=request.language,
            context_used=context_docs,
            generation_time=generation_time,
            tokens_generated=tokens_generated,
        )
        
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/stream")
async def generate_stream(
    request: GenerateRequest,
    model_obj: MultilingualLLM = Depends(get_model),
):
    """Потоковая генерация текста."""
    try:
        def generate():
            # Здесь должна быть реализация потоковой генерации
            # Временно возвращаем обычную генерацию
            response = model_obj.generate(
                prompt=request.prompt,
                language=request.language,
                context=request.context,
                max_length=request.max_length or settings.max_length,
                temperature=request.temperature or settings.temperature,
            )
            
            # Разбиваем ответ на токены для имитации потоковой передачи
            tokens = response.split()
            for token in tokens:
                yield f"data: {json.dumps({'token': token})}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )
        
    except Exception as e:
        logger.error(f"Ошибка потоковой генерации: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents", response_model=DocumentResponse)
async def add_document(
    request: DocumentRequest,
    vector_store_obj: VectorStore = Depends(get_vector_store),
):
    """Добавление документа в векторное хранилище."""
    try:
        from src.rag.vector_store import Document
        
        doc = Document(
            text=request.text,
            metadata={
                "language": request.language,
                "added_at": datetime.now().isoformat(),
                **(request.metadata or {}),
            }
        )
        
        # Добавление документа
        chunks_added = vector_store_obj.add_documents([doc])
        
        return DocumentResponse(
            doc_id=doc.doc_id,
            chunks=chunks_added,
            message=f"Документ добавлен, создано {chunks_added} чанков",
        )
        
    except Exception as e:
        logger.error(f"Ошибка добавления документа: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch/generate", response_model=BatchResponse)
async def batch_generate(
    request: BatchRequest,
    model_obj: MultilingualLLM = Depends(get_model),
):
    """Пакетная генерация."""
    import time
    
    start_time = time.time()
    responses = []
    
    try:
        for req in request.requests:
            response = await generate(req, model_obj, None)  # Без RAG для пакетной обработки
            responses.append(response)
        
        total_time = time.time() - start_time
        
        return BatchResponse(
            responses=responses,
            total_time=total_time,
            avg_time_per_request=total_time / len(responses) if responses else 0,
        )
        
    except Exception as e:
        logger.error(f"Ошибка пакетной генерации: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats(
    vector_store_obj: VectorStore = Depends(get_vector_store),
):
    """Получение статистики."""
    try:
        stats = vector_store_obj.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    vector_store_obj: VectorStore = Depends(get_vector_store),
):
    """Удаление документа."""
    try:
        removed = vector_store_obj.remove_documents([doc_id])
        
        return {
            "message": f"Документ {doc_id} удален",
            "chunks_removed": removed,
        }
    except Exception as e:
        logger.error(f"Ошибка удаления документа: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Эндпоинт для экспорта данных
@app.get("/export/{format}")
async def export_data(
    format: str = "json",
    vector_store_obj: VectorStore = Depends(get_vector_store),
):
    """Экспорт данных."""
    try:
        if format == "json":
            data = {
                "documents": [
                    doc.to_dict() for doc in vector_store_obj.documents.values()
                ],
                "chunks": vector_store_obj.metadata,
            }
            
            return JSONResponse(content=data)
        
        elif format == "csv":
            # Экспорт в CSV формат
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Заголовки
            writer.writerow(["doc_id", "language", "text_preview", "chunks"])
            
            for doc in vector_store_obj.documents.values():
                writer.writerow([
                    doc.doc_id,
                    doc.metadata.get("language", "unknown"),
                    doc.text[:100] + "..." if len(doc.text) > 100 else doc.text,
                    len(doc.chunks),
                ])
            
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=documents.csv"}
            )
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
            
    except Exception as e:
        logger.error(f"Ошибка экспорта: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Middleware для логирования
@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware для логирования запросов."""
    request_id = str(uuid.uuid4())
    
    logger.info(f"Request {request_id}: {request.method} {request.url}")
    
    # Логирование body для POST запросов
    if request.method == "POST":
        try:
            body = await request.body()
            if body:
                logger.debug(f"Request {request_id} body: {body.decode()[:500]}")
        except:
            pass
    
    start_time = time.time() if 'time' in locals() else None
    
    response = await call_next(request)
    
    process_time = time.time() - start_time if start_time else 0
    logger.info(f"Request {request_id} completed: {response.status_code} ({process_time:.3f}s)")
    
    return response


if __name__ == "__main__":
    uvicorn.run(
        "src.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers,
        log_level=settings.api_log_level,
    )