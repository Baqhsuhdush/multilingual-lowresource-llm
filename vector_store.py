"""
Модуль для создания и управления векторным хранилищем FAISS.
"""

import pickle
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.preprocessing import normalize

from src.utils.logger import setup_logger
from src.utils.multilingual import normalize_text

logger = setup_logger(__name__)


@dataclass
class VectorStoreConfig:
    """Конфигурация векторного хранилища."""
    
    # Модель эмбеддингов
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_dim: int = 768
    cache_embeddings: bool = True
    
    # Параметры FAISS
    index_type: str = "IndexFlatIP"  # IndexFlatIP, IndexHNSWFlat, IndexIVFFlat
    metric_type: str = "inner_product"  # inner_product, L2
    nprobe: int = 10
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128
    
    # Параметры чанкинга
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_method: str = "recursive"  # recursive, sliding, sentence
    
    # Параметры хранения
    persist_dir: str = "./data/faiss_indexes"
    index_name: str = "multilingual_index"
    metadata_file: str = "metadata.json"
    
    # Производительность
    batch_size: int = 32
    use_gpu: bool = True
    gpu_id: int = 0
    normalize_embeddings: bool = True
    
    # Индексация
    train_size: int = 10000
    rebuild_threshold: int = 1000


class Document:
    """Класс для представления документа."""
    
    def __init__(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ):
        self.text = text
        self.metadata = metadata or {}
        self.doc_id = doc_id or self._generate_id()
        self.embedding: Optional[np.ndarray] = None
        self.chunks: List["DocumentChunk"] = []
    
    def _generate_id(self) -> str:
        """Генерация уникального ID для документа."""
        content = f"{self.text}_{json.dumps(self.metadata, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_chunk(self, chunk: "DocumentChunk"):
        """Добавление чанка к документу."""
        self.chunks.append(chunk)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация документа в словарь."""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "chunk_count": len(self.chunks),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Создание документа из словаря."""
        return cls(
            text=data["text"],
            metadata=data.get("metadata", {}),
            doc_id=data.get("doc_id"),
        )


class DocumentChunk:
    """Класс для представления чанка документа."""
    
    def __init__(
        self,
        text: str,
        doc_id: str,
        chunk_id: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.metadata = metadata or {}
        self.embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация чанка в словарь."""
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Создание чанка из словаря."""
        return cls(
            text=data["text"],
            doc_id=data["doc_id"],
            chunk_id=data["chunk_id"],
            metadata=data.get("metadata", {}),
        )


class VectorStore:
    """Векторное хранилище на основе FAISS."""
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self.embedding_model = None
        self.index = None
        self.metadata = []
        self.documents: Dict[str, Document] = {}
        self.chunks: List[DocumentChunk] = []
        
        # Инициализация модели эмбеддингов
        self._init_embedding_model()
        
        # Создание/загрузка индекса
        self._init_index()
    
    def _init_embedding_model(self):
        """Инициализация модели эмбеддингов."""
        logger.info(f"Загрузка модели эмбеддингов: {self.config.embedding_model}")
        
        try:
            self.embedding_model = SentenceTransformer(
                self.config.embedding_model,
                device="cuda" if self.config.use_gpu else "cpu",
                cache_folder="./models/embeddings"
            )
            
            # Тестирование модели
            test_embedding = self.embedding_model.encode(
                ["test"],
                show_progress_bar=False,
                normalize_embeddings=self.config.normalize_embeddings
            )
            
            self.config.embedding_dim = test_embedding.shape[1]
            logger.info(f"Модель загружена, размерность эмбеддингов: {self.config.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели эмбеддингов: {e}")
            raise
    
    def _init_index(self):
        """Инициализация FAISS индекса."""
        index_path = Path(self.config.persist_dir) / f"{self.config.index_name}.faiss"
        metadata_path = Path(self.config.persist_dir) / self.config.metadata_file
        
        # Попытка загрузки существующего индекса
        if index_path.exists() and metadata_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.metadata = metadata.get("chunks", [])
                self.documents = {
                    doc_id: Document.from_dict(doc_data)
                    for doc_id, doc_data in metadata.get("documents", {}).items()
                }
                
                # Восстановление чанков
                self.chunks = []
                for chunk_data in self.metadata:
                    chunk = DocumentChunk.from_dict(chunk_data)
                    self.chunks.append(chunk)
                
                logger.info(f"Индекс загружен из {index_path}")
                logger.info(f"Документов: {len(self.documents)}, Чанков: {len(self.chunks)}")
                return
                
            except Exception as e:
                logger.error(f"Ошибка загрузки индекса: {e}")
        
        # Создание нового индекса
        logger.info("Создание нового FAISS индекса")
        self._create_new_index()
    
    def _create_new_index(self):
        """Создание нового FAISS индекса."""
        dimension = self.config.embedding_dim
        
        if self.config.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)
        elif self.config.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(dimension)
        elif self.config.index_type == "IndexHNSWFlat":
            self.index = faiss.IndexHNSWFlat(dimension, self.config.hnsw_m)
            self.index.hnsw.efConstruction = self.config.hnsw_ef_construction
            self.index.hnsw.efSearch = self.config.hnsw_ef_search
        elif self.config.index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                dimension,
                min(256, self.config.train_size // 39),
                faiss.METRIC_L2
            )
        else:
            raise ValueError(f"Неизвестный тип индекса: {self.config.index_type}")
        
        # Настройка GPU если доступно
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, self.config.gpu_id, self.index)
                logger.info(f"Индекс перенесен на GPU {self.config.gpu_id}")
            except Exception as e:
                logger.warning(f"Не удалось перенести индекс на GPU: {e}")
        
        logger.info(f"Создан индекс типа {self.config.index_type}")
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        Разбиение документа на чанки.
        
        Args:
            document: Документ для чанкирования
        
        Returns:
            Список чанков документа
        """
        text = document.text
        chunks = []
        
        if self.config.chunking_method == "recursive":
            # Рекурсивное разбиение по разделителям
            separators = ["\n\n", "\n", ". ", "? ", "! ", " "]
            
            def recursive_split(t: str, separators_list: List[str]) -> List[str]:
                if len(t) <= self.config.chunk_size:
                    return [t]
                
                for sep in separators_list:
                    if sep in t:
                        parts = t.split(sep)
                        if len(parts) > 1:
                            # Объединяем части с разделителем
                            results = []
                            current_chunk = ""
                            
                            for part in parts:
                                if len(current_chunk) + len(part) + len(sep) <= self.config.chunk_size:
                                    if current_chunk:
                                        current_chunk += sep + part
                                    else:
                                        current_chunk = part
                                else:
                                    if current_chunk:
                                        results.append(current_chunk)
                                    current_chunk = part
                            
                            if current_chunk:
                                results.append(current_chunk)
                            
                            if len(results) > 1:
                                return results
                
                # Если не нашли подходящего разделителя, разбиваем по длине
                return [t[i:i + self.config.chunk_size] 
                       for i in range(0, len(t), self.config.chunk_size - self.config.chunk_overlap)]
            
            chunk_texts = recursive_split(text, separators)
            
        elif self.config.chunking_method == "sliding":
            # Скользящее окно
            chunk_texts = []
            start = 0
            
            while start < len(text):
                end = min(start + self.config.chunk_size, len(text))
                chunk_texts.append(text[start:end])
                start = end - self.config.chunk_overlap
        
        else:  # sentence
            # Разбиение по предложениям
            import re
            sentences = re.split(r'[.!?]+', text)
            chunk_texts = []
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if len(current_chunk) + len(sentence) + 1 <= self.config.chunk_size:
                    if current_chunk:
                        current_chunk += ". " + sentence
                    else:
                        current_chunk = sentence
                else:
                    if current_chunk:
                        chunk_texts.append(current_chunk)
                    current_chunk = sentence
            
            if current_chunk:
                chunk_texts.append(current_chunk)
        
        # Создание объектов чанков
        for i, chunk_text in enumerate(chunk_texts):
            chunk = DocumentChunk(
                text=chunk_text.strip(),
                doc_id=document.doc_id,
                chunk_id=i,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunk_texts),
                }
            )
            chunks.append(chunk)
            document.add_chunk(chunk)
        
        return chunks
    
    def add_documents(
        self,
        documents: List[Union[Document, Dict[str, Any], str]],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> int:
        """
        Добавление документов в векторное хранилище.
        
        Args:
            documents: Список документов
            batch_size: Размер пакета для обработки
            show_progress: Показывать прогресс-бар
        
        Returns:
            Количество добавленных чанков
        """
        if not documents:
            return 0
        
        # Конвертация документов в объекты Document
        document_objects = []
        for doc in documents:
            if isinstance(doc, Document):
                document_objects.append(doc)
            elif isinstance(doc, dict):
                document_objects.append(Document(
                    text=doc.get("text", ""),
                    metadata=doc.get("metadata", {}),
                    doc_id=doc.get("doc_id"),
                ))
            elif isinstance(doc, str):
                document_objects.append(Document(text=doc))
            else:
                raise TypeError(f"Неизвестный тип документа: {type(doc)}")
        
        # Чанкирование документов
        all_chunks = []
        for doc in tqdm(document_objects, desc="Чанкирование", disable=not show_progress):
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("Не удалось создать чанки из документов")
            return 0
        
        # Получение эмбеддингов
        chunk_texts = [chunk.text for chunk in all_chunks]
        embeddings = self._get_embeddings_batch(chunk_texts, batch_size, show_progress)
        
        # Сохранение эмбеддингов в чанках
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding
        
        # Добавление в индекс
        self._add_to_index(all_chunks, embeddings)
        
        # Сохранение документов и метаданных
        for doc in document_objects:
            self.documents[doc.doc_id] = doc
        
        self.chunks.extend(all_chunks)
        
        logger.info(f"Добавлено {len(document_objects)} документов, {len(all_chunks)} чанков")
        
        # Сохранение индекса
        self.save()
        
        return len(all_chunks)
    
    def _get_embeddings_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Получение эмбеддингов для списка текстов.
        
        Args:
            texts: Список текстов
            batch_size: Размер пакета
            show_progress: Показывать прогресс-бар
        
        Returns:
            Массив эмбеддингов
        """
        batch_size = batch_size or self.config.batch_size
        
        embeddings = []
        
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Получение эмбеддингов",
            disable=not show_progress or len(texts) <= batch_size
        ):
            batch_texts = texts[i:i + batch_size]
            
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=False,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True,
            )
            
            embeddings.append(batch_embeddings)
        
        if embeddings:
            return np.vstack(embeddings)
        else:
            return np.array([])
    
    def _add_to_index(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """
        Добавление эмбеддингов в индекс.
        
        Args:
            chunks: Список чанков
            embeddings: Массив эмбеддингов
        """
        if len(embeddings) == 0:
            return
        
        # Преобразование в float32 для FAISS
        embeddings_f32 = embeddings.astype('float32')
        
        # Добавление в индекс
        if self.index.ntotal == 0 and hasattr(self.index, 'train') and self.config.train_size > 0:
            # Обучение индекса если требуется
            train_size = min(self.config.train_size, len(embeddings_f32))
            self.index.train(embeddings_f32[:train_size])
        
        self.index.add(embeddings_f32)
        
        # Сохранение метаданных
        for chunk in chunks:
            self.metadata.append(chunk.to_dict())
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Поиск похожих документов.
        
        Args:
            query: Поисковый запрос
            k: Количество результатов
            filter_dict: Фильтр по метаданным
            score_threshold: Порог схожести
        
        Returns:
            Список результатов поиска
        """
        if self.index.ntotal == 0:
            logger.warning("Индекс пуст")
            return []
        
        # Получение эмбеддинга запроса
        query_embedding = self.embedding_model.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )[0].astype('float32')
        
        # Поиск в индексе
        if self.config.index_type == "IndexIVFFlat" and self.index.is_trained:
            self.index.nprobe = self.config.nprobe
        
        k = min(k, self.index.ntotal)
        
        # Для inner_product ищем максимальные значения
        if self.config.metric_type == "inner_product":
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
            scores = distances[0]
        else:  # L2 distance
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
            scores = 1 / (1 + distances[0])  # Конвертация расстояния в схожесть
        
        # Фильтрация и форматирование результатов
        results = []
        for idx, score in zip(indices[0], scores):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            # Применение порога схожести
            if score_threshold is not None and score < score_threshold:
                continue
            
            metadata = self.metadata[idx]
            chunk = DocumentChunk.from_dict(metadata)
            
            # Применение фильтров по метаданным
            if filter_dict:
                skip = False
                for key, value in filter_dict.items():
                    if key not in chunk.metadata or chunk.metadata[key] != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # Получение полного документа
            doc = self.documents.get(chunk.doc_id)
            
            results.append({
                "score": float(score),
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "metadata": chunk.metadata,
                "document": doc.to_dict() if doc else None,
            })
        
        return results
    
    def search_batch(
        self,
        queries: List[str],
        k: int = 5,
        batch_size: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Пакетный поиск.
        
        Args:
            queries: Список запросов
            k: Количество результатов на запрос
            batch_size: Размер пакета
        
        Returns:
            Список результатов для каждого запроса
        """
        batch_size = batch_size or self.config.batch_size
        
        all_results = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            
            # Получение эмбеддингов для пакета запросов
            query_embeddings = self.embedding_model.encode(
                batch_queries,
                show_progress_bar=False,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True,
            ).astype('float32')
            
            # Поиск в индексе
            k_actual = min(k, self.index.ntotal)
            
            if self.config.index_type == "IndexIVFFlat" and self.index.is_trained:
                self.index.nprobe = self.config.nprobe
            
            distances, indices = self.index.search(query_embeddings, k_actual)
            
            # Обработка результатов для каждого запроса
            for query_idx, query in enumerate(batch_queries):
                query_results = []
                
                for result_idx, (idx, distance) in enumerate(
                    zip(indices[query_idx], distances[query_idx])
                ):
                    if idx < 0 or idx >= len(self.metadata):
                        continue
                    
                    metadata = self.metadata[idx]
                    chunk = DocumentChunk.from_dict(metadata)
                    
                    # Расчет схожести
                    if self.config.metric_type == "inner_product":
                        score = distance
                    else:
                        score = 1 / (1 + distance)
                    
                    doc = self.documents.get(chunk.doc_id)
                    
                    query_results.append({
                        "score": float(score),
                        "text": chunk.text,
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "metadata": chunk.metadata,
                        "document": doc.to_dict() if doc else None,
                    })
                
                all_results.append(query_results)
        
        return all_results
    
    def save(self):
        """Сохранение индекса и метаданных."""
        save_dir = Path(self.config.persist_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохранение FAISS индекса
        index_path = save_dir / f"{self.config.index_name}.faiss"
        
        # Если индекс на GPU, переносим на CPU для сохранения
        if hasattr(self.index, 'index') and hasattr(self.index.index, 'ntotal'):
            # Это GPU индекс, сохраняем CPU версию
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self.index, str(index_path))
        
        # Сохранение метаданных
        metadata_path = save_dir / self.config.metadata_file
        metadata = {
            "config": self.config.__dict__,
            "documents": {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
            "chunks": self.metadata,
            "stats": {
                "total_documents": len(self.documents),
                "total_chunks": len(self.chunks),
                "index_size": self.index.ntotal,
                "embedding_dim": self.config.embedding_dim,
                "last_updated": datetime.now().isoformat(),
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Индекс сохранен в {index_path}")
        logger.info(f"Метаданные сохранены в {metadata_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики хранилища."""
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "index_size": self.index.ntotal,
            "embedding_dim": self.config.embedding_dim,
            "config": self.config.__dict__,
        }
    
    def clear(self):
        """Очистка хранилища."""
        self._create_new_index()
        self.documents.clear()
        self.chunks.clear()
        self.metadata.clear()
        logger.info("Хранилище очищено")
    
    def remove_documents(self, doc_ids: List[str]) -> int:
        """
        Удаление документов из хранилища.
        
        Args:
            doc_ids: Список ID документов для удаления
        
        Returns:
            Количество удаленных чанков
        """
        # Находим чанки для удаления
        chunks_to_remove = []
        for i, chunk in enumerate(self.chunks):
            if chunk.doc_id in doc_ids:
                chunks_to_remove.append(i)
        
        # Удаляем документы
        for doc_id in doc_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
        
        # Удаляем чанки (в обратном порядке чтобы индексы не сдвигались)
        for i in sorted(chunks_to_remove, reverse=True):
            del self.chunks[i]
            del self.metadata[i]
        
        # Перестраиваем индекс
        self._rebuild_index_from_chunks()
        
        logger.info(f"Удалено {len(doc_ids)} документов, {len(chunks_to_remove)} чанков")
        return len(chunks_to_remove)
    
    def _rebuild_index_from_chunks(self):
        """Перестроение индекса из существующих чанков."""
        if not self.chunks:
            self._create_new_index()
            return
        
        # Собираем эмбеддинги из чанков
        embeddings = []
        valid_chunks = []
        
        for chunk in self.chunks:
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
                valid_chunks.append(chunk)
        
        if not embeddings:
            logger.warning("Нет эмбеддингов для перестроения индекса")
            self._create_new_index()
            return
        
        # Создаем новый индекс
        self._create_new_index()
        
        # Добавляем эмбеддинги
        embeddings_array = np.vstack(embeddings).astype('float32')
        self.index.add(embeddings_array)
        
        # Обновляем метаданные
        self.metadata = [chunk.to_dict() for chunk in valid_chunks]
        
        logger.info(f"Индекс перестроен из {len(valid_chunks)} чанков")