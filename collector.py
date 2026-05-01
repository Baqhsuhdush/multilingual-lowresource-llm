"""
Модуль для сбора и подготовки многоязычных данных.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import re

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

from src.utils.logger import setup_logger
from src.utils.multilingual import detect_language, normalize_text

logger = setup_logger(__name__)


@dataclass
class DataSourceConfig:
    """Конфигурация источника данных."""
    name: str
    dataset: str
    language: str
    split: str = "train"
    max_samples: int = 10000
    trust_remote_code: bool = True
    streaming: bool = False


@dataclass
class DataCollectorConfig:
    """Конфигурация сборщика данных."""
    languages: List[str] = field(default_factory=lambda: ["kk", "ru", "en"])
    sources: List[DataSourceConfig] = field(default_factory=list)
    
    # Параметры фильтрации
    min_text_length: int = 50
    max_text_length: int = 20000
    min_word_count: int = 10
    max_word_count: int = 1000
    
    # Распределение данных
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Дополнительные параметры
    seed: int = 42
    cache_dir: Optional[str] = None
    force_rebuild: bool = False


class MultilingualDataCollector:
    """Класс для сбора многоязычных данных из различных источников."""
    
    def __init__(self, config: DataCollectorConfig):
        self.config = config
        self.data = {}
        
        # Предопределенные источники для языков
        self.default_sources = {
            "kk": [
                DataSourceConfig(
                    name="oscar_kk",
                    dataset="oscar-corpus/OSCAR-2301",
                    language="kk",
                    max_samples=5000
                ),
                DataSourceConfig(
                    name="cc100_kk",
                    dataset="facebook/cc_100",
                    language="kk",
                    max_samples=5000
                )
            ],
            "ru": [
                DataSourceConfig(
                    name="oscar_ru",
                    dataset="oscar-corpus/OSCAR-2301",
                    language="ru",
                    max_samples=10000
                ),
                DataSourceConfig(
                    name="cc100_ru",
                    dataset="facebook/cc_100",
                    language="ru",
                    max_samples=10000
                )
            ],
            "en": [
                DataSourceConfig(
                    name="oscar_en",
                    dataset="oscar-corpus/OSCAR-2301",
                    language="en",
                    max_samples=5000
                )
            ]
        }
    
    def collect_from_hf(self, source: DataSourceConfig) -> Optional[Dataset]:
        """
        Сбор данных из Hugging Face Datasets.
        
        Args:
            source: Конфигурация источника
        
        Returns:
            Dataset или None в случае ошибки
        """
        try:
            logger.info(f"Загрузка {source.dataset} для языка {source.language}")
            
            # Загрузка датасета
            dataset = load_dataset(
                source.dataset,
                language=source.language,
                split=f"{source.split}[:{source.max_samples}]",
                trust_remote_code=source.trust_remote_code,
                streaming=source.streaming,
                cache_dir=self.config.cache_dir,
            )
            
            if source.streaming:
                # Конвертация streaming датасета в обычный
                data = []
                for i, item in enumerate(dataset):
                    if i >= source.max_samples:
                        break
                    data.append(item)
                dataset = Dataset.from_list(data)
            
            logger.info(f"Загружено {len(dataset)} примеров из {source.dataset}")
            return dataset
            
        except Exception as e:
            logger.error(f"Ошибка загрузки {source.dataset}: {e}")
            return None
    
    def collect_wikipedia(self, language: str, max_samples: int = 1000) -> Optional[Dataset]:
        """
        Сбор данных из Wikipedia.
        
        Args:
            language: Код языка
            max_samples: Максимальное количество примеров
        
        Returns:
            Dataset с данными Wikipedia
        """
        try:
            logger.info(f"Загрузка Wikipedia для языка {language}")
            
            dataset = load_dataset(
                "wikipedia",
                language=language,
                date="20240101",  # Конкретная дата для воспроизводимости
                split=f"train[:{max_samples}]",
                cache_dir=self.config.cache_dir,
            )
            
            # Преобразуем в нужный формат
            data = []
            for item in dataset:
                text = item.get("text", "")
                title = item.get("title", "")
                
                if text:
                    data.append({
                        "text": text,
                        "title": title,
                        "language": language,
                        "source": "wikipedia",
                        "url": f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    })
            
            logger.info(f"Загружено {len(data)} примеров из Wikipedia")
            return Dataset.from_list(data)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки Wikipedia: {e}")
            return None
    
    def collect_news(self, language: str, max_samples: int = 1000) -> Optional[Dataset]:
        """
        Сбор новостных данных.
        
        Args:
            language: Код языка
            max_samples: Максимальное количество примеров
        
        Returns:
            Dataset с новостными данными
        """
        try:
            logger.info(f"Сбор новостей для языка {language}")
            
            # Здесь можно добавить парсинг новостных сайтов
            # Временно возвращаем пустой датасет
            data = []
            
            # Пример для казахских новостей
            if language == "kk":
                news_sources = [
                    "https://www.inform.kz",
                    "https://www.kazinform.kz",
                ]
                
                for source in news_sources:
                    try:
                        response = requests.get(source, timeout=10)
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Простой парсинг заголовков
                        for heading in soup.find_all(['h1', 'h2', 'h3']):
                            text = heading.get_text().strip()
                            if text and len(text) > self.config.min_text_length:
                                data.append({
                                    "text": text,
                                    "title": "",
                                    "language": language,
                                    "source": source,
                                    "url": source
                                })
                    except Exception as e:
                        logger.warning(f"Ошибка парсинга {source}: {e}")
            
            logger.info(f"Собрано {len(data)} новостей для языка {language}")
            return Dataset.from_list(data)
            
        except Exception as e:
            logger.error(f"Ошибка сбора новостей: {e}")
            return None
    
    def load_local_data(self, data_dir: Path) -> Dict[str, List[Dict]]:
        """
        Загрузка локальных данных.
        
        Args:
            data_dir: Директория с данными
        
        Returns:
            Словарь с данными по языкам
        """
        data = {lang: [] for lang in self.config.languages}
        
        for lang in self.config.languages:
            lang_dir = data_dir / "raw" / lang
            if not lang_dir.exists():
                continue
            
            # Чтение текстовых файлов
            for file_path in lang_dir.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        
                        if self._validate_text(text, lang):
                            data[lang].append({
                                "text": text,
                                "title": file_path.stem,
                                "language": lang,
                                "source": "local",
                                "url": str(file_path)
                            })
                except Exception as e:
                    logger.error(f"Ошибка чтения файла {file_path}: {e}")
            
            # Чтение JSONL файлов
            for file_path in lang_dir.glob("*.jsonl"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            try:
                                item = json.loads(line.strip())
                                text = item.get("text", "")
                                
                                if self._validate_text(text, lang):
                                    data[lang].append({
                                        "text": text,
                                        "title": item.get("title", file_path.stem),
                                        "language": lang,
                                        "source": "local_jsonl",
                                        "url": str(file_path),
                                        "metadata": item.get("metadata", {})
                                    })
                            except json.JSONDecodeError as e:
                                logger.error(f"Ошибка JSON в строке {line_num} файла {file_path}: {e}")
                except Exception as e:
                    logger.error(f"Ошибка чтения файла {file_path}: {e}")
        
        return data
    
    def _validate_text(self, text: str, language: str) -> bool:
        """
        Валидация текста.
        
        Args:
            text: Текст для валидации
            language: Ожидаемый язык
        
        Returns:
            True если текст валиден
        """
        if not text or not isinstance(text, str):
            return False
        
        # Проверка длины
        if len(text) < self.config.min_text_length:
            logger.debug(f"Текст слишком короткий: {len(text)} символов")
            return False
        
        if len(text) > self.config.max_text_length:
            logger.debug(f"Текст слишком длинный: {len(text)} символов")
            return False
        
        # Проверка количества слов
        words = text.split()
        if len(words) < self.config.min_word_count:
            logger.debug(f"Слишком мало слов: {len(words)}")
            return False
        
        if len(words) > self.config.max_word_count:
            logger.debug(f"Слишком много слов: {len(words)}")
            return False
        
        # Проверка языка
        detected_lang = detect_language(text)
        if detected_lang and detected_lang != language:
            logger.debug(f"Язык не совпадает: ожидался {language}, получен {detected_lang}")
            return False
        
        # Проверка на специальные символы
        special_char_ratio = len(re.findall(r'[^а-яА-ЯӘәҒғҚқҢңӨөҰұҮүҺһІіa-zA-Z0-9\s.,!?-]', text)) / len(text)
        if special_char_ratio > 0.3:
            logger.debug(f"Слишком много специальных символов: {special_char_ratio:.2f}")
            return False
        
        return True
    
    def _preprocess_text(self, text: str, language: str) -> str:
        """
        Предобработка текста.
        
        Args:
            text: Исходный текст
            language: Язык текста
        
        Returns:
            Предобработанный текст
        """
        # Нормализация
        text = normalize_text(text, language=language)
        
        # Удаление лишних пробелов и переносов строк
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Удаление HTML тегов
        text = re.sub(r'<[^>]+>', '', text)
        
        # Удаление URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Удаление email адресов
        text = re.sub(r'\S+@\S+', '', text)
        
        # Удаление телефонных номеров
        text = re.sub(r'\+\d[\d\s\-\(\)]+\d', '', text)
        
        return text
    
    def collect_all(self, output_dir: Path) -> DatasetDict:
        """
        Сбор всех данных.
        
        Args:
            output_dir: Директория для сохранения результатов
        
        Returns:
            DatasetDict с собранными данными
        """
        logger.info(f"Начало сбора данных для языков: {self.config.languages}")
        
        all_data = []
        
        # Используем источники из конфигурации или дефолтные
        sources = self.config.sources if self.config.sources else []
        
        for lang in self.config.languages:
            logger.info(f"Сбор данных для языка: {lang}")
            
            # Добавляем дефолтные источники если не указаны
            if not sources:
                lang_sources = self.default_sources.get(lang, [])
            else:
                lang_sources = [s for s in sources if s.language == lang]
            
            # Сбор из каждого источника
            for source in lang_sources:
                dataset = self.collect_from_hf(source)
                if dataset:
                    for item in tqdm(dataset, desc=f"Обработка {source.name}"):
                        text = item.get("text", "")
                        
                        if text and self._validate_text(text, lang):
                            # Предобработка
                            text = self._preprocess_text(text, lang)
                            
                            all_data.append({
                                "text": text,
                                "language": lang,
                                "source": source.name,
                                "timestamp": datetime.now().isoformat()
                            })
            
            # Сбор из Wikipedia
            wiki_dataset = self.collect_wikipedia(lang, max_samples=2000)
            if wiki_dataset:
                for item in tqdm(wiki_dataset, desc="Обработка Wikipedia"):
                    text = item.get("text", "")
                    
                    if text and self._validate_text(text, lang):
                        text = self._preprocess_text(text, lang)
                        
                        all_data.append({
                            "text": text,
                            "title": item.get("title", ""),
                            "language": lang,
                            "source": "wikipedia",
                            "url": item.get("url", ""),
                            "timestamp": datetime.now().isoformat()
                        })
            
            # Сбор новостей
            news_dataset = self.collect_news(lang, max_samples=500)
            if news_dataset:
                for item in tqdm(news_dataset, desc="Обработка новостей"):
                    text = item.get("text", "")
                    
                    if text and self._validate_text(text, lang):
                        text = self._preprocess_text(text, lang)
                        
                        all_data.append({
                            "text": text,
                            "title": item.get("title", ""),
                            "language": lang,
                            "source": "news",
                            "url": item.get("url", ""),
                            "timestamp": datetime.now().isoformat()
                        })
        
        # Загрузка локальных данных
        local_data = self.load_local_data(output_dir / "..")
        for lang, items in local_data.items():
            for item in items:
                text = item.get("text", "")
                
                if text and self._validate_text(text, lang):
                    text = self._preprocess_text(text, lang)
                    
                    all_data.append({
                        "text": text,
                        "title": item.get("title", ""),
                        "language": lang,
                        "source": item.get("source", "local"),
                        "url": item.get("url", ""),
                        "metadata": item.get("metadata", {}),
                        "timestamp": datetime.now().isoformat()
                    })
        
        logger.info(f"Всего собрано {len(all_data)} примеров")
        
        # Создание датасета
        if not all_data:
            raise ValueError("Не удалось собрать данные. Проверьте источники.")
        
        df = pd.DataFrame(all_data)
        
        # Сохранение промежуточных результатов
        intermediate_dir = output_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = intermediate_dir / f"collected_data_{timestamp}.parquet"
        df.to_parquet(intermediate_file, index=False)
        logger.info(f"Промежуточные данные сохранены в {intermediate_file}")
        
        # Удаление дубликатов
        df = df.drop_duplicates(subset=["text", "language"])
        logger.info(f"После удаления дубликатов: {len(df)} примеров")
        
        # Создание Dataset
        dataset = Dataset.from_pandas(df)
        
        # Разделение на train/val/test
        dataset = dataset.train_test_split(
            test_size=self.config.val_split + self.config.test_split,
            seed=self.config.seed
        )
        
        val_test = dataset["test"].train_test_split(
            test_size=self.config.test_split / (self.config.val_split + self.config.test_split),
            seed=self.config.seed
        )
        
        dataset_dict = DatasetDict({
            "train": dataset["train"],
            "validation": val_test["train"],
            "test": val_test["test"]
        })
        
        # Сохранение
        save_dir = output_dir / "processed"
        dataset_dict.save_to_disk(str(save_dir))
        
        # Сохранение статистики
        stats = self._calculate_statistics(dataset_dict)
        stats_file = save_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Датасет сохранен в {save_dir}")
        logger.info(f"Статистика сохранена в {stats_file}")
        
        return dataset_dict
    
    def _calculate_statistics(self, dataset_dict: DatasetDict) -> Dict[str, Any]:
        """
        Расчет статистики по датасету.
        
        Args:
            dataset_dict: DatasetDict
        
        Returns:
            Словарь со статистикой
        """
        stats = {}
        
        for split_name, dataset in dataset_dict.items():
            split_stats = {
                "total_samples": len(dataset),
                "languages": {},
                "avg_text_length": 0,
                "avg_word_count": 0,
            }
            
            # Статистика по языкам
            languages = dataset["language"]
            unique_langs = set(languages)
            
            for lang in unique_langs:
                lang_indices = [i for i, l in enumerate(languages) if l == lang]
                lang_texts = [dataset[i]["text"] for i in lang_indices]
                
                split_stats["languages"][lang] = {
                    "count": len(lang_indices),
                    "percentage": len(lang_indices) / len(dataset) * 100,
                    "avg_length": np.mean([len(t) for t in lang_texts]) if lang_texts else 0,
                    "avg_words": np.mean([len(t.split()) for t in lang_texts]) if lang_texts else 0,
                }
            
            # Общая статистика
            texts = dataset["text"]
            split_stats["avg_text_length"] = np.mean([len(t) for t in texts]) if texts else 0
            split_stats["avg_word_count"] = np.mean([len(t.split()) for t in texts]) if texts else 0
            
            stats[split_name] = split_stats
        
        return stats
    
    def create_training_data(self, dataset_dict: DatasetDict, output_file: Path):
        """
        Создание данных для тренировки в формате инструкций.
        
        Args:
            dataset_dict: DatasetDict с данными
            output_file: Файл для сохранения
        """
        logger.info("Создание данных для тренировки в формате инструкций")
        
        training_data = []
        
        for split_name, dataset in dataset_dict.items():
            for item in tqdm(dataset, desc=f"Обработка {split_name}"):
                text = item.get("text", "")
                language = item.get("language", "kk")
                
                if not text:
                    continue
                
                # Создание инструкции в зависимости от языка
                if language == "kk":
                    instruction = f"Мәтін: {text[:200]}...\nБұл мәтін туралы қысқаша мазмұн жазыңыз."
                    output = text[:100] + "..."
                elif language == "ru":
                    instruction = f"Текст: {text[:200]}...\nНапишите краткое содержание этого текста."
                    output = text[:100] + "..."
                else:
                    instruction = f"Text: {text[:200]}...\nWrite a brief summary of this text."
                    output = text[:100] + "..."
                
                training_data.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "language": language,
                    "source": item.get("source", ""),
                })
        
        # Сохранение
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Тренировочные данные сохранены в {output_file} ({len(training_data)} примеров)")