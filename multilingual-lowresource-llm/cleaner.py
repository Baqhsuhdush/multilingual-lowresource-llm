"""
Модуль для очистки и фильтрации многоязычных данных.
"""

import re
import string
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
import unicodedata

import numpy as np
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords
import nltk

from src.utils.logger import setup_logger
from src.utils.multilingual import normalize_text

logger = setup_logger(__name__)


@dataclass
class CleaningConfig:
    """Конфигурация очистки данных."""
    
    # Основные параметры
    min_length: int = 50
    max_length: int = 10000
    min_words: int = 5
    max_words: int = 1000
    
    # Фильтрация по языку
    target_languages: List[str] = field(default_factory=lambda: ["kk", "ru", "en"])
    language_threshold: float = 0.8
    
    # Фильтрация качества
    remove_duplicates: bool = True
    deduplication_method: str = "exact"  # exact, fuzzy, semantic
    similarity_threshold: float = 0.9
    
    # Очистка текста
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    remove_dates: bool = True
    remove_special_chars: bool = True
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    lowercase: bool = False
    
    # Фильтрация контента
    remove_low_quality: bool = True
    quality_metrics: List[str] = field(default_factory=lambda: [
        "readability", "repetition", "special_char_ratio"
    ])
    
    # Списки стоп-слов
    custom_stopwords: Dict[str, List[str]] = field(default_factory=lambda: {
        "kk": ["және", "бірақ", "ол", "бұл", "мен", "сен", "сіз", "олар"],
        "ru": ["и", "в", "на", "с", "по", "для", "это", "что", "как"],
        "en": ["the", "and", "in", "to", "of", "a", "is", "that", "for"]
    })


class DataCleaner:
    """Класс для очистки и фильтрации текстовых данных."""
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.stopwords = {}
        self._initialize_stopwords()
    
    def _initialize_stopwords(self):
        """Инициализация стоп-слов."""
        try:
            nltk.download('stopwords', quiet=True)
            
            # Загрузка стандартных стоп-слов
            for lang in self.config.target_languages:
                lang_code = lang[:2]  # Берем первые две буквы
                try:
                    if lang_code == "kk":
                        # Для казахского используем кастомные стоп-слова
                        self.stopwords[lang] = set(self.config.custom_stopwords.get("kk", []))
                    elif lang_code in ["ru", "en"]:
                        self.stopwords[lang] = set(stopwords.words(lang_code))
                        # Добавляем кастомные стоп-слова
                        self.stopwords[lang].update(self.config.custom_stopwords.get(lang, []))
                    else:
                        self.stopwords[lang] = set(self.config.custom_stopwords.get(lang, []))
                except:
                    self.stopwords[lang] = set(self.config.custom_stopwords.get(lang, []))
        except:
            # Если nltk не доступен, используем только кастомные стоп-слова
            for lang in self.config.target_languages:
                self.stopwords[lang] = set(self.config.custom_stopwords.get(lang, []))
    
    def clean_text(self, text: str, language: str = "kk") -> str:
        """
        Очистка одного текста.
        
        Args:
            text: Исходный текст
            language: Язык текста
        
        Returns:
            Очищенный текст
        """
        if not text:
            return ""
        
        # Нормализация Unicode
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Удаление URL
        if self.config.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Удаление email адресов
        if self.config.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Удаление телефонных номеров
        if self.config.remove_phone_numbers:
            phone_patterns = [
                r'\+\d[\d\s\-\(\)]+\d',  # Международный формат
                r'\d[\d\s\-]+\d',         # Локальный формат
            ]
            for pattern in phone_patterns:
                text = re.sub(pattern, '', text)
        
        # Удаление дат
        if self.config.remove_dates:
            date_patterns = [
                r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}',  # DD/MM/YYYY
                r'\d{4}[./-]\d{1,2}[./-]\d{1,2}',    # YYYY/MM/DD
            ]
            for pattern in date_patterns:
                text = re.sub(pattern, '', text)
        
        # Удаление специальных символов
        if self.config.remove_special_chars:
            # Сохраняем буквы, цифры, пробелы и основные знаки препинания
            allowed_chars = string.ascii_letters + string.digits + ' '
            
            # Добавляем специфичные символы для разных языков
            if language == "kk":
                allowed_chars += 'ӘәҒғҚқҢңӨөҰұҮүҺһІі'
            elif language == "ru":
                allowed_chars += 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
            
            allowed_chars += '.,!?-:;"\'()'
            
            # Удаляем все символы не из allowed_chars
            text = ''.join(c for c in text if c in allowed_chars)
        
        # Нормализация пробелов
        if self.config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Приведение к нижнему регистру
        if self.config.lowercase:
            text = text.lower()
        
        # Удаление лишних точек
        text = re.sub(r'\.{2,}', '.', text)
        
        # Удаление пустых скобок и кавычек
        text = re.sub(r'[\[\]\(\)\{\}\"\']{2,}', '', text)
        
        return text
    
    def filter_by_length(self, text: str) -> bool:
        """
        Фильтрация по длине текста.
        
        Args:
            text: Текст для проверки
        
        Returns:
            True если текст проходит фильтрацию
        """
        if not text:
            return False
        
        text_length = len(text)
        word_count = len(text.split())
        
        return (
            self.config.min_length <= text_length <= self.config.max_length and
            self.config.min_words <= word_count <= self.config.max_words
        )
    
    def detect_language(self, text: str) -> Tuple[Optional[str], float]:
        """
        Определение языка текста с уверенностью.
        
        Args:
            text: Текст для определения языка
        
        Returns:
            Кортеж (код языка, уверенность)
        """
        if len(text) < 10:
            return None, 0.0
        
        try:
            from langdetect import detect_langs
            
            languages = detect_langs(text)
            if languages:
                best_lang = languages[0]
                return best_lang.lang, best_lang.prob
        except (LangDetectException, ImportError):
            # Попытка простого определения по символам
            try:
                detected = detect(text)
                return detected, 0.5  # Низкая уверенность
            except:
                pass
        
        return None, 0.0
    
    def filter_by_language(self, text: str, expected_language: str) -> bool:
        """
        Фильтрация по языку.
        
        Args:
            text: Текст для проверки
            expected_language: Ожидаемый язык
        
        Returns:
            True если текст на ожидаемом языке
        """
        detected_lang, confidence = self.detect_language(text)
        
        if detected_lang is None:
            return False
        
        # Проверка совпадения языка и уверенности
        if detected_lang != expected_language:
            return False
        
        return confidence >= self.config.language_threshold
    
    def calculate_quality_score(self, text: str, language: str) -> float:
        """
        Расчет оценки качества текста.
        
        Args:
            text: Текст для оценки
            language: Язык текста
        
        Returns:
            Оценка качества от 0 до 1
        """
        if not text:
            return 0.0
        
        scores = []
        
        # 1. Оценка читаемости (отношение уникальных слов)
        words = text.split()
        if words:
            unique_words = set(words)
            readability_score = len(unique_words) / len(words)
            scores.append(readability_score)
        
        # 2. Оценка повторений
        if len(words) > 10:
            # Проверка повторения последовательностей
            max_repetition = self._calculate_max_repetition(text)
            repetition_score = 1.0 - min(max_repetition, 0.5)  # Максимум 0.5 штраф
            scores.append(repetition_score)
        
        # 3. Оценка по стоп-словам
        if language in self.stopwords and words:
            stopword_count = sum(1 for word in words if word in self.stopwords[language])
            stopword_ratio = stopword_count / len(words)
            stopword_score = 1.0 - min(stopword_ratio, 0.5)  # Максимум 0.5 штраф
            scores.append(stopword_score)
        
        # 4. Оценка по специальным символам
        special_chars = len(re.findall(r'[^а-яА-ЯӘәҒғҚқҢңӨөҰұҮүҺһІіa-zA-Z0-9\s.,!?-]', text))
        special_char_ratio = special_chars / len(text) if text else 0
        char_score = 1.0 - min(special_char_ratio * 2, 0.5)  # Максимум 0.5 штраф
        scores.append(char_score)
        
        # 5. Оценка по структуре предложений
        sentence_count = len(re.findall(r'[.!?]+', text))
        if sentence_count > 0:
            avg_sentence_length = len(words) / sentence_count
            # Идеальная длина предложения: 15-25 слов
            if 15 <= avg_sentence_length <= 25:
                structure_score = 1.0
            else:
                structure_score = max(0.5, 1.0 - abs(avg_sentence_length - 20) / 50)
            scores.append(structure_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_max_repetition(self, text: str, max_pattern_length: int = 10) -> float:
        """
        Расчет максимального повторения в тексте.
        
        Args:
            text: Текст для анализа
            max_pattern_length: Максимальная длина паттерна для поиска
        
        Returns:
            Коэффициент повторения
        """
        max_repetition = 0
        
        for pattern_length in range(1, min(max_pattern_length, len(text) // 2) + 1):
            for i in range(len(text) - pattern_length * 2 + 1):
                pattern = text[i:i + pattern_length]
                count = 1
                
                # Проверяем повторения паттерна
                j = i + pattern_length
                while j + pattern_length <= len(text) and text[j:j + pattern_length] == pattern:
                    count += 1
                    j += pattern_length
                
                if count > 1:
                    repetition_ratio = (pattern_length * count) / len(text)
                    max_repetition = max(max_repetition, repetition_ratio)
        
        return max_repetition
    
    def remove_duplicates(
        self,
        texts: List[str],
        method: str = "exact",
        similarity_threshold: float = 0.9
    ) -> List[str]:
        """
        Удаление дубликатов из списка текстов.
        
        Args:
            texts: Список текстов
            method: Метод дедупликации (exact, fuzzy, semantic)
            similarity_threshold: Порог схожести
        
        Returns:
            Список уникальных текстов
        """
        if method == "exact":
            return list(dict.fromkeys(texts))  # Сохраняем порядок
        
        elif method == "fuzzy":
            # Простая fuzzy дедупликация по префиксам
            unique_texts = []
            seen_prefixes = set()
            
            for text in texts:
                if not text:
                    continue
                
                # Берем префикс текста для сравнения
                prefix = text[:100]  # Первые 100 символов
                is_duplicate = False
                
                for seen_prefix in seen_prefixes:
                    # Вычисляем схожесть префиксов
                    similarity = self._calculate_similarity(prefix, seen_prefix)
                    if similarity > similarity_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_texts.append(text)
                    seen_prefixes.add(prefix)
            
            return unique_texts
        
        else:  # semantic
            # Для семантической дедупликации нужны эмбеддинги
            # Временно используем fuzzy метод
            return self.remove_duplicates(texts, method="fuzzy", similarity_threshold=similarity_threshold)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Расчет схожести двух текстов.
        
        Args:
            text1: Первый текст
            text2: Второй текст
        
        Returns:
            Коэффициент схожести от 0 до 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Используем отношение длины общего префикса к средней длине
        common_prefix = 0
        min_len = min(len(text1), len(text2))
        
        for i in range(min_len):
            if text1[i] == text2[i]:
                common_prefix += 1
            else:
                break
        
        avg_len = (len(text1) + len(text2)) / 2
        return common_prefix / avg_len if avg_len > 0 else 0.0
    
    def process_batch(
        self,
        texts: List[str],
        languages: Optional[List[str]] = None,
        min_quality_score: float = 0.5
    ) -> Tuple[List[str], List[str], Dict[str, any]]:
        """
        Пакетная обработка текстов.
        
        Args:
            texts: Список текстов
            languages: Список языков (если None, определяется автоматически)
            min_quality_score: Минимальная оценка качества
        
        Returns:
            Кортеж (очищенные тексты, языки, статистика)
        """
        cleaned_texts = []
        cleaned_languages = []
        
        stats = {
            "total": len(texts),
            "filtered_by_length": 0,
            "filtered_by_language": 0,
            "filtered_by_quality": 0,
            "final_count": 0,
            "quality_scores": [],
        }
        
        for i, text in enumerate(texts):
            if not text:
                continue
            
            # Определение языка
            if languages and i < len(languages):
                language = languages[i]
            else:
                detected_lang, _ = self.detect_language(text)
                language = detected_lang or "kk"
            
            # Фильтрация по длине
            if not self.filter_by_length(text):
                stats["filtered_by_length"] += 1
                continue
            
            # Фильтрация по языку
            if not self.filter_by_language(text, language):
                stats["filtered_by_language"] += 1
                continue
            
            # Очистка текста
            cleaned_text = self.clean_text(text, language)
            
            # Расчет качества
            quality_score = self.calculate_quality_score(cleaned_text, language)
            stats["quality_scores"].append(quality_score)
            
            # Фильтрация по качеству
            if quality_score < min_quality_score:
                stats["filtered_by_quality"] += 1
                continue
            
            # Сохранение
            cleaned_texts.append(cleaned_text)
            cleaned_languages.append(language)
        
        # Дедупликация
        if self.config.remove_duplicates:
            original_count = len(cleaned_texts)
            cleaned_texts = self.remove_duplicates(
                cleaned_texts,
                method=self.config.deduplication_method,
                similarity_threshold=self.config.similarity_threshold
            )
            stats["duplicates_removed"] = original_count - len(cleaned_texts)
        
        stats["final_count"] = len(cleaned_texts)
        stats["avg_quality_score"] = np.mean(stats["quality_scores"]) if stats["quality_scores"] else 0
        
        return cleaned_texts, cleaned_languages, stats
    
    def process_dataset(
        self,
        dataset: "Dataset",
        text_column: str = "text",
        language_column: str = "language"
    ) -> "Dataset":
        """
        Обработка датасета.
        
        Args:
            dataset: Hugging Face Dataset
            text_column: Название колонки с текстом
            language_column: Название колонки с языком
        
        Returns:
            Очищенный датасет
        """
        from datasets import Dataset
        
        logger.info(f"Начало очистки датасета ({len(dataset)} примеров)")
        
        texts = dataset[text_column]
        languages = dataset[language_column] if language_column in dataset.column_names else None
        
        cleaned_texts, cleaned_languages, stats = self.process_batch(texts, languages)
        
        logger.info(f"Статистика очистки: {stats}")
        
        # Создание нового датасета
        data = {text_column: cleaned_texts}
        if cleaned_languages:
            data[language_column] = cleaned_languages
        
        # Сохранение других колонок
        for column in dataset.column_names:
            if column not in [text_column, language_column]:
                # Фильтруем значения по тем же индексам
                column_values = []
                for i, (text, lang) in enumerate(zip(texts, languages if languages else [None] * len(texts))):
                    if text in cleaned_texts:
                        column_values.append(dataset[i][column])
                data[column] = column_values
        
        return Dataset.from_dict(data)