"""
Утилиты для многоязычной обработки текста.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Union
from langdetect import detect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import googletrans
from googletrans import Translator
from pymorphy2 import MorphAnalyzer
import numpy as np

# Инициализация детектора языков с фиксированным seed
DetectorFactory.seed = 0

# Инициализация переводчика
_translator = None

# Морфологические анализаторы
_morph_analyzers = {}


def get_translator() -> Translator:
    """Получение экземпляра переводчика."""
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator


def get_morph_analyzer(language: str = "ru") -> MorphAnalyzer:
    """Получение морфологического анализатора для языка."""
    global _morph_analyzers
    
    if language not in _morph_analyzers:
        if language == "ru":
            _morph_analyzers[language] = MorphAnalyzer()
        else:
            # Для других языков можно использовать другие анализаторы
            _morph_analyzers[language] = None
    
    return _morph_analyzers.get(language)


def detect_language(text: str, fast: bool = False) -> Optional[str]:
    """
    Определение языка текста.
    
    Args:
        text: Текст для анализа
        fast: Быстрый режим (только первый символ)
    
    Returns:
        Код языка или None если не удалось определить
    """
    if not text or len(text.strip()) < 10:
        return None
    
    try:
        if fast:
            # Быстрая проверка по первым символам
            text_sample = text[:100].strip()
            if not text_sample:
                return None
            return detect(text_sample)
        else:
            # Точное определение с вероятностями
            langs = detect_langs(text)
            if langs:
                return langs[0].lang
    except (LangDetectException, Exception):
        pass
    
    return None


def normalize_text(
    text: str,
    language: Optional[str] = None,
    remove_diacritics: bool = False,
    normalize_whitespace: bool = True,
    fix_unicode: bool = True,
) -> str:
    """
    Нормализация текста.
    
    Args:
        text: Исходный текст
        language: Язык текста (для языковых специфичных нормализаций)
        remove_diacritics: Удалять диакритические знаки
        normalize_whitespace: Нормализовать пробелы
        fix_unicode: Исправлять Unicode символы
    
    Returns:
        Нормализованный текст
    """
    if not text:
        return ""
    
    # Исправление Unicode
    if fix_unicode:
        text = unicodedata.normalize("NFKC", text)
    
    # Удаление диакритических знаков
    if remove_diacritics:
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
    
    # Нормализация пробелов
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
    
    # Языковые специфичные нормализации
    if language == "kk":
        # Нормализация казахских символов
        text = text.replace("ә", "Ә").replace("ғ", "Ғ").replace("қ", "Қ")
        text = text.replace("ң", "Ң").replace("ө", "Ө").replace("ұ", "Ұ")
        text = text.replace("ү", "Ү").replace("һ", "Һ").replace("і", "І")
    elif language == "ru":
        # Замена ё на е
        text = text.replace("ё", "е").replace("Ё", "Е")
    
    return text


def split_sentences(text: str, language: str = "kk") -> List[str]:
    """
    Разделение текста на предложения.
    
    Args:
        text: Исходный текст
        language: Язык текста
    
    Returns:
        Список предложений
    """
    # Простое разделение по точкам, вопросительным и восклицательным знакам
    sentence_endings = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_endings, text)
    
    # Фильтрация пустых предложений
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def tokenize_kazakh(text: str) -> List[str]:
    """
    Токенизация казахского текста.
    
    Args:
        text: Казахский текст
    
    Returns:
        Список токенов
    """
    # Простая токенизация по пробелам и знакам препинания
    tokens = re.findall(r'\b[\wӘәҒғҚқҢңӨөҰұҮүҺһІі]+\b', text)
    return tokens


def translate_text(
    text: str,
    target_lang: str = "kk",
    source_lang: Optional[str] = None,
    max_length: int = 5000,
) -> Optional[str]:
    """
    Перевод текста с использованием Google Translate.
    
    Args:
        text: Текст для перевода
        target_lang: Целевой язык
        source_lang: Исходный язык (автоопределение если None)
        max_length: Максимальная длина текста для перевода
    
    Returns:
        Переведенный текст или None при ошибке
    """
    if not text or len(text) > max_length:
        return None
    
    try:
        translator = get_translator()
        
        if source_lang:
            translation = translator.translate(
                text,
                src=source_lang,
                dest=target_lang
            )
        else:
            translation = translator.translate(text, dest=target_lang)
        
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return None


def get_language_info(language_code: str) -> Dict[str, str]:
    """
    Получение информации о языке по коду.
    
    Args:
        language_code: Код языка (kk, ru, en и т.д.)
    
    Returns:
        Словарь с информацией о языке
    """
    languages = {
        "kk": {
            "name": "Kazakh",
            "native_name": "Қазақша",
            "family": "Turkic",
            "script": "Cyrillic",
            "direction": "ltr",
        },
        "ru": {
            "name": "Russian",
            "native_name": "Русский",
            "family": "Indo-European",
            "script": "Cyrillic",
            "direction": "ltr",
        },
        "en": {
            "name": "English",
            "native_name": "English",
            "family": "Indo-European",
            "script": "Latin",
            "direction": "ltr",
        },
    }
    
    return languages.get(language_code, {})


def calculate_perplexity(text: str, language: str = "kk") -> float:
    """
    Расчет перплексии текста (упрощенный).
    
    Args:
        text: Текст для анализа
        language: Язык текста
    
    Returns:
        Значение перплексии
    """
    if not text:
        return float('inf')
    
    # Упрощенный расчет перплексии на основе длины слов
    words = text.split()
    if len(words) < 2:
        return float('inf')
    
    # Средняя длина слова
    avg_word_length = np.mean([len(w) for w in words])
    
    # Эмпирическая формула для перплексии
    perplexity = avg_word_length * 2.0
    
    return perplexity


def create_language_prompt(
    text: str,
    language: str,
    task: str = "qa",
    context: Optional[str] = None,
) -> str:
    """
    Создание промпта для конкретного языка и задачи.
    
    Args:
        text: Основной текст/вопрос
        language: Язык промпта
        task: Тип задачи (qa, summary, translation, etc.)
        context: Контекст для задачи
    
    Returns:
        Сформированный промпт
    """
    prompts = {
        "kk": {
            "qa": f"Сұрақ: {text}\nКонтекст: {context}\nЖауап:",
            "summary": f"Мәтінді қысқартыңыз: {text}\nҚысқартылған мәтін:",
            "translation": f"Аударыңыз: {text}\nАудармасы:",
        },
        "ru": {
            "qa": f"Вопрос: {text}\nКонтекст: {context}\nОтвет:",
            "summary": f"Сократите текст: {text}\nСокращенный текст:",
            "translation": f"Переведите: {text}\nПеревод:",
        },
        "en": {
            "qa": f"Question: {text}\nContext: {context}\nAnswer:",
            "summary": f"Summarize: {text}\nSummary:",
            "translation": f"Translate: {text}\nTranslation:",
        },
    }
    
    lang_prompts = prompts.get(language, prompts["en"])
    return lang_prompts.get(task, lang_prompts["qa"])


def check_text_quality(
    text: str,
    language: str,
    min_length: int = 10,
    max_special_chars: float = 0.3,
) -> Tuple[bool, Dict[str, any]]:
    """
    Проверка качества текста.
    
    Args:
        text: Текст для проверки
        language: Ожидаемый язык
        min_length: Минимальная длина
        max_special_chars: Максимальная доля специальных символов
    
    Returns:
        Кортеж (валиден, метрики)
    """
    if not text:
        return False, {"error": "Empty text"}
    
    metrics = {
        "length": len(text),
        "word_count": len(text.split()),
        "detected_language": detect_language(text),
    }
    
    # Проверка длины
    if len(text) < min_length:
        return False, {**metrics, "error": f"Text too short (<{min_length})"}
    
    # Проверка языка
    detected = detect_language(text)
    if detected and detected != language:
        return False, {**metrics, "error": f"Language mismatch: {detected} != {language}"}
    
    # Проверка специальных символов
    special_chars = len(re.findall(r'[^а-яА-ЯӘәҒғҚқҢңӨөҰұҮүҺһІіa-zA-Z0-9\s.,!?-]', text))
    special_ratio = special_chars / len(text) if text else 0
    metrics["special_char_ratio"] = special_ratio
    
    if special_ratio > max_special_chars:
        return False, {**metrics, "error": f"Too many special chars: {special_ratio:.2f}"}
    
    return True, metrics