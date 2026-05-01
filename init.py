"""
Многоязычная LLM для низкоресурсных языков.

Этот пакет предоставляет инструменты для разработки и использования
многоязычных языковых моделей для казахского и русского языков.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from src.utils.logger import setup_logger

# Инициализация логгера по умолчанию
logger = setup_logger(__name__)

# Экспорт основных компонентов
__all__ = [
    "logger",
    "data_processing",
    "models",
    "rag",
    "utils",
    "api",
]