"""
Модуль для конфигурации логирования.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger as loguru_logger
import json
import datetime


class InterceptHandler(logging.Handler):
    """Перехватчик логов для совместимости с loguru."""
    
    def emit(self, record):
        # Получаем соответствующий уровень loguru
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Находием фрейм для перехвата
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        loguru_logger.opt(
            depth=depth,
            exception=record.exc_info
        ).log(level, record.getMessage())


def setup_logger(
    name: str = "multilingual_llm",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    format: str = "json",
    serialize: bool = False,
) -> loguru_logger:
    """
    Настройка логгера для проекта.
    
    Args:
        name: Имя логгера
        level: Уровень логирования
        log_file: Путь к файлу логов
        rotation: Политика ротации логов
        retention: Политика хранения логов
        format: Формат логов (json, plain)
        serialize: Сериализовать ли логи в JSON
    
    Returns:
        Сконфигурированный логгер
    """
    # Удаляем стандартный обработчик
    loguru_logger.remove()
    
    # Формат для консоли
    if format == "json":
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    else:
        console_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
    
    # Добавляем консольный обработчик
    loguru_logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True,
        enqueue=True,
    )
    
    # Добавляем файловый обработчик если указан файл
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json" or serialize:
            loguru_logger.add(
                str(log_file),
                rotation=rotation,
                retention=retention,
                level=level,
                format="{message}",
                serialize=serialize,
                enqueue=True,
                compression="zip",
            )
        else:
            loguru_logger.add(
                str(log_file),
                rotation=rotation,
                retention=retention,
                level=level,
                format=console_format,
                enqueue=True,
                compression="zip",
            )
    
    # Перехватываем стандартное логирование
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Отключаем логирование от внешних библиотек если нужно
    for lib in ["transformers", "datasets", "torch", "faiss"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    return loguru_logger.bind(name=name)


def get_structured_log(
    message: str,
    level: str,
    module: str,
    function: str,
    line: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Создание структурированного лога.
    
    Args:
        message: Сообщение лога
        level: Уровень логирования
        module: Имя модуля
        function: Имя функции
        line: Номер строки
        extra: Дополнительные поля
    
    Returns:
        Структурированный лог
    """
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "level": level,
        "module": module,
        "function": function,
        "line": line,
        "message": message,
        "pid": sys.platform,
        "hostname": sys.platform,
    }
    
    if extra:
        log_entry.update(extra)
    
    return log_entry


class MetricsLogger:
    """Логгер для метрик."""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("metrics")
        self.metrics = {}
    
    def log_metric(self, name: str, value: float, step: int = None):
        """Логирование метрики."""
        metric_entry = {
            "metric_name": name,
            "value": value,
            "step": step,
        }
        
        self.metrics[name] = value
        
        if self.logger:
            self.logger.info(
                f"Metric: {name} = {value}",
                extra={"metric": metric_entry}
            )
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Логирование нескольких метрик."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def get_metrics(self) -> Dict[str, float]:
        """Получение всех метрик."""
        return self.metrics.copy()


# Глобальный логгер
logger = setup_logger()