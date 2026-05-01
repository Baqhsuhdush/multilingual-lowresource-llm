#!/usr/bin/env python3
"""
Основной скрипт для запуска проекта.
"""

import argparse
import sys
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description="Многоязычная LLM для низкоресурсных языков")
    
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")
    
    # Команда setup
    setup_parser = subparsers.add_parser("setup", help="Настройка проекта")
    setup_parser.add_argument("--force", action="store_true", help="Принудительная настройка")
    
    # Команда train
    train_parser = subparsers.add_parser("train", help="Тренировка модели")
    train_parser.add_argument("--config", default="./config/training_config.yaml", help="Конфигурационный файл")
    train_parser.add_argument("--output", default="./models/finetuned", help="Выходная директория")
    
    # Команда serve
    serve_parser = subparsers.add_parser("serve", help="Запуск API сервера")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Хост")
    serve_parser.add_argument("--port", type=int, default=8000, help="Порт")
    serve_parser.add_argument("--reload", action="store_true", help="Автоперезагрузка")
    
    # Команда demo
    demo_parser = subparsers.add_parser("demo", help="Запуск демо")
    demo_parser.add_argument("--port", type=int, default=8501, help="Порт Streamlit")
    
    # Команда test
    test_parser = subparsers.add_parser("test", help="Запуск тестов")
    test_parser.add_argument("--coverage", action="store_true", help="С измерением покрытия")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        from scripts.setup_project import setup_project
        setup_project(force=args.force)
    
    elif args.command == "train":
        from src.models.qlora_trainer import MultilingualQLoRATrainer
        import yaml
        
        with open(args.config, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Здесь должна быть загрузка конфигурации и запуск тренировки
        logger.info(f"Начало тренировки с конфигурацией из {args.config}")
        # TODO: Реализовать тренировку
    
    elif args.command == "serve":
        import uvicorn
        from src.api.server import app
        
        logger.info(f"Запуск API сервера на {args.host}:{args.port}")
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    
    elif args.command == "demo":
        import subprocess
        
        logger.info(f"Запуск демо на порту {args.port}")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "demo/app.py",
            "--server.port", str(args.port),
            "--server.address", "0.0.0.0"
        ])
    
    elif args.command == "test":
        import pytest
        import subprocess
        
        test_args = ["pytest", "tests/", "-v"]
        if args.coverage:
            test_args.extend(["--cov=src", "--cov-report=html"])
        
        result = subprocess.run(test_args)
        sys.exit(result.returncode)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()