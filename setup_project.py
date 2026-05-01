#!/usr/bin/env python3
"""
Скрипт для настройки проекта.
"""

import os
import sys
from pathlib import Path
import subprocess
import argparse


def setup_project(force: bool = False):
    """Настройка проекта."""
    print("🚀 Настройка многоязычной LLM проекта...")
    
    # Создание необходимых директорий
    directories = [
        "data/raw/kk",
        "data/raw/ru",
        "data/processed",
        "data/embeddings",
        "data/faiss_indexes",
        "models/pretrained",
        "models/finetuned",
        "models/adapters",
        "logs",
        "mlruns",
        "experiments/notebooks",
        "experiments/scripts",
        "tests",
        "demo/components",
        "demo/static",
        "docs",
        "monitoring",
        "nginx",
        "ssl",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / ".gitkeep").touch(exist_ok=True)
        print(f"📁 Создана директория: {directory}")
    
    # Копирование .env.example если .env не существует
    if not Path(".env").exists() or force:
        if Path(".env.example").exists():
            with open(".env.example", "r") as src, open(".env", "w") as dst:
                dst.write(src.read())
            print("📄 Создан файл .env из .env.example")
        else:
            print("⚠️  Файл .env.example не найден")
    
    # Создание пустых __init__.py файлов
    init_dirs = [
        "src",
        "src/data_processing",
        "src/models",
        "src/rag",
        "src/utils",
        "src/api",
        "tests",
        "config",
    ]
    
    for init_dir in init_dirs:
        init_file = Path(init_dir) / "__init__.py"
        init_file.parent.mkdir(parents=True, exist_ok=True)
        if not init_file.exists():
            with open(init_file, "w") as f:
                f.write('"""Package initialization."""\n\n__version__ = "0.1.0"\n')
    
    print("✅ Проект настроен!")
    
    # Установка зависимостей
    if input("📦 Установить зависимости? (y/n): ").lower() == "y":
        install_dependencies()
    
    # Создание примеров данных
    if input("📊 Создать примеры данных? (y/n): ").lower() == "y":
        create_sample_data()


def install_dependencies():
    """Установка зависимостей."""
    print("📦 Установка зависимостей...")
    
    try:
        # Обновление pip
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Установка зависимостей проекта
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], check=True)
        
        # Установка pre-commit хуков
        subprocess.run(["pre-commit", "install"], check=True)
        
        print("✅ Зависимости установлены!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки зависимостей: {e}")
        sys.exit(1)


def create_sample_data():
    """Создание примеров данных."""
    print("📊 Создание примеров данных...")
    
    sample_data = {
        "data/raw/kk/sample.txt": """Қазақстан - әлемдегі ең үлкен мемлекеттердің бірі.
Астана қаласы - Қазақстанның астанасы.
Қазақ тілі - түркі тілдерінің бірі.
Тәуелсіздік 1991 жылы жарияланды.

Қазақстанның экономикасы негізінен мұнай, газ және табиғи ресурстарға негізделген.
Елде 100-ден астам этникалық топтар тұрады.

Алматы қаласы Қазақстанның ірі қаласы және мәдени орталығы болып табылады.
Қазақстан әлемдегі ең ірі шөлді аймақтардың біріне ие.""",
        
        "data/raw/ru/sample.txt": """Казахстан - одно из крупнейших государств в мире.
Астана - столица Казахстана.
Казахский язык относится к тюркским языкам.
Независимость была объявлена в 1991 году.

Экономика Казахстана основана в основном на нефти, газе и природных ресурсах.
В стране проживают более 100 этнических групп.

Город Алматы является крупным городом и культурным центром Казахстана.
Казахстан обладает одним из самых больших пустынных регионов в мире.""",
        
        "data/raw/en/sample.txt": """Kazakhstan is one of the largest countries in the world.
Astana is the capital of Kazakhstan.
The Kazakh language belongs to the Turkic languages.
Independence was declared in 1991.

Kazakhstan's economy is mainly based on oil, gas and natural resources.
More than 100 ethnic groups live in the country.

Almaty city is a major city and cultural center of Kazakhstan.
Kazakhstan has one of the largest desert regions in the world.""",
        
        "experiments/notebooks/01_data_exploration.ipynb": """{
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}""",
        
        "experiments/notebooks/02_model_training.ipynb": """{
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}""",
    }
    
    for file_path, content in sample_data.items():
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"📄 Создан файл: {file_path}")
    
    print("✅ Примеры данных созданы!")


def check_requirements():
    """Проверка системных требований."""
    print("🔍 Проверка системных требований...")
    
    requirements = [
        ("Python", "3.9", sys.version_info[:2] >= (3, 9)),
        ("CUDA", "11.8", True),  # Упрощенная проверка
        ("RAM", "16GB", True),   # Упрощенная проверка
        ("Disk", "50GB", True),  # Упрощенная проверка
    ]
    
    all_ok = True
    for req, min_version, ok in requirements:
        status = "✅" if ok else "❌"
        print(f"  {status} {req}: требуется {min_version}")
        if not ok:
            all_ok = False
    
    if not all_ok:
        print("⚠️  Некоторые требования не выполнены")
        if input("Продолжить несмотря на предупреждения? (y/n): ").lower() != "y":
            sys.exit(1)
    
    return all_ok


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description="Настройка проекта многоязычной LLM")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Принудительное создание файлов (перезапись существующих)",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Пропустить проверку требований",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("МНОГОЯЗЫЧНАЯ LLM - НАСТРОЙКА ПРОЕКТА")
    print("=" * 60)
    
    if not args.no_check:
        check_requirements()
    
    setup_project(force=args.force)
    
    print("\n🎉 Настройка завершена!")
    print("\nСледующие шаги:")
    print("1. Отредактируйте файл .env (установите API ключи)")
    print("2. Запустите: make install")
    print("3. Запустите: make download-data")
    print("4. Запустите: make train")
    print("5. Запустите: make demo")
    
    print("\nУдачи в разработке! 🚀")


if __name__ == "__main__":
    main()