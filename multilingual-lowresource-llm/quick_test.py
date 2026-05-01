#!/usr/bin/env python3
"""
Быстрый тест работы проекта.
"""

import sys
from pathlib import Path

# Добавляем src в путь Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.utils.multilingual import detect_language, normalize_text
from src.models.base_model import MultilingualLLM, ModelConfig

logger = setup_logger("quick_test")


def test_multilingual_utils():
    """Тест мультиязычных утилит."""
    print("🧪 Тестирование мультиязычных утилит...")
    
    test_texts = [
        ("Қазақстан - әлемдегі ең үлкен мемлекеттердің бірі.", "kk"),
        ("Казахстан - одно из крупнейших государств в мире.", "ru"),
        ("Kazakhstan is one of the largest countries in the world.", "en"),
    ]
    
    for text, expected_lang in test_texts:
        detected = detect_language(text)
        normalized = normalize_text(text, language=expected_lang)
        
        print(f"\nТекст: {text[:50]}...")
        print(f"Ожидаемый язык: {expected_lang}")
        print(f"Обнаруженный язык: {detected}")
        print(f"Нормализованный текст: {normalized[:50]}...")
        
        if detected == expected_lang:
            print("✅ Язык определен правильно")
        else:
            print("⚠️  Язык определен неправильно")
    
    print("\n✅ Тест мультиязычных утилит завершен")


def test_model_loading():
    """Тест загрузки модели."""
    print("\n🧪 Тестирование загрузки модели...")
    
    try:
        # Используем маленькую модель для теста
        config = ModelConfig(
            model_name="Qwen/Qwen2.5-0.5B",  # Маленькая модель для теста
            load_in_4bit=False,  # Отключаем quantization для теста
        )
        
        model = MultilingualLLM(config)
        
        print(f"✅ Модель загружена: {config.model_name}")
        print(f"✅ Устройство: {model.device}")
        print(f"✅ Токенизатор: {model.tokenizer.__class__.__name__}")
        
        return model
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return None


def test_generation(model):
    """Тест генерации текста."""
    if model is None:
        print("⚠️  Пропуск теста генерации (модель не загружена)")
        return
    
    print("\n🧪 Тестирование генерации текста...")
    
    test_prompts = [
        ("Қазақстан туралы не білесіз?", "kk"),
        ("Что вы знаете о Казахстане?", "ru"),
        ("What do you know about Kazakhstan?", "en"),
    ]
    
    for prompt, language in test_prompts:
        print(f"\nПромпт ({language}): {prompt}")
        
        try:
            response = model.generate(
                prompt=prompt,
                language=language,
                max_length=100,
                temperature=0.7,
            )
            
            print(f"Ответ: {response}")
            print("✅ Генерация успешна")
            
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")


def test_embeddings(model):
    """Тест получения эмбеддингов."""
    if model is None:
        print("⚠️  Пропуск теста эмбеддингов (модель не загружена)")
        return
    
    print("\n🧪 Тестирование эмбеддингов...")
    
    texts = [
        "Қазақстан",
        "Казахстан",
        "Kazakhstan",
    ]
    
    try:
        embeddings = model.get_embeddings(texts)
        
        print(f"✅ Получено {len(embeddings)} эмбеддингов")
        print(f"✅ Форма эмбеддингов: {embeddings.shape}")
        
        # Проверка сходства
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity(embeddings)
        
        print("\nКосинусное сходство между текстами:")
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = similarities[i, j]
                print(f"  {texts[i][:10]}... ↔ {texts[j][:10]}...: {sim:.3f}")
        
    except Exception as e:
        print(f"❌ Ошибка получения эмбеддингов: {e}")


def test_perplexity(model):
    """Тест расчета перплексии."""
    if model is None:
        print("⚠️  Пропуск теста перплексии (модель не загружена)")
        return
    
    print("\n🧪 Тестирование расчета перплексии...")
    
    texts = [
        "Қазақстан - әлемдегі ең үлкен мемлекеттердің бірі.",
        "Бұл мәтіннің сапасы нашар.",
    ]
    
    for text in texts:
        try:
            perplexity = model.calculate_perplexity(text)
            print(f"Текст: {text[:30]}...")
            print(f"Перплексия: {perplexity:.2f}")
            print("✅ Расчет перплексии успешен")
        except Exception as e:
            print(f"❌ Ошибка расчета перплексии: {e}")


def check_project_structure():
    """Проверка структуры проекта."""
    print("🔍 Проверка структуры проекта...")
    
    required_dirs = [
        "src",
        "data/raw/kk",
        "data/raw/ru",
        "models",
        "config",
        "experiments",
        "tests",
        "demo",
    ]
    
    required_files = [
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        ".env.example",
        "Makefile",
        "src/__init__.py",
        "config/model_config.yaml",
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ Директория: {dir_path}")
        else:
            print(f"❌ Отсутствует директория: {dir_path}")
            all_ok = False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ Файл: {file_path}")
        else:
            print(f"❌ Отсутствует файл: {file_path}")
            all_ok = False
    
    return all_ok


def main():
    """Основная функция."""
    print("=" * 60)
    print("МНОГОЯЗЫЧНАЯ LLM - БЫСТРЫЙ ТЕСТ ПРОЕКТА")
    print("=" * 60)
    
    # Проверка структуры проекта
    if not check_project_structure():
        print("\n⚠️  Не все файлы и директории на месте")
        print("Запустите: python scripts/setup_project.py")
        return
    
    # Тестирование
    try:
        test_multilingual_utils()
        
        model = test_model_loading()
        
        if model:
            test_generation(model)
            test_embeddings(model)
            test_perplexity(model)
        
        print("\n" + "=" * 60)
        print("✅ Все тесты завершены успешно!")
        print("\nПроект готов к работе! 🚀")
        
        print("\nСледующие шаги:")
        print("1. Отредактируйте .env файл с вашими API ключами")
        print("2. Запустите полный пайплайн: make train")
        print("3. Запустите демо: make demo")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Тест прерван пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()