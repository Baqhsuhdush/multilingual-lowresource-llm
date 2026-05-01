"""
Тесты для обработки данных.
"""

import pytest
from pathlib import Path
import tempfile
import json

from src.data_processing.cleaner import DataCleaner, CleaningConfig
from src.data_processing.collector import MultilingualDataCollector, DataCollectorConfig


class TestDataCleaner:
    """Тесты для очистки данных."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.config = CleaningConfig(
            min_length=10,
            max_length=1000,
            min_words=2,
            target_languages=["kk", "ru", "en"],
        )
        self.cleaner = DataCleaner(self.config)
    
    def test_clean_text_basic(self):
        """Тест базовой очистки текста."""
        text = "  Hello   World!  \n\nThis is a test.  "
        cleaned = self.cleaner.clean_text(text, "en")
        
        assert cleaned == "Hello World! This is a test."
        assert "  " not in cleaned
        assert "\n" not in cleaned
    
    def test_clean_text_urls(self):
        """Тест удаления URL."""
        text = "Visit https://example.com for more info. Also check www.test.org"
        cleaned = self.cleaner.clean_text(text, "en")
        
        assert "https://example.com" not in cleaned
        assert "www.test.org" not in cleaned
        assert "Visit for more info. Also check" in cleaned
    
    def test_clean_text_emails(self):
        """Тест удаления email адресов."""
        text = "Contact us at info@example.com or support@test.org"
        cleaned = self.cleaner.clean_text(text, "en")
        
        assert "info@example.com" not in cleaned
        assert "support@test.org" not in cleaned
        assert "Contact us at or" in cleaned
    
    def test_filter_by_length(self):
        """Тест фильтрации по длине."""
        # Слишком короткий
        assert not self.cleaner.filter_by_length("Hi")
        
        # Нормальный
        assert self.cleaner.filter_by_length("This is a normal length text")
        
        # Слишком длинный (симулируем)
        long_text = "word " * 1000  # 5000 символов
        assert not self.cleaner.filter_by_length(long_text)
    
    def test_process_batch(self):
        """Тест пакетной обработки."""
        texts = [
            "Қазақстан - әлемдегі ең үлкен мемлекеттердің бірі.",
            "Казахстан - одно из крупнейших государств в мире.",
            "Short",  # Будет отфильтрован
            "Another good text in English with proper length and structure.",
        ]
        
        languages = ["kk", "ru", "kk", "en"]
        
        cleaned_texts, cleaned_langs, stats = self.cleaner.process_batch(texts, languages)
        
        assert len(cleaned_texts) == 3  # Один отфильтрован
        assert "Short" not in cleaned_texts
        assert stats["total"] == 4
        assert stats["filtered_by_length"] == 1
        assert stats["final_count"] == 3


class TestDataCollector:
    """Тесты для сборщика данных."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.config = DataCollectorConfig(
            languages=["kk", "ru"],
            max_samples_per_lang=100,
            min_text_length=20,
        )
        
        self.temp_dir = tempfile.mkdtemp()
        self.collector = MultilingualDataCollector(self.config)
    
    def test_load_local_data(self):
        """Тест загрузки локальных данных."""
        # Создаем тестовые файлы
        test_data = {
            "kk": ["Қазақстан туралы мәтін.", "Тағы бір қазақша мәтін."],
            "ru": ["Текст о Казахстане.", "Еще один русский текст."],
        }
        
        for lang, texts in test_data.items():
            lang_dir = Path(self.temp_dir) / "raw" / lang
            lang_dir.mkdir(parents=True, exist_ok=True)
            
            for i, text in enumerate(texts):
                file_path = lang_dir / f"test_{i}.txt"
                file_path.write_text(text, encoding='utf-8')
        
        # Загрузка данных
        data = self.collector.load_local_data(Path(self.temp_dir))
        
        assert "kk" in data
        assert "ru" in data
        assert len(data["kk"]) == 2
        assert len(data["ru"]) == 2
    
    def test_validate_text(self):
        """Тест валидации текста."""
        # Хороший текст
        good_text = "Қазақстан - әлемдегі ең үлкен мемлекеттердің бірі."
        assert self.collector._validate_text(good_text, "kk")
        
        # Слишком короткий
        short_text = "Қазақстан"
        assert not self.collector._validate_text(short_text, "kk")
        
        # Неправильный язык
        russian_text = "Казахстан - большое государство."
        assert not self.collector._validate_text(russian_text, "kk")
    
    def test_preprocess_text(self):
        """Тест предобработки текста."""
        text = "  Қазақстан - https://example.com әлемдегі   ең үлкен мемлекет.  "
        processed = self.collector._preprocess_text(text, "kk")
        
        assert "https://example.com" not in processed
        assert "  " not in processed
        assert processed.startswith("Қазақстан")
        assert processed.endswith("мемлекет.")
    
    def test_create_training_data(self):
        """Тест создания тренировочных данных."""
        from datasets import Dataset
        
        # Создаем тестовый датасет
        test_data = {
            "text": [
                "Қазақстан туралы қысқа мәтін.",
                "Краткий текст о Казахстане.",
            ],
            "language": ["kk", "ru"],
            "source": ["test", "test"],
        }
        
        dataset = Dataset.from_dict(test_data)
        dataset_dict = {"train": dataset}
        
        # Создаем тренировочные данные
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            output_file = Path(f.name)
            self.collector.create_training_data(dataset_dict, output_file)
            
            # Проверяем созданный файл
            assert output_file.exists()
            
            with open(output_file, 'r', encoding='utf-8') as result_file:
                lines = result_file.readlines()
                assert len(lines) == 2
                
                for line in lines:
                    data = json.loads(line)
                    assert "instruction" in data
                    assert "output" in data
                    assert "language" in data
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestEdgeCases:
    """Тесты граничных случаев."""
    
    def test_empty_input(self):
        """Тест с пустым вводом."""
        cleaner = DataCleaner()
        
        texts = []
        languages = []
        
        cleaned_texts, cleaned_langs, stats = cleaner.process_batch(texts, languages)
        
        assert len(cleaned_texts) == 0
        assert len(cleaned_langs) == 0
        assert stats["total"] == 0
        assert stats["final_count"] == 0
    
    def test_special_characters_only(self):
        """Тест только со специальными символами."""
        cleaner = DataCleaner()
        
        text = "### @@@ $$$ %%% ^^^ &&&"
        cleaned = cleaner.clean_text(text, "en")
        
        # После очистки должен остаться пустой текст или только пробелы
        assert len(cleaned.strip()) == 0 or all(c in ' #$%&@^' for c in cleaned)
    
    def test_very_long_word(self):
        """Тест с очень длинным словом."""
        cleaner = DataCleaner(CleaningConfig(min_words=1, max_words=10))
        
        text = "a" * 1000  # Очень длинное "слово"
        
        # Одно слово, но очень длинное
        assert cleaner.filter_by_length(text)  # Проходит по количеству слов