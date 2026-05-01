"""
Тесты для мультиязычных утилит.
"""

import pytest
from src.utils.multilingual import (
    detect_language,
    normalize_text,
    translate_text,
    split_sentences,
)


class TestMultilingualUtils:
    """Тесты мультиязычных утилит."""
    
    def test_detect_language_kazakh(self):
        """Тест определения казахского языка."""
        text = "Қазақстан - әлемдегі ең үлкен мемлекеттердің бірі."
        language = detect_language(text)
        assert language == "kk"
    
    def test_detect_language_russian(self):
        """Тест определения русского языка."""
        text = "Казахстан - одно из крупнейших государств в мире."
        language = detect_language(text)
        assert language == "ru"
    
    def test_detect_language_english(self):
        """Тест определения английского языка."""
        text = "Kazakhstan is one of the largest countries in the world."
        language = detect_language(text)
        assert language == "en"
    
    def test_normalize_text_kazakh(self):
        """Тест нормализации казахского текста."""
        text = "  Қазақстан   - әлемдегі   ең үлкен мемлекеттердің бірі.  "
        normalized = normalize_text(text, language="kk")
        
        assert normalized == "Қазақстан - әлемдегі ең үлкен мемлекеттердің бірі."
        assert "  " not in normalized  # Нет двойных пробелов
    
    def test_normalize_text_russian(self):
        """Тест нормализации русского текста."""
        text = "  Казахстан   - одно из   крупнейших государств в мире.  "
        normalized = normalize_text(text, language="ru")
        
        assert normalized == "Казахстан - одно из крупнейших государств в мире."
        assert "  " not in normalized
    
    def test_split_sentences(self):
        """Тест разделения текста на предложения."""
        text = "Қазақстан - әлемдегі ең үлкен мемлекеттердің бірі. Астана - астанасы. Қазақ тілі - түркі тілдерінің бірі."
        sentences = split_sentences(text, language="kk")
        
        assert len(sentences) == 3
        assert sentences[0] == "Қазақстан - әлемдегі ең үлкен мемлекеттердің бірі."
        assert sentences[1] == "Астана - астанасы."
        assert sentences[2] == "Қазақ тілі - түркі тілдерінің бірі."
    
    @pytest.mark.skip(reason="Требуется интернет-соединение")
    def test_translate_text(self):
        """Тест перевода текста."""
        text = "Казахстан - большое государство."
        translated = translate_text(text, target_lang="kk", source_lang="ru")
        
        assert translated is not None
        assert len(translated) > 0


class TestLanguageDetectionEdgeCases:
    """Тесты граничных случаев определения языка."""
    
    def test_short_text(self):
        """Тест с коротким текстом."""
        text = "Қазақстан"
        language = detect_language(text)
        # Короткие тексты могут не определяться точно
        assert language is None or language in ["kk", "ru"]
    
    def test_mixed_language(self):
        """Тест со смешанным языком."""
        text = "Қазақстан - это Kazakhstan"
        language = detect_language(text)
        # Может определить один из языков
        assert language in ["kk", "ru", "en"] or language is None
    
    def test_empty_text(self):
        """Тест с пустым текстом."""
        text = ""
        language = detect_language(text)
        assert language is None
    
    def test_special_characters(self):
        """Тест со специальными символами."""
        text = "### @@@ 123 $$$"
        language = detect_language(text)
        assert language is None


class TestTextQuality:
    """Тесты качества текста."""
    
    def test_text_validation(self):
        """Тест валидации текста."""
        from src.utils.multilingual import check_text_quality
        
        # Хороший текст
        text_good = "Қазақстан - әлемдегі ең үлкен мемлекеттердің бірі."
        is_valid_good, metrics_good = check_text_quality(text_good, "kk")
        assert is_valid_good is True
        
        # Короткий текст
        text_short = "Қазақстан"
        is_valid_short, metrics_short = check_text_quality(text_short, "kk", min_length=20)
        assert is_valid_short is False
        
        # Много специальных символов
        text_special = "### @@@ Қазақстан $$$ %%%"
        is_valid_special, metrics_special = check_text_quality(text_special, "kk", max_special_chars=0.1)
        assert is_valid_special is False