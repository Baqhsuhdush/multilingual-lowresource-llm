"""
Базовый класс для многоязычной языковой модели.
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
from peft import PeftModel, LoraConfig, get_peft_model
import numpy as np

from src.utils.logger import setup_logger
from src.utils.multilingual import detect_language, normalize_text

logger = setup_logger(__name__)


@dataclass
class ModelConfig:
    """Конфигурация модели."""
    model_name: str = "Qwen/Qwen2.5-1.5B"
    revision: str = "main"
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.float16
    load_in_4bit: bool = True
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None
    
    # Параметры генерации
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    length_penalty: float = 1.0
    
    # Параметры LoRA
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None


class MultilingualLLM:
    """Базовый класс для многоязычной языковой модели."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Инициализация модели и токенизатора
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        # Кэш для эмбеддингов
        self.embedding_cache = {}
        
        # Загрузка модели
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели и токенизатора."""
        logger.info(f"Загрузка модели: {self.config.model_name}")
        
        try:
            # Конфигурация quantization
            bnb_config = None
            if self.config.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.config.torch_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            
            # Загрузка токенизатора
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                revision=self.config.revision,
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir,
                padding_side="right",
            )
            
            # Установка pad token если не установлен
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Загрузка модели
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                revision=self.config.revision,
                device_map=self.config.device_map,
                torch_dtype=self.config.torch_dtype,
                quantization_config=bnb_config,
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir,
            )
            
            # Настройка конфигурации генерации
            self.generation_config = GenerationConfig(
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.reputation_penalty,
                do_sample=self.config.do_sample,
                num_beams=self.config.num_beams,
                length_penalty=self.config.length_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            logger.info(f"Модель загружена на устройство: {self.model.device}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def apply_lora(self, lora_config: Optional[LoraConfig] = None):
        """Применение LoRA адаптеров к модели."""
        if not self.config.lora_enabled:
            logger.info("LoRA отключен в конфигурации")
            return
        
        logger.info("Применение LoRA адаптеров")
        
        if lora_config is None:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules or [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                bias="none",
                task_type="CAUSAL_LM",
            )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def load_adapter(self, adapter_path: str):
        """Загрузка адаптеров LoRA."""
        try:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            logger.info(f"Адаптеры загружены из: {adapter_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки адаптеров: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        language: str = "kk",
        context: Optional[str] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Генерация текста на основе промпта.
        
        Args:
            prompt: Промпт для генерации
            language: Язык промпта
            context: Контекст для RAG
            max_length: Максимальная длина ответа
            temperature: Температура для генерации
            **kwargs: Дополнительные параметры генерации
        
        Returns:
            Сгенерированный текст
        """
        # Подготовка промпта с учетом языка и контекста
        full_prompt = self._prepare_prompt(prompt, language, context)
        
        # Токенизация
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
        ).to(self.model.device)
        
        # Настройка параметров генерации
        generation_config = self.generation_config
        if max_length is not None:
            generation_config = generation_config.__class__.from_dict({
                **generation_config.to_dict(),
                "max_length": max_length,
            })
        if temperature is not None:
            generation_config = generation_config.__class__.from_dict({
                **generation_config.to_dict(),
                "temperature": temperature,
            })
        
        # Обновление параметров из kwargs
        if kwargs:
            generation_config = generation_config.__class__.from_dict({
                **generation_config.to_dict(),
                **kwargs,
            })
        
        # Генерация
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )
        
        # Декодирование
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Извлечение только ответа (без промпта)
        response = self._extract_response(generated_text, full_prompt, language)
        
        return response
    
    def _prepare_prompt(
        self,
        prompt: str,
        language: str,
        context: Optional[str] = None,
    ) -> str:
        """Подготовка промпта для генерации."""
        # Нормализация текста
        prompt = normalize_text(prompt, language=language)
        
        # Создание полного промпта
        if language == "kk":
            if context:
                full_prompt = (
                    f"Контекст: {context}\n\n"
                    f"Сұрақ: {prompt}\n\n"
                    f"Жауап:"
                )
            else:
                full_prompt = f"Сұрақ: {prompt}\nЖауап:"
        
        elif language == "ru":
            if context:
                full_prompt = (
                    f"Контекст: {context}\n\n"
                    f"Вопрос: {prompt}\n\n"
                    f"Ответ:"
                )
            else:
                full_prompt = f"Вопрос: {prompt}\nОтвет:"
        
        else:  # English
            if context:
                full_prompt = (
                    f"Context: {context}\n\n"
                    f"Question: {prompt}\n\n"
                    f"Answer:"
                )
            else:
                full_prompt = f"Question: {prompt}\nAnswer:"
        
        return full_prompt
    
    def _extract_response(
        self,
        generated_text: str,
        prompt: str,
        language: str,
    ) -> str:
        """Извлечение ответа из сгенерированного текста."""
        # Удаляем промпт из сгенерированного текста
        if prompt in generated_text:
            response = generated_text[len(prompt):].strip()
        else:
            response = generated_text.strip()
        
        # Обрезаем по стоп-символам
        stop_sequences = []
        if language == "kk":
            stop_sequences = ["\nСұрақ:", "\nКонтекст:"]
        elif language == "ru":
            stop_sequences = ["\nВопрос:", "\nКонтекст:"]
        else:
            stop_sequences = ["\nQuestion:", "\nContext:"]
        
        for stop_seq in stop_sequences:
            if stop_seq in response:
                response = response.split(stop_seq)[0].strip()
        
        return response
    
    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        layer: int = -1,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Получение эмбеддингов текстов.
        
        Args:
            texts: Текст или список текстов
            layer: Слой для извлечения эмбеддингов
            normalize: Нормализовать ли эмбеддинги
        
        Returns:
            Массив эмбеддингов
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for text in texts:
            # Проверка кэша
            cache_key = f"{text}_{layer}"
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
                continue
            
            # Токенизация
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(self.model.device)
            
            # Получение эмбеддингов
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                )
                
                # Извлечение эмбеддингов из указанного слоя
                hidden_states = outputs.hidden_states[layer]
                
                # Усреднение по токенам (исключая padding)
                attention_mask = inputs["attention_mask"]
                token_embeddings = hidden_states * attention_mask.unsqueeze(-1)
                sentence_embedding = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                # Конвертация в numpy
                embedding = sentence_embedding.cpu().numpy()[0]
                
                # Нормализация
                if normalize:
                    embedding = embedding / np.linalg.norm(embedding)
                
                # Сохранение в кэш
                self.embedding_cache[cache_key] = embedding
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Расчет перплексии текста.
        
        Args:
            text: Текст для расчета
        
        Returns:
            Значение перплексии
        """
        # Токенизация
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.model.device)
        
        # Расчет потерь
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        
        # Расчет перплексии
        perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def batch_generate(
        self,
        prompts: List[str],
        language: str = "kk",
        batch_size: int = 4,
        **kwargs,
    ) -> List[str]:
        """
        Пакетная генерация текстов.
        
        Args:
            prompts: Список промптов
            language: Язык промптов
            batch_size: Размер пакета
            **kwargs: Дополнительные параметры генерации
        
        Returns:
            Список сгенерированных текстов
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Пакетная обработка
            batch_inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
            ).to(self.model.device)
            
            # Генерация
            with torch.no_grad():
                batch_outputs = self.model.generate(
                    **batch_inputs,
                    generation_config=self.generation_config,
                    **kwargs,
                )
            
            # Декодирование
            batch_results = self.tokenizer.batch_decode(
                batch_outputs,
                skip_special_tokens=True,
            )
            
            # Извлечение ответов
            for prompt, result in zip(batch_prompts, batch_results):
                response = self._extract_response(result, prompt, language)
                results.append(response)
        
        return results
    
    def save(self, path: str):
        """Сохранение модели."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Модель сохранена в: {path}")
    
    def load(self, path: str):
        """Загрузка модели."""
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info(f"Модель загружена из: {path}")
        