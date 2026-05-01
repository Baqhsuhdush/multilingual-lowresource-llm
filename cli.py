"""
Командная строка для управления проектом.
"""

import click
import sys
from pathlib import Path
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Многоязычная LLM для низкоресурсных языков."""
    pass


@cli.group()
def data():
    """Управление данными."""
    pass


@data.command()
@click.option("--output", "-o", default="./data/processed", help="Выходная директория")
@click.option("--languages", "-l", multiple=True, default=["kk", "ru"], help="Языки для сбора")
@click.option("--max-samples", "-m", default=10000, help="Максимальное количество примеров на язык")
def collect(output, languages, max_samples):
    """Сбор данных из различных источников."""
    from src.data_processing.collector import MultilingualDataCollector, DataCollectorConfig
    
    config = DataCollectorConfig(
        languages=list(languages),
        max_samples_per_lang=max_samples,
    )
    
    collector = MultilingualDataCollector(config)
    
    try:
        dataset = collector.collect_all(Path(output))
        click.echo(f"✅ Данные собраны: {len(dataset['train'])} тренировочных примеров")
    except Exception as e:
        click.echo(f"❌ Ошибка сбора данных: {e}", err=True)
        sys.exit(1)


@data.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Выходной файл")
@click.option("--language", "-l", default="kk", help="Язык текстов")
def clean(input_file, output, language):
    """Очистка текстовых данных."""
    from src.data_processing.cleaner import DataCleaner, CleaningConfig
    
    config = CleaningConfig()
    cleaner = DataCleaner(config)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        cleaned_texts, cleaned_langs, stats = cleaner.process_batch(texts, [language] * len(texts))
        
        with open(output, 'w', encoding='utf-8') as f:
            for text in cleaned_texts:
                f.write(text + '\n')
        
        click.echo(f"✅ Данные очищены: {len(cleaned_texts)}/{len(texts)} текстов сохранено")
        click.echo(f"📊 Статистика: {stats}")
        
    except Exception as e:
        click.echo(f"❌ Ошибка очистки данных: {e}", err=True)
        sys.exit(1)


@cli.group()
def model():
    """Управление моделями."""
    pass


@model.command()
@click.option("--model-name", "-m", default="Qwen/Qwen2.5-1.5B", help="Название модели")
@click.option("--dataset", "-d", default="./data/processed", help="Путь к датасету")
@click.option("--output", "-o", default="./models/finetuned", help="Выходная директория")
@click.option("--epochs", "-e", default=3, help="Количество эпох")
@click.option("--batch-size", "-b", default=4, help="Размер пакета")
@click.option("--learning-rate", "-lr", default=2e-4, help="Скорость обучения")
def train(model_name, dataset, output, epochs, batch_size, learning_rate):
    """Тренировка модели с QLoRA."""
    from src.models.qlora_trainer import MultilingualQLoRATrainer, QLoRAConfig
    
    config = QLoRAConfig(
        model_name=model_name,
        output_dir=output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    trainer = MultilingualQLoRATrainer(config)
    
    try:
        # Загрузка датасета
        from datasets import load_from_disk
        dataset_obj = load_from_disk(dataset)
        
        click.echo(f"🧠 Начало тренировки модели {model_name}")
        click.echo(f"📊 Датасет: {len(dataset_obj['train'])} примеров")
        
        model, tokenizer = trainer.train(dataset_obj["train"], dataset_obj["validation"])
        
        click.echo(f"✅ Модель обучена и сохранена в {output}")
        
    except Exception as e:
        click.echo(f"❌ Ошибка тренировки: {e}", err=True)
        sys.exit(1)


@model.command()
@click.argument("prompt")
@click.option("--model-path", "-m", default="./models/finetuned", help="Путь к модели")
@click.option("--language", "-l", default="kk", help="Язык промпта")
@click.option("--max-length", default=200, help="Максимальная длина ответа")
@click.option("--temperature", default=0.7, help="Температура генерации")
def generate(prompt, model_path, language, max_length, temperature):
    """Генерация текста с помощью модели."""
    from src.models.base_model import MultilingualLLM, ModelConfig
    
    config = ModelConfig(
        model_name="Qwen/Qwen2.5-1.5B",
        load_in_4bit=True,
    )
    
    model = MultilingualLLM(config)
    
    # Загрузка fine-tuned модели если существует
    if Path(model_path).exists():
        try:
            model.load(model_path)
            click.echo(f"✅ Загружена fine-tuned модель из {model_path}")
        except Exception as e:
            click.echo(f"⚠️  Не удалось загрузить fine-tuned модель: {e}")
    
    try:
        click.echo(f"🤖 Генерация ответа на: {prompt}")
        
        response = model.generate(
            prompt=prompt,
            language=language,
            max_length=max_length,
            temperature=temperature,
        )
        
        click.echo(f"📝 Ответ: {response}")
        
    except Exception as e:
        click.echo(f"❌ Ошибка генерации: {e}", err=True)
        sys.exit(1)


@cli.group()
def rag():
    """Управление RAG системой."""
    pass


@rag.command()
@click.argument("documents_dir", type=click.Path(exists=True))
@click.option("--output", "-o", default="./data/faiss_indexes", help="Выходная директория")
@click.option("--chunk-size", default=512, help="Размер чанка")
@click.option("--chunk-overlap", default=50, help="Перекрытие чанков")
def build_index(documents_dir, output, chunk_size, chunk_overlap):
    """Построение FAISS индекса."""
    from src.rag.vector_store import VectorStore, VectorStoreConfig
    import json
    
    config = VectorStoreConfig(
        persist_dir=output,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    store = VectorStore(config)
    
    try:
        # Загрузка документов
        documents = []
        docs_path = Path(documents_dir)
        
        for file_path in docs_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": str(file_path),
                            "language": "kk" if "kk" in str(file_path) else "ru",
                        }
                    })
        
        for file_path in docs_path.glob("*.jsonl"):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        if "text" in doc:
                            documents.append(doc)
                    except json.JSONDecodeError:
                        continue
        
        click.echo(f"📚 Загружено {len(documents)} документов")
        
        # Добавление в индекс
        chunks_added = store.add_documents(documents, show_progress=True)
        
        click.echo(f"✅ Индекс построен: {chunks_added} чанков")
        click.echo(f"📊 Статистика: {store.get_stats()}")
        
    except Exception as e:
        click.echo(f"❌ Ошибка построения индекса: {e}", err=True)
        sys.exit(1)


@rag.command()
@click.argument("query")
@click.option("--index-path", "-i", default="./data/faiss_indexes", help="Путь к индексу")
@click.option("--top-k", "-k", default=5, help="Количество результатов")
def search(query, index_path, top_k):
    """Поиск в векторном хранилище."""
    from src.rag.vector_store import VectorStore, VectorStoreConfig
    
    config = VectorStoreConfig(persist_dir=index_path)
    
    try:
        store = VectorStore(config)
        
        click.echo(f"🔍 Поиск: {query}")
        
        results = store.search(query, k=top_k)
        
        if not results:
            click.echo("❌ Результаты не найдены")
            return
        
        for i, result in enumerate(results, 1):
            click.echo(f"\n{i}. Схожесть: {result['score']:.3f}")
            click.echo(f"   Текст: {result['text'][:200]}...")
            if result.get('metadata'):
                click.echo(f"   Метаданные: {result['metadata']}")
        
    except Exception as e:
        click.echo(f"❌ Ошибка поиска: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Хост сервера")
@click.option("--port", default=8000, help="Порт сервера")
@click.option("--reload", is_flag=True, help="Автоматическая перезагрузка")
def serve(host, port, reload):
    """Запуск API сервера."""
    import uvicorn
    
    click.echo(f"🚀 Запуск API сервера на {host}:{port}")
    
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@cli.command()
def demo():
    """Запуск демо интерфейса."""
    import subprocess
    import sys
    
    click.echo("🚀 Запуск демо интерфейса...")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "demo/app.py"])
    except KeyboardInterrupt:
        click.echo("\n👋 Демо остановлено")
    except Exception as e:
        click.echo(f"❌ Ошибка запуска демо: {e}", err=True)
        sys.exit(1)


@cli.command()
def test():
    """Запуск тестов."""
    import subprocess
    import sys
    
    click.echo("🧪 Запуск тестов...")
    
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    
    if result.returncode != 0:
        sys.exit(result.returncode)


@cli.command()
@click.option("--check", is_flag=True, help="Только проверка (без изменений)")
def format_code(check):
    """Форматирование кода."""
    import subprocess
    import sys
    
    click.echo("🎨 Форматирование кода...")
    
    args = ["black", "src", "tests", "demo"]
    if check:
        args.append("--check")
    
    result = subprocess.run(args)
    
    if result.returncode != 0:
        sys.exit(result.returncode)
    
    if not check:
        click.echo("✅ Код отформатирован")


@cli.command()
def setup():
    """Настройка проекта."""
    import subprocess
    import sys
    
    click.echo("⚙️  Настройка проекта...")
    
    # Запуск скрипта настройки
    result = subprocess.run([sys.executable, "scripts/setup_project.py"])
    
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    cli()