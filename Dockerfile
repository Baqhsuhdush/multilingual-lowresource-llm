# ============================================
# МНОГОЯЗЫЧНАЯ LLM - Dockerfile
# ============================================

# Базовый образ с Python и CUDA
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-dev \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    libopenblas-dev \
    libomp-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    libncursesw5-dev \
    libgdbm-dev \
    libnss3-dev \
    liblzma-dev \
    tk-dev \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python 3.9 как версию по умолчанию
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Устанавливаем pip и основные инструменты
RUN pip3 install --upgrade pip setuptools wheel

# Создаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
COPY pyproject.toml .

# Устанавливаем PyTorch с поддержкой CUDA
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Устанавливаем основные зависимости
RUN pip3 install -r requirements.txt

# Устанавливаем проект в режиме разработки
COPY . .
RUN pip3 install -e ".[full]"

# Создаем необходимые директории
RUN mkdir -p /app/data/raw /app/data/processed /app/data/embeddings /app/data/faiss_indexes \
    && mkdir -p /app/models /app/logs /app/mlruns \
    && chmod -R 777 /app/logs /app/mlruns

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets

# Открываем порты
EXPOSE 8000  # FastAPI
EXPOSE 8501  # Streamlit
EXPOSE 9090  # Prometheus
EXPOSE 5000  # MLflow

# Создаем точку входа
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Создаем пользователя для безопасности
RUN useradd -m -u 1000 -s /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Точка входа
ENTRYPOINT ["/entrypoint.sh"]