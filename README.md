<h1 align="center">Welcome to multilingual-lowresource-llm 👋</h1>
<p>
</p>

## Show your support
Research-oriented repository for developing an efficient multilingual LLM for low-resource languages (Kazakh/Russian). Includes QLoRA fine-tuning, RAG retrieval system, FAISS indexing, and a demo interface.


# Multilingual Low-Resource LLM with QLoRA & RAG 

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** Kurbanali Bakdaulet  
**Target Focus:** Natural Language Processing (NLP), Low-Resource Languages (Kazakh / Russian), Parameter-Efficient Fine-Tuning (PEFT), Retrieval-Augmented Generation (RAG).

---

## 📌 Abstract & Overview

Standard Large Language Models (LLMs) often exhibit degraded performance and high tokenization overhead when applied to **low-resource languages** (e.g., Kazakh). This research-oriented repository implements an end-to-end framework designed to fine-tune open-weights LLMs using **QLoRA (4-bit Quantized Low-Rank Adaptation)**, coupled with a high-throughput **RAG (Retrieval-Augmented Generation)** framework backed by **FAISS vector indexing**.

The architecture aims to mitigate factual hallucination and maximize context efficiency in low-resource domain specific tasks.

---

## 🏗 System Architecture & Key Features

---


┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Data Collection │ ───► │  Data Cleaning   │ ───► │ Tokenization &   │
│ (collector.py)  │      │   (cleaner.py)   │      │ Preprocessing    │
└─────────────────┘      └──────────────────┘      └──────────────────┘
│
▼
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ User Interface  │ ◄─── │  RAG Engine &    │ ◄─── │ QLoRA Adapter &  │
│ (app.py/server) │      │ FAISS Retrieval  │      │ Base LLM Inference│
└─────────────────┘      └──────────────────┘      └──────────────────┘

1. **Data Preprocessing & Scraping Pipeline (`collector.py`, `cleaner.py`):** Automated raw data ingestion, text normalization, and filtering optimized for agglutinative language morphology[cite: 4].
2. **PEFT / QLoRA Fine-Tuning (`base_model.py`, `multilingual.py`):** Efficient 4-bit/8-bit quantized fine-tuning setup using HuggingFace `bitsandbytes` and `peft` to reduce GPU VRAM consumption while preserving downstream evaluation accuracy[cite: 4, 5].
3. **FAISS Vector Indexing & RAG (`vector_store.py`, `rag_config.yaml`):** Dense retrieval architecture indexing domain embeddings into FAISS indices for zero-shot context augmentation[cite: 4, 5].
4. **Production Deployment (`server.py`, `app.py`, `Dockerfile`):** Microservice-ready architecture served via FastAPI/Streamlit and containerized using Docker and Docker Compose[cite: 4, 5].

---

## 📂 Repository Structure


multilingual-lowresource-llm/
├── .env.example              # Environment variables template
├── architecture.md           # Deep dive into system architecture & design choices
├── api_documentation.md      # RESTful API specifications
├── docker-compose.yml        # Multi-container orchestration (API + Vector Store)
├── Dockerfile                # Production container definition
├── Makefile                  # Build, test, and execution shortcuts
├── model_config.yaml         # LLM hyperparameters & quantization configs
├── rag_config.yaml           # Chunking strategy & vector store parameters
├── training_config.yaml      # QLoRA fine-tuning hyperparameters
│
├── base_model.py             # LLM loading & quantization configurations
├── multilingual.py           # Multilingual tokenization & generation logic
├── vector_store.py           # FAISS index management & retrieval engine
├── collector.py              # Data ingestion module
├── cleaner.py                # Text normalization and data preprocessing
├── chat_interface.py         # Conversational state manager
├── cli.py                    # Command-line interface for evaluation
├── app.py / server.py        # Web API / Interactive UI entrypoints
└── tests/                    # Unit & integration test suites

---

## 🚀 Quick Start Guide

### Prerequisites
* **OS:** Linux / WSL2 / macOS
* **Python:** 3.10+
* **NVIDIA GPU:** Required for QLoRA fine-tuning (CUDA 11.8+ / 12.0+)

### 1. Installation

bash
# Clone the repository
git clone [https://github.com/your-username/multilingual-lowresource-llm.git](https://github.com/your-username/multilingual-lowresource-llm.git)
cd multilingual-lowresource-llm

# Setup environment variables
cp .env.example .env

# Install dependencies using Makefile or Pip
make setup
# OR: pip install -r pyproject.toml


### 2. Execution Workflows

Using `Makefile` shortcuts for standardized operations:

bash
# 1. Run Data Preprocessing
python cleaner.py

# 2. Build FAISS Vector Store Index
python vector_store.py --build

# 3. Fine-Tune LLM with QLoRA
python multilingual.py --train --config training_config.yaml

# 4. Launch Local Web Interface & Server
python server.py

### 3. Docker Deployment

bash
# Build and launch using Docker Compose
docker-compose up --build -d

---

## 📊 Evaluation & Methodology

* **Base Models:** LLaMA-3 / Mistral-7B / Gemma.
* **Fine-Tuning Technique:** Low-Rank Adaptation (LoRA) with 4-bit NormalFloat (NF4) quantization.
* **Retrieval Metric:** Recall@K and MRR (Mean Reciprocal Rank) on custom Kazakh-Russian evaluation datasets.

---

## 📄 License & Citation

This project is released under the [MIT License](https://www.google.com/search?q=LICENSE&utm_source=gemini).
bibtex
@article{bakdaulet2026multilingual,
  title={Multilingual Low-Resource LLM Adaptation using QLoRA and RAG Retrieval},
  author={Bakdaulet, Kurbanali},
  year={2026}
}
