"""
Демонстрационное приложение Streamlit для многоязычной LLM.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
import time
import requests
import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.utils.multilingual import detect_language, normalize_text

# Настройка страницы
st.set_page_config(
    page_title="Multilingual LLM Demo",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация логгера
logger = setup_logger("streamlit_demo")

# Настройки API
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Стили CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E8F5E9;
        border: 1px solid #C8E6C9;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E3F2FD;
        border: 1px solid #90CAF9;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFF3E0;
        border: 1px solid #FFCC80;
    }
    .language-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: bold;
    }
    .kazakh-badge {
        background-color: #4CAF50;
        color: white;
    }
    .russian-badge {
        background-color: #2196F3;
        color: white;
    }
    .english-badge {
        background-color: #FF9800;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


class MultilingualDemo:
    """Класс для управления демо приложением."""
    
    def __init__(self):
        self.setup_session_state()
        self.setup_sidebar()
    
    def setup_session_state(self):
        """Инициализация состояния сессии."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "search_history" not in st.session_state:
            st.session_state.search_history = []
        
        if "documents" not in st.session_state:
            st.session_state.documents = []
        
        if "api_connected" not in st.session_state:
            st.session_state.api_connected = False
        
        if "selected_language" not in st.session_state:
            st.session_state.selected_language = "kk"
    
    def setup_sidebar(self):
        """Настройка боковой панели."""
        with st.sidebar:
            st.title("⚙️ Настройки")
            
            # Подключение к API
            st.subheader("🔗 Подключение к API")
            api_url = st.text_input("URL API", API_URL)
            
            if st.button("Проверить подключение", key="check_connection"):
                if self.check_api_connection(api_url):
                    st.success("✅ Подключено к API")
                    st.session_state.api_connected = True
                else:
                    st.error("❌ Не удалось подключиться к API")
                    st.session_state.api_connected = False
            
            # Выбор языка
            st.subheader("🌐 Язык")
            language_options = {
                "Қазақша": "kk",
                "Русский": "ru",
                "English": "en"
            }
            
            selected_lang = st.selectbox(
                "Выберите язык интерфейса:",
                list(language_options.keys()),
                index=0
            )
            st.session_state.selected_language = language_options[selected_lang]
            
            # Настройки генерации
            st.subheader("🤖 Настройки модели")
            st.session_state.temperature = st.slider(
                "Температура", 0.1, 1.0, 0.7, 0.1,
                help="Высокая температура = более креативные ответы"
            )
            st.session_state.max_length = st.slider(
                "Макс. длина ответа", 50, 500, 200, 50
            )
            st.session_state.top_k = st.slider("Top-K", 1, 100, 50, 1)
            st.session_state.top_p = st.slider("Top-P", 0.1, 1.0, 0.9, 0.1)
            
            # Настройки RAG
            st.subheader("🔍 Настройки RAG")
            st.session_state.use_rag = st.checkbox("Использовать RAG", value=True)
            st.session_state.rag_top_k = st.slider(
                "Количество документов для RAG", 1, 10, 3, 1
            )
            
            # Информация о системе
            st.subheader("ℹ️ Информация")
            st.markdown("""
            **Модель:** Qwen2.5-1.5B + QLoRA  
            **Языки:** Казахский, Русский, Английский  
            **Технологии:** FAISS, BM25, RAG  
            **Версия:** 0.1.0
            """)
            
            # Очистка истории
            if st.button("Очистить историю", type="secondary"):
                st.session_state.messages = []
                st.session_state.search_history = []
                st.rerun()
    
    def check_api_connection(self, api_url: str) -> bool:
        """Проверка подключения к API."""
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def call_api(self, endpoint: str, method: str = "GET", data: dict = None) -> dict:
        """Вызов API эндпоинта."""
        try:
            url = f"{API_URL}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=30)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=30)
            else:
                return {"error": f"Unsupported method: {method}"}
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}
    
    def render_header(self):
        """Отображение заголовка."""
        st.markdown('<h1 class="main-header">🌐 Многоязычная LLM для казахского и русского языков</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>🎯 Назначение</h4>
                <p>Демонстрация возможностей многоязычной языковой модели для низкоресурсных языков</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>🚀 Особенности</h4>
                <p>QLoRA fine-tuning • RAG поиск • Мультиязычная поддержка</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-box">
                <h4>📊 Технологии</h4>
                <p>FAISS • Sentence Transformers • Hugging Face • Streamlit</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_chat_interface(self):
        """Отображение интерфейса чата."""
        st.markdown('<h2 class="sub-header">💬 Чат с моделью</h2>', unsafe_allow_html=True)
        
        # Отображение истории сообщений
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    # Отображение языка
                    lang_badge = self.get_language_badge(message.get("language", "kk"))
                    st.markdown(f'<div style="margin-bottom: 0.5rem;">{lang_badge}</div>', unsafe_allow_html=True)
                    
                    st.markdown(message["content"])
                    
                    # Отображение контекста если есть
                    if "context" in message:
                        with st.expander("📚 Использованный контекст"):
                            for i, ctx in enumerate(message["context"][:3]):
                                st.markdown(f"**Документ {i+1}:**")
                                st.markdown(f"{ctx['text'][:200]}...")
                                st.progress(ctx['score'])
                    
                    # Отображение метрик если есть
                    if "metrics" in message:
                        cols = st.columns(3)
                        cols[0].metric("Время", f"{message['metrics'].get('time', 0):.2f}с")
                        cols[1].metric("Токены", message['metrics'].get('tokens', 0))
                        cols[2].metric("Схожесть", f"{message['metrics'].get('score', 0):.3f}")
        
        # Ввод сообщения
        prompt = st.chat_input(
            self.get_placeholder_text(),
            key="chat_input"
        )
        
        if prompt:
            # Добавление сообщения пользователя
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "language": st.session_state.selected_language,
                "timestamp": datetime.now().isoformat()
            })
            
            # Определение языка если не выбран
            if st.session_state.selected_language == "auto":
                detected_lang = detect_language(prompt)
                if detected_lang:
                    lang = detected_lang
                else:
                    lang = "kk"
            else:
                lang = st.session_state.selected_language
            
            # Генерация ответа
            with st.chat_message("assistant"):
                with st.spinner("🤔 Думаю..."):
                    try:
                        # Вызов API
                        start_time = time.time()
                        
                        response_data = self.call_api("/generate", "POST", {
                            "prompt": prompt,
                            "language": lang,
                            "use_rag": st.session_state.use_rag,
                            "rag_top_k": st.session_state.rag_top_k,
                            "temperature": st.session_state.temperature,
                            "max_length": st.session_state.max_length,
                            "top_p": st.session_state.top_p,
                            "top_k": st.session_state.top_k
                        })
                        
                        generation_time = time.time() - start_time
                        
                        if "error" not in response_data:
                            response = response_data["response"]
                            context_used = response_data.get("context_used", [])
                            
                            # Отображение ответа
                            lang_badge = self.get_language_badge(lang)
                            st.markdown(f'<div style="margin-bottom: 0.5rem;">{lang_badge}</div>', unsafe_allow_html=True)
                            
                            st.markdown(response)
                            
                            # Отображение контекста если есть
                            if context_used:
                                with st.expander("📚 Использованные источники"):
                                    for i, ctx in enumerate(context_used):
                                        st.markdown(f"**Источник {i+1}** (схожесть: {ctx['score']:.3f}):")
                                        st.markdown(f"{ctx['text'][:300]}...")
                                        st.divider()
                            
                            # Добавление сообщения ассистента
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "language": lang,
                                "context": context_used,
                                "metrics": {
                                    "time": generation_time,
                                    "tokens": response_data.get("tokens_generated", 0),
                                    "score": context_used[0]["score"] if context_used else 0
                                },
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            # Сохранение в историю поиска
                            if context_used:
                                st.session_state.search_history.append({
                                    "query": prompt,
                                    "results": context_used,
                                    "language": lang,
                                    "timestamp": datetime.now().isoformat()
                                })
                        
                        else:
                            st.error(f"Ошибка API: {response_data['error']}")
                    
                    except Exception as e:
                        st.error(f"Ошибка генерации: {str(e)}")
    
    def get_placeholder_text(self) -> str:
        """Получение текста placeholder для чата."""
        lang = st.session_state.selected_language
        
        if lang == "kk":
            return "Қазақша сұрақ қойыңыз..."
        elif lang == "ru":
            return "Задайте вопрос на русском..."
        else:
            return "Ask a question in English..."
    
    def get_language_badge(self, language: str) -> str:
        """Получение HTML для бейджа языка."""
        badges = {
            "kk": '<span class="language-badge kazakh-badge">Қазақша</span>',
            "ru": '<span class="language-badge russian-badge">Русский</span>',
            "en": '<span class="language-badge english-badge">English</span>'
        }
        return badges.get(language, '<span class="language-badge">Unknown</span>')
    
    def render_search_interface(self):
        """Отображение интерфейса поиска."""
        st.markdown('<h2 class="sub-header">🔍 Поиск документов</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Поисковый запрос",
                placeholder="Введите запрос для поиска в документах..."
            )
        
        with col2:
            search_lang = st.selectbox(
                "Язык запроса",
                ["kk", "ru", "en"],
                format_func=lambda x: {
                    "kk": "Қазақша",
                    "ru": "Русский",
                    "en": "English"
                }[x],
                key="search_lang"
            )
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            top_k = st.number_input("Количество результатов", 1, 20, 5, 1)
        
        with col4:
            score_threshold = st.slider("Порог схожести", 0.0, 1.0, 0.5, 0.05)
        
        with col5:
            if st.button("🔍 Поиск", type="primary", use_container_width=True):
                if search_query:
                    self.perform_search(search_query, search_lang, top_k, score_threshold)
                else:
                    st.warning("Введите поисковый запрос")
    
    def perform_search(self, query: str, language: str, top_k: int, score_threshold: float):
        """Выполнение поиска."""
        with st.spinner("🔍 Ищу документы..."):
            try:
                response_data = self.call_api("/search", "POST", {
                    "query": query,
                    "language": language,
                    "top_k": top_k
                })
                
                if "error" not in response_data:
                    results = response_data["results"]
                    
                    # Фильтрация по порогу схожести
                    filtered_results = [r for r in results if r["score"] >= score_threshold]
                    
                    if filtered_results:
                        st.success(f"Найдено {len(filtered_results)} результатов")
                        
                        # Отображение результатов
                        for i, result in enumerate(filtered_results):
                            with st.expander(f"📄 Результат {i+1} (схожесть: {result['score']:.3f})", expanded=i==0):
                                st.markdown(f"**Текст:**")
                                st.markdown(result["text"])
                                
                                if result.get("metadata"):
                                    st.markdown(f"**Метаданные:**")
                                    st.json(result["metadata"])
                        
                        # Сохранение в историю
                        st.session_state.search_history.append({
                            "query": query,
                            "results": filtered_results,
                            "language": language,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    else:
                        st.warning("Не найдено документов, соответствующих критериям")
                
                else:
                    st.error(f"Ошибка поиска: {response_data['error']}")
                    
            except Exception as e:
                st.error(f"Ошибка при поиске: {str(e)}")
    
    def render_document_management(self):
        """Отображение управления документами."""
        st.markdown('<h2 class="sub-header">📄 Управление документами</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📤 Добавить", "📊 Просмотр", "⚙️ Настройки"])
        
        with tab1:
            self.render_add_document_tab()
        
        with tab2:
            self.render_view_documents_tab()
        
        with tab3:
            self.render_document_settings_tab()
    
    def render_add_document_tab(self):
        """Вкладка добавления документа."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            doc_text = st.text_area(
                "Текст документа",
                height=200,
                placeholder="Введите текст документа..."
            )
        
        with col2:
            doc_lang = st.selectbox(
                "Язык документа",
                ["kk", "ru", "en"],
                format_func=lambda x: {
                    "kk": "Қазақша",
                    "ru": "Русский", 
                    "en": "English"
                }[x]
            )
            
            chunk_size = st.number_input("Размер чанка", 128, 1024, 512, 64)
            chunk_overlap = st.number_input("Перекрытие чанков", 0, 256, 50, 10)
            
            if st.button("📥 Добавить документ", type="primary", use_container_width=True):
                if doc_text:
                    self.add_document(doc_text, doc_lang, chunk_size, chunk_overlap)
                else:
                    st.warning("Введите текст документа")
        
        # Загрузка из файла
        st.markdown("---")
        st.markdown("**Или загрузите файл**")
        
        uploaded_file = st.file_uploader(
            "Выберите файл",
            type=["txt", "pdf", "docx", "json", "jsonl"],
            help="Поддерживаемые форматы: TXT, PDF, DOCX, JSON, JSONL"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "text/plain":
                    text = uploaded_file.getvalue().decode("utf-8")
                elif uploaded_file.type == "application/json":
                    data = json.loads(uploaded_file.getvalue().decode("utf-8"))
                    text = data.get("text", str(data))
                else:
                    # Для бинарных форматов используем простой текст
                    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                
                st.text_area("Текст из файла", text[:1000] + ("..." if len(text) > 1000 else ""), height=150)
                
                if st.button("📥 Добавить из файла", type="secondary"):
                    self.add_document(text, doc_lang, chunk_size, chunk_overlap)
                    
            except Exception as e:
                st.error(f"Ошибка чтения файла: {str(e)}")
    
    def add_document(self, text: str, language: str, chunk_size: int, chunk_overlap: int):
        """Добавление документа."""
        with st.spinner("Добавляю документ..."):
            try:
                response_data = self.call_api("/documents", "POST", {
                    "text": text,
                    "language": language,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                })
                
                if "error" not in response_data:
                    st.success(f"✅ Документ добавлен! Создано {response_data['chunks']} чанков")
                    st.session_state.documents.append({
                        "text": text[:100] + "...",
                        "language": language,
                        "chunks": response_data['chunks'],
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    st.error(f"Ошибка добавления: {response_data['error']}")
                    
            except Exception as e:
                st.error(f"Ошибка при добавлении документа: {str(e)}")
    
    def render_view_documents_tab(self):
        """Вкладка просмотра документов."""
        # Статистика
        response_data = self.call_api("/stats", "GET")
        
        if "error" not in response_data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Всего документов", response_data.get("total_documents", 0))
            
            with col2:
                st.metric("Всего чанков", response_data.get("total_chunks", 0))
            
            with col3:
                st.metric("Размер индекса", response_data.get("index_size", 0))
            
            with col4:
                st.metric("Размерность", response_data.get("embedding_dim", 0))
        
        # Экспорт данных
        st.markdown("---")
        st.markdown("**Экспорт данных**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Экспорт в JSON", use_container_width=True):
                self.export_data("json")
        
        with col2:
            if st.button("📥 Экспорт в CSV", use_container_width=True):
                self.export_data("csv")
    
    def export_data(self, format: str):
        """Экспорт данных."""
        try:
            response_data = self.call_api(f"/export/{format}", "GET")
            
            if format == "json":
                st.download_button(
                    label="📥 Скачать JSON",
                    data=json.dumps(response_data, ensure_ascii=False, indent=2),
                    file_name=f"documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            elif format == "csv":
                st.download_button(
                    label="📥 Скачать CSV",
                    data=response_data,
                    file_name=f"documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Ошибка экспорта: {str(e)}")
    
    def render_document_settings_tab(self):
        """Вкладка настроек документов."""
        st.markdown("### Очистка данных")
        
        if st.button("🗑️ Очистить все документы", type="secondary"):
            st.warning("Это действие невозможно отменить!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Да, очистить", type="primary"):
                    st.info("Функция очистки в разработке")
            
            with col2:
                if st.button("❌ Нет, отмена"):
                    st.rerun()
    
    def render_analytics(self):
        """Отображение аналитики."""
        st.markdown('<h2 class="sub-header">📊 Аналитика</h2>', unsafe_allow_html=True)
        
        if not st.session_state.search_history and not st.session_state.messages:
            st.info("Нет данных для анализа. Начните общаться с моделью или искать документы.")
            return
        
        tab1, tab2, tab3 = st.tabs(["📈 Поиск", "💬 Чат", "🌐 Языки"])
        
        with tab1:
            self.render_search_analytics()
        
        with tab2:
            self.render_chat_analytics()
        
        with tab3:
            self.render_language_analytics()
    
    def render_search_analytics(self):
        """Аналитика поиска."""
        if not st.session_state.search_history:
            st.info("Нет истории поиска")
            return
        
        # Преобразование в DataFrame
        search_data = []
        for item in st.session_state.search_history:
            search_data.append({
                "Запрос": item["query"][:50] + ("..." if len(item["query"]) > 50 else ""),
                "Язык": item["language"],
                "Результаты": len(item["results"]),
                "Лучшая схожесть": max([r["score"] for r in item["results"]]) if item["results"] else 0,
                "Время": item["timestamp"]
            })
        
        df = pd.DataFrame(search_data)
        
        # Отображение таблицы
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Графики
        col1, col2 = st.columns(2)
        
        with col1:
            # Распределение по языкам
            lang_counts = df["Язык"].value_counts()
            fig1 = px.pie(
                values=lang_counts.values,
                names=lang_counts.index,
                title="Распределение запросов по языкам",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Распределение схожести
            fig2 = px.histogram(
                df, x="Лучшая схожесть",
                title="Распределение лучшей схожести",
                nbins=20,
                color_discrete_sequence=["#4CAF50"]
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    def render_chat_analytics(self):
        """Аналитика чата."""
        if not st.session_state.messages:
            st.info("Нет истории чата")
            return
        
        # Фильтрация сообщений ассистента
        assistant_messages = [m for m in st.session_state.messages if m["role"] == "assistant"]
        
        if not assistant_messages:
            st.info("Нет сообщений от ассистента")
            return
        
        # Преобразование в DataFrame
        chat_data = []
        for msg in assistant_messages:
            chat_data.append({
                "Язык": msg.get("language", "kk"),
                "Длина ответа": len(msg["content"]),
                "Время генерации": msg.get("metrics", {}).get("time", 0),
                "Использован контекст": "context" in msg,
                "Время": msg.get("timestamp", "")
            })
        
        df = pd.DataFrame(chat_data)
        
        # Графики
        col1, col2 = st.columns(2)
        
        with col1:
            # Время генерации
            fig1 = px.scatter(
                df, x=df.index, y="Время генерации",
                title="Время генерации по ответам",
                color="Язык",
                size="Длина ответа"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Использование контекста
            context_usage = df["Использован контекст"].value_counts()
            fig2 = px.bar(
                x=["С контекстом", "Без контекста"],
                y=[context_usage.get(True, 0), context_usage.get(False, 0)],
                title="Использование контекста",
                color_discrete_sequence=["#2196F3", "#FF9800"]
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    def render_language_analytics(self):
        """Аналитика по языкам."""
        # Сбор данных из всех источников
        lang_data = {"kk": 0, "ru": 0, "en": 0}
        
        # Из истории поиска
        for item in st.session_state.search_history:
            lang = item["language"]
            if lang in lang_data:
                lang_data[lang] += 1
        
        # Из истории чата
        for msg in st.session_state.messages:
            lang = msg.get("language", "kk")
            if lang in lang_data:
                lang_data[lang] += 1
        
        # Создание DataFrame
        df = pd.DataFrame({
            "Язык": ["Қазақша", "Русский", "English"],
            "Код": ["kk", "ru", "en"],
            "Количество": [lang_data["kk"], lang_data["ru"], lang_data["en"]]
        })
        
        # Отображение
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(df[["Язык", "Количество"]], use_container_width=True, hide_index=True)
        
        with col2:
            fig = px.bar(
                df, x="Язык", y="Количество",
                title="Использование языков",
                color="Код",
                color_discrete_map={
                    "kk": "#4CAF50",
                    "ru": "#2196F3",
                    "en": "#FF9800"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_footer(self):
        """Отображение подвала."""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔗 Ссылки**")
            st.markdown("[GitHub](https://github.com) • [Документация](https://example.com)")
        
        with col2:
            st.markdown("**📄 Лицензия**")
            st.markdown("MIT License © 2024")
        
        with col3:
            st.markdown("**👥 Контакты**")
            st.markdown("your.email@example.com")
    
    def run(self):
        """Запуск демо приложения."""
        # Заголовок
        self.render_header()
        
        # Проверка подключения к API
        if not st.session_state.api_connected:
            st.warning("⚠️ Не подключено к API. Проверьте подключение в настройках.")
        
        # Основные вкладки
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 Чат", 
            "🔍 Поиск", 
            "📄 Документы", 
            "📊 Аналитика"
        ])
        
        with tab1:
            self.render_chat_interface()
        
        with tab2:
            self.render_search_interface()
        
        with tab3:
            self.render_document_management()
        
        with tab4:
            self.render_analytics()
        
        # Подвал
        self.render_footer()


def main():
    """Основная функция."""
    try:
        demo = MultilingualDemo()
        demo.run()
        
    except Exception as e:
        st.error(f"Критическая ошибка: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)


if __name__ == "__main__":
    main()