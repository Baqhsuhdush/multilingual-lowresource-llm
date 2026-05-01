"""
Компоненты для интерфейса чата.
"""

import streamlit as st
from typing import List, Dict, Any
from datetime import datetime


class ChatMessage:
    """Класс для представления сообщения чата."""
    
    def __init__(self, role: str, content: str, language: str = "kk", metadata: Dict[str, Any] = None):
        self.role = role
        self.content = content
        self.language = language
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        return {
            "role": self.role,
            "content": self.content,
            "language": self.language,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """Создание из словаря."""
        message = cls(
            role=data["role"],
            content=data["content"],
            language=data.get("language", "kk"),
            metadata=data.get("metadata", {})
        )
        if "timestamp" in data:
            message.timestamp = datetime.fromisoformat(data["timestamp"])
        return message


class ChatHistory:
    """Класс для управления историей чата."""
    
    def __init__(self, max_messages: int = 100):
        self.max_messages = max_messages
        self.messages: List[ChatMessage] = []
    
    def add_message(self, message: ChatMessage):
        """Добавление сообщения в историю."""
        self.messages.append(message)
        
        # Ограничение размера истории
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def add_user_message(self, content: str, language: str = "kk"):
        """Добавление сообщения пользователя."""
        message = ChatMessage("user", content, language)
        self.add_message(message)
    
    def add_assistant_message(self, content: str, language: str = "kk", metadata: Dict[str, Any] = None):
        """Добавление сообщения ассистента."""
        message = ChatMessage("assistant", content, language, metadata)
        self.add_message(message)
    
    def get_messages(self, last_n: int = None) -> List[ChatMessage]:
        """Получение сообщений."""
        if last_n:
            return self.messages[-last_n:]
        return self.messages
    
    def clear(self):
        """Очистка истории."""
        self.messages.clear()
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Конвертация в список словарей."""
        return [msg.to_dict() for msg in self.messages]
    
    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> "ChatHistory":
        """Создание из списка словарей."""
        history = cls()
        history.messages = [ChatMessage.from_dict(item) for item in data]
        return history


class ChatRenderer:
    """Класс для отображения чата."""
    
    def __init__(self):
        self.language_colors = {
            "kk": "#4CAF50",  # Зеленый
            "ru": "#2196F3",  # Синий
            "en": "#FF9800",  # Оранжевый
        }
    
    def render_message(self, message: ChatMessage):
        """Отображение одного сообщения."""
        with st.chat_message(message.role):
            # Бейдж языка
            self.render_language_badge(message.language)
            
            # Содержимое
            st.markdown(message.content)
            
            # Метаданные
            if message.metadata:
                self.render_metadata(message.metadata)
    
    def render_language_badge(self, language: str):
        """Отображение бейджа языка."""
        language_names = {
            "kk": "Қазақша",
            "ru": "Русский",
            "en": "English"
        }
        
        color = self.language_colors.get(language, "#9E9E9E")
        name = language_names.get(language, language)
        
        html = f"""
        <div style="
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            background-color: {color};
            color: white;
            font-size: 0.875rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        ">
            {name}
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    def render_metadata(self, metadata: Dict[str, Any]):
        """Отображение метаданных."""
        with st.expander("📊 Детали"):
            cols = st.columns(3)
            
            if "generation_time" in metadata:
                cols[0].metric("Время", f"{metadata['generation_time']:.2f}с")
            
            if "tokens" in metadata:
                cols[1].metric("Токены", metadata["tokens"])
            
            if "similarity" in metadata:
                cols[2].metric("Схожесть", f"{metadata['similarity']:.3f}")
            
            # Дополнительная информация
            if "context_used" in metadata:
                st.markdown("**Использованный контекст:**")
                for i, ctx in enumerate(metadata["context_used"][:3]):
                    st.markdown(f"{i+1}. {ctx[:100]}...")
    
    def render_history(self, history: ChatHistory):
        """Отображение всей истории."""
        for message in history.get_messages():
            self.render_message(message)