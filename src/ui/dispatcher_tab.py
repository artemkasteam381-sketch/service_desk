import streamlit as st

from src.classification.ui import render_classification_panel


def render_dispatcher_tab():
    """Форма категоризации с кешированием модели и отображением рекомендаций."""
    st.subheader("Помощь диспетчеру")
    # Панель собирает описание и поля регистрации, вызывает RequestClassifier
    # и показывает рекомендуемый вид запроса, топ-3 и сравнение с текущим видом.
    render_classification_panel()
