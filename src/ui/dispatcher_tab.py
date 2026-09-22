import streamlit as st

from src.classification.ui import render_classification_panel


def render_dispatcher_tab():
    st.subheader("Помощь диспетчеру")
    render_classification_panel()