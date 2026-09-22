"""Самостоятельная проверка блока до встраивания в существующий app.py."""
import streamlit as st
from src.classification.ui import render_classification_panel

st.set_page_config(page_title='Категоризация обращений', page_icon='📨', layout='wide')
st.title('Помощник диспетчера')
render_classification_panel()
