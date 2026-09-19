import streamlit as st


def render_dispatcher_tab():
    st.subheader("Помощь диспетчеру")

    st.text_area(
        "Текст обращения",
        placeholder="Введите текст нового обращения...",
        height=180,
    )

    st.button(
        "Проанализировать обращение",
        type="primary",
    )