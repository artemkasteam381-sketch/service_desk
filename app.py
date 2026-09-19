import streamlit as st

from src.ui.dispatcher_tab import render_dispatcher_tab
from src.ui.analytics_tab import render_analytics_tab


st.set_page_config(
    page_title="Анализ обращений поддержки",
    page_icon="📊",
    layout="wide",
)


def main():
    st.title("Анализ обращений технической поддержки")

    dispatcher_tab, analytics_tab = st.tabs(
        ["Помощь диспетчеру", "Аналитика обработки"]
    )

    with dispatcher_tab:
        render_dispatcher_tab()

    with analytics_tab:
        render_analytics_tab()


if __name__ == "__main__":
    main()