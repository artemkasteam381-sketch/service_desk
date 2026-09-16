import streamlit as st

st.set_page_config(
    page_title="Service Desk — помощник",
    layout="wide",
)

st.title("Помощник Service Desk")
st.caption("Каркас приложения. Модели и данные пока не подключены.")

analysis_tab, history_tab, dashboard_tab = st.tabs(
    ["Разбор обращения", "История", "Аналитика"]
)

with analysis_tab:
    st.text_area("Описание обращения")
    st.info("Здесь появятся категория, линия поддержки и похожие случаи.")

with history_tab:
    st.info("Здесь появятся исторические обращения и исходные записи.")

with dashboard_tab:
    st.info("Здесь появятся показатели обработки и SLA.")