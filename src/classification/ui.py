import logging
import streamlit as st
from .service import RequestClassifier
from .settings import load_settings

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _cached_classifier(model_dir, device, batch_size, export_timestamp):
    # Кеш общий для сессий; RequestClassifier защищает загрузку и прогноз блокировкой.
    return RequestClassifier(model_dir, device=device, batch_size=batch_size)


def render_classification_panel():
    st.subheader('Определение вида запроса')
    try:
        settings = load_settings()
        timestamp = (settings.model_dir / 'model_manifest.json').stat().st_mtime_ns
        with st.spinner('Загрузка модели…'):
            classifier = _cached_classifier(str(settings.model_dir), settings.device, settings.batch_size, timestamp)
    except Exception:
        logger.exception('Не удалось загрузить модель категоризации')
        st.error('Модель категоризации не подключена. Проверьте установку по инструкции INTEGRATION.md.')
        return

    with st.form('request_classification_form'):
        description = st.text_area('Описание обращения', height=180, key='classification_description')
        current_kind = st.text_input('Текущий вид запроса, если указан', key='classification_current_kind')
        fields = {}
        primary = ['Услуга', 'Компонент услуги 1 уровня', 'Тип запроса', 'Критичность']
        st.caption('Перенесите доступные поля из карточки обращения, сохраняя их исходные значения.')
        columns = st.columns(2)
        for i, field in enumerate(primary):
            fields[field] = columns[i % 2].text_input(field, key='classification_' + field)
        with st.expander('Остальные поля регистрации'):
            columns = st.columns(2)
            others = [field for field in classifier.meta_columns if field not in primary]
            for i, field in enumerate(others):
                label = 'Нормативное время обработки (SLA)' if field == 'Норматив ное время обработки (SLA)' else field
                fields[field] = columns[i % 2].text_input(label, key='classification_' + field)
        submitted = st.form_submit_button('Определить вид запроса', type='primary')

    if not submitted:
        return
    if not description.strip():
        st.warning('Введите описание обращения.')
        return
    try:
        with st.spinner('Анализ обращения…'):
            result = classifier.predict({'Описание': description, **fields}, current_kind=current_kind)
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception:
        logger.exception('Ошибка выполнения модели категоризации')
        st.error('Не удалось получить прогноз. При первом запуске требуется загрузка языковой модели.')
        return

    st.write('**Рекомендуемый вид запроса:**', result['recommended_kind'])
    if result['comparison'] == 'outside_top15':
        st.info('Модель отнесла обращение к видам вне топ-15. Конкретный вид должен определить оператор.')
    elif result['comparison'] == 'matches':
        st.success('Рекомендация совпадает с текущим видом запроса.')
    elif result['comparison'] == 'differs':
        st.info('Рекомендация отличается от текущего вида запроса. Проверьте категорию перед изменением.')
    st.dataframe(
        [{'Категория': item['label'], 'Оценка модели': item['score']} for item in result['top3']],
        hide_index=True, width='stretch',
        column_config={'Оценка модели': st.column_config.NumberColumn(format='%.3f')},
    )
    st.caption('Оценки помогают сравнить варианты. Окончательную категорию подтверждает оператор.')
