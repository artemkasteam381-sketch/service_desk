"""Выбор карточки из Excel и существующая форма классификации.

Заменяет src/ui/dispatcher_tab.py. Требует установленный комплект src/classification.
История хранится в памяти пользовательской сессии, исходный Excel не изменяется.
"""
from hashlib import sha256
from io import BytesIO
from numbers import Real

import pandas as pd
import streamlit as st
import re

from src.classification.ui import render_classification_panel


# Те же регистрационные поля, что в текущем экспорте классификатора.
REGISTRATION_FIELDS = (
    'Пользователь', 'Услуга', 'Компонент услуги 1 уровня', 'Тип запроса',
    'Критичность', 'Срочность', 'Приоритет', 'Способ обращения',
    'Класс обслуживания', 'Часовой пояс запроса',
    'Норматив ное время обработки (SLA)', 'Плановое время выполнения',
    'Крайний срок обработки',
)
_HISTORY = 'dispatcher_history_data'
_ACTIVE = 'dispatcher_active_card'


_MISSING_VALUES = {
    '',
    '-',
    '--',
    '---',
    '--)',
    '---)',
    'nan',
    'none',
    'null',
    'не указано',
    'не указан',
    'не определено',
    'не определена',
}


def _value(value):
    if value is None or pd.isna(value):
        return ''

    text = str(value).strip()

    if text.casefold() in _MISSING_VALUES:
        return ''

    if re.fullmatch(r'[-–—]+\)?', text):
        return ''

    return text


def _ticket_number(value):
    if value is None or pd.isna(value):
        return ''
    # Числовые номера Excel показываем без .0; текстовые ведущие нули сохраняем.
    if isinstance(value, Real) and not isinstance(value, bool) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _description(record):
    return _value(record.get('Описание 2')) or _value(record.get('Описание'))


def _read_history(content, sheet_name):
    frame = pd.read_excel(BytesIO(content), sheet_name=sheet_name, dtype=object,
                          keep_default_na=False, engine='openpyxl')
    frame.columns = [str(name).strip() for name in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError('В таблице повторяются названия колонок.')
    if 'Номер запроса' not in frame:
        raise ValueError('Нужна колонка «Номер запроса».')
    if not {'Описание', 'Описание 2'}.intersection(frame.columns):
        raise ValueError('Нужна колонка «Описание» или «Описание 2».')
    if frame.empty:
        raise ValueError('На выбранном листе нет обращений.')
    return frame.reset_index(drop=True)


def _fill_form(record):
    # Выполнять ДО создания виджетов. Обнуляем отсутствующие поля предыдущей записи.
    st.session_state['classification_description'] = _description(record)
    st.session_state['classification_current_kind'] = _value(record.get('Вид запроса'))
    for field in REGISTRATION_FIELDS:
        st.session_state['classification_' + field] = _value(record.get(field))


def _source_record(record):
    with st.expander('Открыть исходную запись обращения'):
        st.caption('Значения из загруженного файла. Правки в форме ниже не изменяют исходную запись.')
        text_fields = ('Описание', 'Описание 2', 'Описание решения', 'Результат работ')
        other = [{'Поле': str(field), 'Значение': _value(value)}
                 for field, value in record.items() if field not in text_fields]
        st.dataframe(other, hide_index=True, width='stretch',
                     column_config={'Значение': st.column_config.TextColumn(width='large')})
        for field in text_fields:
            if field in record:
                st.text(field)
                st.text(_value(record[field]) or 'Не заполнено')


def _choose_history_record():
    generation = st.session_state.get('dispatcher_upload_generation', 0)
    uploaded = st.file_uploader('Загрузить историю обращений', type=['xlsx'],
                                key=f'dispatcher_history_upload_{generation}')
    if uploaded is not None:
        content = uploaded.getvalue()
        digest = sha256(content).hexdigest()
        existing = st.session_state.get(_HISTORY)
        if existing is None or existing['digest'] != digest:
            try:
                with pd.ExcelFile(BytesIO(content), engine='openpyxl') as workbook:
                    sheets = list(workbook.sheet_names)
                if not sheets:
                    raise ValueError('В файле нет листов.')
            except Exception:
                st.error('Не удалось прочитать Excel. Выберите корректный файл .xlsx.')
                return None
            st.session_state[_HISTORY] = dict(name=uploaded.name, digest=digest,
                                              content=content, sheets=sheets, sheet=None, frame=None)
    history = st.session_state.get(_HISTORY)
    if history is None:
        st.info('Загрузите Excel с колонками «Номер запроса» и «Описание» или «Описание 2».')
        return None
    st.caption(f"Загружена история: {history['name']}. Она доступна в текущей сессии.")
    if st.button('Очистить историю', key='dispatcher_clear_history'):
        st.session_state.pop(_HISTORY, None)
        st.session_state.pop(_ACTIVE, None)
        st.session_state.pop('dispatcher_selected_ticket', None)
        st.session_state.pop('dispatcher_registration_card', None)
        st.session_state['dispatcher_upload_generation'] = generation + 1
        st.rerun()
    if len(history['sheets']) > 1:
        sheet = st.selectbox('Лист Excel', history['sheets'],
                             key='dispatcher_sheet_' + history['digest'])
    else:
        sheet = history['sheets'][0]
    if history['sheet'] != sheet:
        try:
            frame = _read_history(history['content'], sheet)
        except ValueError as exc:
            st.error(str(exc))
            return None
        except Exception:
            st.error('Не удалось загрузить выбранный лист Excel.')
            return None
        history.update(sheet=sheet, frame=frame)
    frame = history['frame']
    numbers = frame['Номер запроса'].map(_ticket_number)
    positions = frame.index[numbers.ne('')].tolist()
    if not positions:
        st.warning('На этом листе нет заполненных номеров обращений.')
        return None
    if len(positions) != len(frame):
        st.warning(f'Строк без номера: {len(frame) - len(positions)}. Они недоступны для выбора.')
    duplicate = numbers.duplicated(keep=False)
    labels = {i: numbers.iloc[i] + (f' — строка {i + 2}' if duplicate.iloc[i] else '') for i in positions}
    if duplicate[numbers.ne('')].any():
        st.caption('Повторяющиеся номера различаются номером строки Excel.')
    key = 'dispatcher_ticket_' + sha256((history['digest'] + '\0' + sheet).encode()).hexdigest()[:20]
    position = st.selectbox('Номер обращения', positions, index=None, format_func=labels.get,
                            placeholder='Начните вводить номер обращения', key=key)
    st.caption(f'Доступно обращений: {len(positions)}.')
    if position is None:
        return None
    record = frame.iloc[position].to_dict()
    token = (history['digest'], sheet, int(position))
    missing = [field for field in REGISTRATION_FIELDS if not _value(record.get(field))]
    if missing:
        st.info('В выбранной записи не заполнены поля: ' + ', '.join(missing) + '. Их можно дополнить ниже.')
    _source_record(record)
    restore = st.button('Восстановить поля из истории', key='dispatcher_restore_card')
    return token, record, restore


def render_dispatcher_tab():
    st.subheader('Помощь диспетчеру')
    mode = st.radio('Источник обращения', ['Новое обращение', 'Из истории'],
                    horizontal=True, key='dispatcher_input_mode')
    if mode == 'Из истории':
        selection = _choose_history_record()
        if selection is None:
            # Не оставляем доступной предыдущую карточку при пустом/ошибочном выборе.
            st.session_state.pop(_ACTIVE, None)
            st.session_state.pop('dispatcher_selected_ticket', None)
            st.session_state.pop('dispatcher_registration_card', None)
            return
        token, record, restore = selection
        if restore or st.session_state.get(_ACTIVE) != token:
            _fill_form(record)
        st.session_state[_ACTIVE] = token
        st.session_state['dispatcher_selected_ticket'] = record.copy()
        st.caption('Поля заполнены из истории. При необходимости исправьте их перед анализом.')
    else:
        if st.session_state.get(_ACTIVE) != ('new',):
            _fill_form({})
        st.session_state[_ACTIVE] = ('new',)
        st.session_state.pop('dispatcher_selected_ticket', None)

    # Существующая панель кеширует модель, использует whitelist признаков и выводит топ-3.
    render_classification_panel()
    # Общая регистрационная карточка для будущего подключения других модулей.
    # Решение, фактические сроки и исполнители остаются только в исходной записи.
    card = {field: st.session_state.get('classification_' + field, '') for field in REGISTRATION_FIELDS}
    card['Описание'] = st.session_state.get('classification_description', '')
    card['Вид запроса'] = st.session_state.get('classification_current_kind', '')
    selected = st.session_state.get('dispatcher_selected_ticket')
    if selected is not None:
        card['Номер запроса'] = _ticket_number(selected.get('Номер запроса'))
    st.session_state['dispatcher_registration_card'] = card
