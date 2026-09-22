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
from src.integrations.support_service import SupportServiceError, predict_support


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
_SUPPORT_RESULT = 'dispatcher_support_result'
_SUPPORT_FINGERPRINT = 'dispatcher_support_fingerprint'


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


def _card_fingerprint(card):
    """Не показывать результат, рассчитанный для уже изменённой карточки."""
    payload = '\0'.join(str(card.get(key, '')) for key in (
        'Описание', 'Услуга', 'Тип запроса',
        'Компонент услуги 1 уровня', 'Пользователь',
    ))
    return sha256(payload.encode('utf-8')).hexdigest()


def _render_support_result(result):
    line = result.get('recommended_line')
    scores = result.get('scores') or {}

    if line:
        confidence = scores.get(line)
        if isinstance(confidence, Real):
            st.success(f'Рекомендуемая линия: {line} · оценка {float(confidence):.1%}')
        else:
            st.success(f'Рекомендуемая линия: {line}')
    else:
        st.warning('Сервис не смог подобрать линию поддержки.')

    if scores:
        score_rows = [
            {'Линия': str(name), 'Оценка модели': float(score)}
            for name, score in scores.items() if isinstance(score, Real)
        ]
        score_rows.sort(key=lambda row: row['Оценка модели'], reverse=True)
        with st.expander('Оценки всех линий'):
            st.dataframe(score_rows, hide_index=True, width='stretch',
                         column_config={
                             'Оценка модели': st.column_config.ProgressColumn(
                                 min_value=0.0, max_value=1.0, format='%.3f'
                             )
                         })

    for note in result.get('notes') or []:
        if str(note).strip():
            st.caption(str(note))

    st.markdown('#### Похожие обращения')
    analogs = result.get('analogs') or []
    if not analogs:
        st.info('Похожие обращения не найдены.')
        return

    for index, analog in enumerate(analogs, start=1):
        ticket_id = analog.get('id', 'без номера')
        analog_line = analog.get('line') or 'линия не указана'
        with st.expander(f'{index}. Обращение {ticket_id} · {analog_line}',
                         expanded=index == 1):
            if analog.get('category'):
                st.write('**Вид обращения:**', analog['category'])
            st.write(analog.get('text') or 'Описание отсутствует.')
            st.write('**Результат работ:**', analog.get('result') or 'Не указан.')
            score = analog.get('score')
            if isinstance(score, Real):
                st.caption(f'Оценка сходства cross-encoder: {float(score):.3f}')
            if analog.get('source_url'):
                st.markdown(f"[Открыть исходную запись]({analog['source_url']})")


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

    st.divider()
    st.subheader('Маршрутизация и похожие обращения')
    st.caption('Линия определяется по описанию и регистрационным полям; аналоги — по смысловой близости текста.')

    fingerprint = _card_fingerprint(card)
    if st.button('Рекомендовать линию и найти похожие',
                 type='primary', key='dispatcher_run_support'):
        metadata = {
            field: card.get(field, '')
            for field in ('Услуга', 'Тип запроса', 'Компонент услуги 1 уровня', 'Пользователь')
        }
        try:
            with st.spinner('Подбираем линию и похожие обращения…'):
                result = predict_support(card.get('Описание', ''), metadata)
        except SupportServiceError as exc:
            st.session_state.pop(_SUPPORT_RESULT, None)
            st.session_state.pop(_SUPPORT_FINGERPRINT, None)
            st.error(str(exc))
        else:
            st.session_state[_SUPPORT_RESULT] = result
            st.session_state[_SUPPORT_FINGERPRINT] = fingerprint

    if (st.session_state.get(_SUPPORT_RESULT) is not None
            and st.session_state.get(_SUPPORT_FINGERPRINT) == fingerprint):
        _render_support_result(st.session_state[_SUPPORT_RESULT])
