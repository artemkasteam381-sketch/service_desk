from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .service import RequestClassifier
from .settings import load_settings


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Пути и справочники
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_DATA_CANDIDATES = (
    PROJECT_ROOT / "data" / "processed" / "appeals_parsed.csv",
    PROJECT_ROOT / "data" / "raw" / "original.xlsx",
)


# Поля, которые показываем в основной части формы.
PRIMARY_FIELDS = (
    "Услуга",
    "Тип запроса",
    "Приоритет",
)


# Порядок дополнительных полей.
ADDITIONAL_FIELD_ORDER = (
    "Пользователь",
    "Компонент услуги 1 уровня",
    "Критичность",
    "Срочность",
    "Способ обращения",
    "Класс обслуживания",
    "Часовой пояс запроса",
    "Норматив ное время обработки (SLA)",
    "Плановое время выполнения",
    "Крайний срок обработки",
)


# Категориальные поля, для которых желательно показывать selectbox.
CATEGORICAL_FIELDS = {
    "Услуга",
    "Тип запроса",
    "Приоритет",
    "Компонент услуги 1 уровня",
    "Критичность",
    "Срочность",
    "Способ обращения",
    "Класс обслуживания",
    "Часовой пояс запроса",
}


# Возможные названия колонок в исходном/обработанном датасете.
SOURCE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "Услуга": (
        "Услуга",
        "service",
    ),
    "Тип запроса": (
        "Тип запроса",
        "request_type_raw",
    ),
    "Приоритет": (
        "Приоритет",
        "priority",
    ),
    "Компонент услуги 1 уровня": (
        "Компонент услуги 1 уровня",
    ),
    "Критичность": (
        "Критичность",
    ),
    "Срочность": (
        "Срочность",
    ),
    "Способ обращения": (
        "Способ обращения",
    ),
    "Класс обслуживания": (
        "Класс обслуживания",
    ),
    "Часовой пояс запроса": (
        "Часовой пояс запроса",
    ),

    # Это НЕ входной признак модели.
    # Справочник нужен только для выбора уже установленного вида обращения.
    "Текущий вид обращения": (
        "Вид запроса",
        "Вид обращения",
        "request_type",
    ),
}


FIELD_LABELS = {
    "Норматив ное время обработки (SLA)":
        "Нормативное время обработки (SLA)",
}


# ---------------------------------------------------------------------
# Загрузка ML-модели
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _cached_classifier(
    model_dir: str,
    device: str,
    batch_size: int,
    export_timestamp: int,
) -> RequestClassifier:
    """
    Загружает классификатор один раз.

    export_timestamp входит в ключ кеша, поэтому после замены экспорта
    модели Streamlit создаст новый экземпляр классификатора.
    """
    return RequestClassifier(
        model_dir,
        device=device,
        batch_size=batch_size,
    )


def _load_classifier() -> RequestClassifier | None:
    try:
        settings = load_settings()

        manifest_path = settings.model_dir / "model_manifest.json"
        timestamp = manifest_path.stat().st_mtime_ns

        with st.spinner("Загрузка модели классификации…"):
            return _cached_classifier(
                str(settings.model_dir),
                settings.device,
                settings.batch_size,
                timestamp,
            )

    except Exception:
        logger.exception("Не удалось загрузить модель категоризации")

        st.error(
            "Не удалось загрузить модель классификации. "
            "Проверьте файлы экспорта и зависимости ML-модуля."
        )
        return None


# ---------------------------------------------------------------------
# Загрузка справочников из исторических данных
# ---------------------------------------------------------------------

def _reference_source() -> Path | None:
    for path in REFERENCE_DATA_CANDIDATES:
        if path.exists():
            return path

    return None


@st.cache_data(show_spinner=False)
def _load_reference_dataframe(
    path_string: str,
    modified_ns: int,
) -> pd.DataFrame:
    """
    Загружает исторические данные только для построения dropdown-справочников.

    modified_ns используется для автоматического обновления кеша при
    замене файла данных.
    """
    path = Path(path_string)

    try:
        if path.suffix.lower() == ".xlsx":
            return pd.read_excel(path)

        # Сначала обычный UTF-8 CSV.
        try:
            df = pd.read_csv(
                path,
                encoding="utf-8-sig",
                low_memory=False,
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                path,
                encoding="cp1251",
                low_memory=False,
            )

        # Если CSV оказался с другим разделителем и pandas увидел одну колонку.
        if len(df.columns) == 1:
            df = pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding="utf-8-sig",
            )

        return df

    except Exception:
        logger.exception(
            "Не удалось загрузить справочник значений из %s",
            path,
        )
        return pd.DataFrame()


def _natural_sort_key(value: str) -> list[Any]:
    """
    Сортирует значения вроде:
    (1) Критический
    (2) Высокий
    (3) Средний
    (4) Низкий
    в естественном порядке.
    """
    parts = re.split(r"(\d+)", value)

    return [
        int(part) if part.isdigit() else part.casefold()
        for part in parts
    ]


EMPTY_CATEGORY_VALUES = {
    "",
    "-",
    "--",
    "---",
    "--)",
    "---)",
    "nan",
    "none",
    "null",
    "не определена",
    "не определено",
    "не указан",
    "не указано",
}


def _is_missing_category_value(value: str) -> bool:
    """
    Определяет технические заглушки исходных данных.

    В UI вместо них используется единый вариант «Не указано».
    """
    text = str(value).strip()

    if not text:
        return True

    if text.casefold() in EMPTY_CATEGORY_VALUES:
        return True

    # Ловим варианты вроде -, --, ---, --), ---) и т.п.
    if re.fullmatch(r"[-–—]+\)?", text):
        return True

    return False


def _clean_category_values(series: pd.Series) -> list[str]:
    """
    Формирует чистый список значений для dropdown.

    Технические пропуски не попадают в список:
    пользователь видит только единый вариант «Не указано».
    """
    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    cleaned = [
        value
        for value in values
        if not _is_missing_category_value(value)
    ]

    return sorted(
        set(cleaned),
        key=_natural_sort_key,
    )


def _reference_options() -> dict[str, list[str]]:
    """
    Возвращает справочники значений для категориальных полей.

    Например:
        {
            "Услуга": [...],
            "Тип запроса": [...],
            "Приоритет": [...],
            "Текущий вид обращения": [...]
        }
    """
    source = _reference_source()

    if source is None:
        return {}

    try:
        modified_ns = source.stat().st_mtime_ns
    except OSError:
        return {}

    df = _load_reference_dataframe(
        str(source),
        modified_ns,
    )

    if df.empty:
        return {}

    options: dict[str, list[str]] = {}

    for field, aliases in SOURCE_COLUMN_ALIASES.items():
        source_column = next(
            (
                alias
                for alias in aliases
                if alias in df.columns
            ),
            None,
        )

        if source_column is None:
            continue

        values = _clean_category_values(df[source_column])

        if values:
            options[field] = values

    return options


# ---------------------------------------------------------------------
# Общие UI helpers
# ---------------------------------------------------------------------

def _field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field)


def _clean_value(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if _is_missing_category_value(text):
        return None

    return text


def _select_value(
    label: str,
    options: list[str],
    *,
    key: str,
    help_text: str | None = None,
) -> str | None:
    """
    Dropdown с пустым вариантом.
    """
    options = list(options)

    current_value = st.session_state.get(key)

    if (
        current_value
        and current_value != "Не указано"
        and current_value not in options
    ):
        options = [current_value, *options]
        
    selected = st.selectbox(
        label,
        ["Не указано", *options],
        index=0,
        key=key,
        help=help_text,
    )

    return _clean_value(selected)


def _render_input_field(
    field: str,
    options: dict[str, list[str]],
    *,
    key: str,
    help_text: str | None = None,
) -> str | None:
    """
    Категориальные поля показываются через selectbox,
    если для них найден справочник.

    Если справочника нет — безопасный fallback на text_input.
    """
    label = _field_label(field)

    if field in CATEGORICAL_FIELDS:
        field_options = options.get(field, [])

        if field_options:
            return _select_value(
                label,
                field_options,
                key=key,
                help_text=help_text,
            )

    return _clean_value(
        st.text_input(
            label,
            key=key,
            help=help_text,
        )
    )


def _score_text(score: Any) -> str:
    try:
        return f"{float(score):.3f}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------
# Отображение результата
# ---------------------------------------------------------------------

def _render_comparison(result: dict[str, Any]) -> None:
    comparison = result.get("comparison")
    current_kind = result.get("current_kind")
    recommended_kind = result.get("recommended_kind")

    if current_kind:
        left, right = st.columns(2)

        with left:
            st.caption("Текущий вид обращения")
            st.markdown(f"**{current_kind}**")

        with right:
            st.caption("Рекомендация модели")
            st.markdown(
                f"**{recommended_kind or 'Не определено'}**"
            )

    if comparison == "matches":
        st.success(
            "Текущий вид обращения совпадает "
            "с рекомендацией модели."
        )

    elif comparison == "differs":
        st.warning(
            "Модель предлагает другой вид обращения. "
            "Проверьте текущую категоризацию."
        )

    elif comparison == "outside_top15":
        st.info(
            "Модель отнесла обращение к категории "
            "«Вне топ-15». Точный вид обращения "
            "в этом случае модель не определяет."
        )

    elif comparison == "not_supplied":
        st.caption(
            "Текущий вид обращения не указан — "
            "сравнение не выполнялось."
        )


def _render_top3(top3: list[dict[str, Any]]) -> None:
    if not top3:
        return

    st.markdown("#### Top-3 вариантов")

    for index, item in enumerate(top3[:3], start=1):
        label = item.get("label") or "Без названия"
        score = _score_text(item.get("score"))

        with st.container(border=True):
            left, right = st.columns(
                [0.78, 0.22],
                vertical_alignment="center",
            )

            with left:
                st.markdown(
                    f"**{index}. {label}**"
                )

            with right:
                st.caption("Оценка модели")
                st.markdown(f"**{score}**")


def _render_missing_fields(result: dict[str, Any]) -> None:
    missing_fields = result.get("missing_fields") or []

    if not missing_fields:
        return

    with st.expander(
        f"Не заполнено дополнительных полей: "
        f"{len(missing_fields)}",
        expanded=False,
    ):
        st.caption(
            "Прогноз уже рассчитан. Эти поля не обязательны, "
            "но при наличии их можно передать модели."
        )

        for field in missing_fields:
            st.markdown(
                f"- {_field_label(field)}"
            )


def _render_result(result: dict[str, Any]) -> None:
    recommended_kind = (
        result.get("recommended_kind")
        or "Не удалось определить"
    )

    st.markdown("### Результат анализа")

    with st.container(border=True):
        st.caption("Рекомендуемый вид обращения")
        st.markdown(f"## {recommended_kind}")

        st.divider()

        _render_comparison(result)

        st.divider()

        _render_top3(
            result.get("top3") or []
        )

        _render_missing_fields(result)

        st.caption(
            "Числовая оценка используется для сравнения "
            "вариантов и не является гарантированной вероятностью. "
            "Окончательное решение подтверждает оператор."
        )


# ---------------------------------------------------------------------
# Основная панель классификации
# ---------------------------------------------------------------------

def render_classification_panel() -> None:
    classifier = _load_classifier()

    if classifier is None:
        return

    reference_options = _reference_options()

    # Реальные поля текущего экспорта модели.
    meta_columns = list(
        getattr(classifier, "meta_columns", []) or []
    )

    with st.container(border=True):
        st.markdown("### Новое обращение")

        st.caption(
            "Введите описание и доступные данные карточки. "
            "Модель независимо определит рекомендуемый вид обращения "
            "и сравнит его с текущим, если он указан."
        )

        with st.form(
            "request_classification_form",
            clear_on_submit=False,
        ):
            # ---------------------------------------------------------
            # Описание
            # ---------------------------------------------------------

            description = st.text_area(
                "Описание обращения",
                height=190,
                placeholder=(
                    "Введите текст обращения пользователя…"
                ),
                key="classification_description",
            )

            # ---------------------------------------------------------
            # Основные классифицированные поля
            # ---------------------------------------------------------

            service_col, type_col, priority_col = st.columns(3)

            with service_col:
                service = _render_input_field(
                    "Услуга",
                    reference_options,
                    key="classification_Услуга",
                )

            with type_col:
                request_type = _render_input_field(
                    "Тип запроса",
                    reference_options,
                    key="classification_Тип запроса",
                    help_text=(
                        "Регистрационный признак исходной карточки. "
                        "Это НЕ тот же параметр, что «Вид обращения», "
                        "который определяет модель."
                    ),
                )

            with priority_col:
                priority = _render_input_field(
                    "Приоритет",
                    reference_options,
                    key="classification_Приоритет",
                )

            # ---------------------------------------------------------
            # Уже назначенный вид обращения
            # ---------------------------------------------------------

            current_kind_options = reference_options.get(
                "Текущий вид обращения",
                [],
            )

            if current_kind_options:
                current_kind = _select_value(
                    "Текущий вид обращения",
                    current_kind_options,
                    key="classification_current_kind",
                    help_text=(
                        "Категория, уже установленная в карточке. "
                        "Она НЕ используется как признак модели — "
                        "только сравнивается с независимым прогнозом."
                    ),
                )
            else:
                current_kind = _clean_value(
                    st.text_input(
                        "Текущий вид обращения",
                        key="classification_current_kind",
                        help=(
                            "Категория, уже установленная в карточке. "
                            "Она не используется как признак модели."
                        ),
                    )
                )

            # ---------------------------------------------------------
            # Дополнительные поля
            # ---------------------------------------------------------

            additional_values: dict[str, str | None] = {}

            excluded_fields = set(PRIMARY_FIELDS)

            # Сначала известные поля в понятном порядке.
            additional_fields = [
                field
                for field in ADDITIONAL_FIELD_ORDER
                if field in meta_columns
                and field not in excluded_fields
            ]

            # Если ML-команда когда-нибудь добавит новое поле,
            # оно автоматически появится в конце формы.
            additional_fields.extend(
                field
                for field in meta_columns
                if field not in excluded_fields
                and field not in additional_fields
            )

            if additional_fields:
                with st.expander(
                    "Дополнительные поля",
                    expanded=False,
                ):
                    st.caption(
                        "Заполняйте только данные, которые уже есть "
                        "в карточке обращения. Сохраняйте исходные "
                        "значения системы."
                    )

                    columns = st.columns(2)

                    for index, field in enumerate(
                        additional_fields
                    ):
                        with columns[index % 2]:
                            additional_values[field] = (
                                _render_input_field(
                                    field,
                                    reference_options,
                                    key="classification_" + field,
                                )
                            )

            submitted = st.form_submit_button(
                "Проанализировать обращение",
                type="primary",
                use_container_width=True,
            )

    # -----------------------------------------------------------------
    # Форма ещё не отправлена
    # -----------------------------------------------------------------

    if not submitted:
        return

    # -----------------------------------------------------------------
    # Минимальная валидация
    # -----------------------------------------------------------------

    if not description.strip():
        st.warning(
            "Введите описание обращения."
        )
        return

    # -----------------------------------------------------------------
    # Собираем признаки модели
    # -----------------------------------------------------------------

    appeal: dict[str, Any] = {
        "Описание": description.strip(),
        "Услуга": service,
        "Тип запроса": request_type,
        "Приоритет": priority,
    }

    appeal.update(additional_values)

    # None можно передавать модели, но убираем пустые строки
    # и лишние значения ради чистого payload.
    appeal = {
        field: value
        for field, value in appeal.items()
        if value is not None
    }

    # ВАЖНО:
    # current_kind НЕ добавляем в appeal.
    # Он передаётся отдельно и не участвует в предсказании.
    clean_current_kind = _clean_value(current_kind)

    # -----------------------------------------------------------------
    # Предсказание
    # -----------------------------------------------------------------

    try:
        with st.spinner(
            "Анализ обращения…"
        ):
            result = classifier.predict(
                appeal,
                current_kind=clean_current_kind,
            )

    except ValueError as exc:
        logger.warning(
            "Ошибка валидации классификатора: %s",
            exc,
        )
        st.error(str(exc))
        return

    except Exception:
        logger.exception(
            "Ошибка выполнения модели категоризации"
        )

        st.error(
            "Не удалось получить прогноз модели. "
            "Проверьте журнал приложения."
        )
        return

    # -----------------------------------------------------------------
    # Карточка результата
    # -----------------------------------------------------------------

    _render_result(result)