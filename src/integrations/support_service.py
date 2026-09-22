"""HTTP-клиент сервиса рекомендации линии и поиска похожих обращений."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_URL = "http://127.0.0.1:8765/api/predict"


class SupportServiceError(RuntimeError):
    """Понятная пользователю ошибка обращения к ML-сервису."""


def predict_support(
    text: str,
    metadata: Mapping[str, Any] | None = None,
    *,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Вернуть рекомендацию линии и похожие обращения.

    Адрес можно переопределить переменной окружения ``SUPPORT_SERVICE_URL``.
    В сервис передаются только поля, доступные во время регистрации обращения.
    """
    text = str(text or "").strip()
    if not text:
        raise SupportServiceError("Введите описание обращения.")

    url = os.getenv("SUPPORT_SERVICE_URL", DEFAULT_URL).strip() or DEFAULT_URL
    payload = {
        "text": text,
        "metadata": {
            str(key): "" if value is None else str(value).strip()
            for key, value in (metadata or {}).items()
        },
    }
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            details = json.loads(exc.read().decode("utf-8")).get("error")
        except Exception:
            details = None
        raise SupportServiceError(
            f"Сервис отклонил запрос: {details or exc.reason}."
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise SupportServiceError(
            "Сервис линии поддержки недоступен. Запустите "
            "ml/support_service_weights/app.py на порту 8765."
        ) from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SupportServiceError("Сервис вернул некорректный ответ.") from exc

    if not isinstance(result, dict):
        raise SupportServiceError("Сервис вернул ответ неизвестного формата.")
    if result.get("error"):
        raise SupportServiceError(str(result["error"]))
    return result
