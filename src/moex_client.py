"""Общий устойчивый HTTP-клиент для запросов к MOEX ISS, smart-lab и подобным API.

Использует `requests.Session` с `urllib3.Retry`, ротацию User-Agent,
тайм-ауты и экспоненциальную ретрансляцию при 429/5xx.
"""

import logging
import random
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
]

_DEFAULT_TIMEOUT: int = 15


def create_http_session() -> requests.Session:
    """Создать HTTP-сессию с повторными запросами ( Retry-адптер ).

    Возвращает:
        requests.Session: настроенная сессия.

    Правила:
        * 5 повторных попыток соединения;
        * ретраи только по статусам 429, 500, 502, 503, 504;
        * уважение к заголовку Retry-After;
        * фиксированный IPv4-пул (pool_connections = pool_maxsize = 10).
    """
    retry = Retry(
        total=5,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=10,
    )

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "Accept": "application/json",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8",
        }
    )
    return session


def get_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Получить JSON через устойчивую HTTP-сессию.

    Args:
        session: HTTP-сессия.
        url: Адрес запрашиваемого ресурса.
        params: Дополнительные параметры GET.
        timeout: Тайм-аут в секундах (по умолчанию 15).

    Returns:
        dict[str, Any]: полученный JSON-объект.

    Raises:
        requests.RequestException: при ошибке TCP/HTTP.
        ValueError: при невалидном или не-словаре JSON-ответе.
    """
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
    }

    try:
        response = session.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()

        result = response.json()

        if not isinstance(result, dict):
            raise ValueError("Некорректная структура JSON: ответ не является объектом")

        return result

    except (requests.RequestException, ValueError):
        logger.exception("Ошибка получения JSON: %s", url)
        raise