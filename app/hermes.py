"""Клиент к Hermes Gateway.

Голосовой слой не подменяет агента: он только превращает речь в текст и обратно,
а сам ответ формирует тот же Hermes, что обслуживает и текстовый чат.
Используем OpenAI-совместимый ``POST /v1/chat/completions`` со стримингом,
чтобы начинать озвучку до того, как агент закончит печатать.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class HermesError(RuntimeError):
    """Ошибка при обращении к Hermes Gateway."""


class HermesClient:
    """Асинхронный клиент Hermes с потоковой отдачей текста."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    # ------------------------------------------------------------------ служебное

    def _headers(self, session_id: Optional[str] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # Hermes держит память диалога по этому заголовку (см. /v1/capabilities).
        if session_id:
            headers["X-Hermes-Session-Id"] = session_id
        return headers

    async def health(self) -> bool:
        """Жив ли gateway."""
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.status_code < 500
        except httpx.HTTPError as exc:
            logger.warning("Hermes недоступен: %s", exc)
            return False

    # ------------------------------------------------------------------ основной вызов

    async def stream_reply(
        self,
        text: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncIterator[str]:
        """Отдавать куски ответа агента по мере генерации.

        ``history`` — предыдущие реплики, если хочется держать контекст без сессий
        на стороне Hermes (по умолчанию контекст ведёт сам gateway).
        """
        messages: List[Dict[str, str]] = list(history or [])
        messages.append({"role": "user", "content": text})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    headers=self._headers(session_id),
                    json=payload,
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")[:400]
                        raise HermesError(f"Hermes вернул {resp.status_code}: {body}")

                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        for choice in event.get("choices") or []:
                            delta = choice.get("delta") or {}
                            piece = delta.get("content")
                            if piece:
                                yield piece
        except httpx.HTTPError as exc:
            raise HermesError(f"Сбой соединения с Hermes: {exc}") from exc

    async def reply(
        self,
        text: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Собрать полный ответ (для REST-эндпоинта и тестов)."""
        parts: List[str] = []
        async for piece in self.stream_reply(text, session_id=session_id, history=history):
            parts.append(piece)
        return "".join(parts).strip()
