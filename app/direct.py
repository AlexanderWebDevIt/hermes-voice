"""Прямой вызов модели — быстрый путь для голосового диалога.

Зачем это вообще нужно, если есть агент:

    Замеры на этом железе (Ryzen 5 3500U, провайдер kodikrouter → deepseek-v4-flash):
        Hermes Agent (с инструментами)   первый токен  5-17 с
        прямой вызов модели              первый токен  0.8-3.6 с

Агент тратит время на системный промпт, выбор инструментов и их вызовы. Для
голоса это слишком много: человек уже договорил и ждёт ответа. Прямой вызов
даёт ответ за доли секунды, но **не умеет пользоваться инструментами** — не
ищет в вебе, не читает файлы, не помнит прошлые сессии.

Поэтому режим переключаемый: в приложении есть кнопка «Агент / Быстрый».
Быстрый — для болтовни и коротких вопросов, агент — когда нужны инструменты.

Настройки берутся из конфига Hermes (config.yaml + .env), чтобы не дублировать
их руками: если поменяешь модель у агента, голос поедет на ней же.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Промпт подобран под то, что текст будет прочитан вслух.
VOICE_SYSTEM_PROMPT = (
    "Ты — голосовой собеседник. Твой ответ будет прочитан вслух синтезатором речи.\n"
    "Правила:\n"
    "— Отвечай кратко: одно-три предложения, если не просят подробнее.\n"
    "— Только обычный текст: никакого markdown, списков, таблиц, ссылок и эмодзи.\n"
    "— Числа, аббревиатуры и формулы пиши так, как их произносят вслух.\n"
    "— Не описывай свои действия и не упоминай, что ты языковая модель.\n"
    "— Отвечай на языке собеседника."
)


class DirectError(RuntimeError):
    """Прямой вызов не удался."""


def hermes_home() -> Path:
    """Каталог данных Hermes.

    Порядок: HV_HERMES_HOME → %LOCALAPPDATA%\\hermes (Windows) → ~/.hermes.
    """
    override = os.getenv("HV_HERMES_HOME")
    if override:
        return Path(override)
    local = os.getenv("LOCALAPPDATA")
    if local and (Path(local) / "hermes").is_dir():
        return Path(local) / "hermes"
    return Path.home() / ".hermes"


def discover_model() -> Optional[Dict[str, str]]:
    """Вытащить base_url/модель/ключ из конфига Hermes.

    Возвращает None, если конфиг недоступен — тогда берутся явные настройки
    hermes-voice (HV_DIRECT_* / config.json).
    """
    home = hermes_home()
    config_path = home / "config.yaml"
    if not config_path.is_file():
        logger.info("Прямой режим: %s не найден, беру настройки из config.json", config_path)
        return None

    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Прямой режим: не удалось прочитать config.yaml: %s", exc)
        return None

    model_cfg = config.get("model") or {}
    provider = str(model_cfg.get("provider") or "")
    base_url = str(model_cfg.get("base_url") or "")
    model = str(model_cfg.get("default") or "")

    if not base_url:
        providers = config.get("providers") or {}
        entry = providers.get(provider) or {}
        base_url = str(entry.get("base_url") or "")
        model = model or str(entry.get("model") or "")

    if not base_url or not model:
        return None

    return {
        "base_url": base_url.rstrip("/"),
        "model": model,
        "api_key": _key_from_env(home, provider),
        "provider": provider,
    }


def _key_from_env(home: Path, provider: str) -> str:
    """Ключ провайдера из .env Hermes (KODIKROUTER_API_KEY и т.п.)."""
    env_path = home / ".env"
    if not env_path.is_file() or not provider:
        return ""
    wanted = f"{provider.upper()}_API_KEY"
    try:
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip().upper() == wanted:
                return value.strip().strip('"').strip("'")
    except OSError as exc:
        logger.warning("Прямой режим: не удалось прочитать %s: %s", env_path, exc)
    return ""


class DirectLLM:
    """OpenAI-совместимый стриминг напрямую в модель, без агентной обвязки."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str = VOICE_SYSTEM_PROMPT,
        max_tokens: int = 400,
        temperature: float = 0.6,
        timeout: float = 120.0,
        history_limit: int = 16,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history_limit = history_limit
        # Постоянный клиент: переиспользование соединения экономит заметную
        # часть первого токена (проверено — первый запрос холодный, дальше быстрее).
        self._client = httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_keepalive_connections=4))

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        if not self.enabled:
            return False
        try:
            resp = await self._client.get(f"{self.base_url}/models", headers=self._headers())
            return resp.status_code < 500
        except Exception:
            return False

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _messages(self, text: str, history: Optional[List[dict]]) -> List[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history[-self.history_limit:])
        messages.append({"role": "user", "content": text})
        return messages

    async def stream_reply(self, text: str, history: Optional[List[dict]] = None) -> AsyncIterator[str]:
        """Отдавать дельты ответа по мере генерации."""
        if not self.enabled:
            raise DirectError("Прямой режим не настроен: нет base_url или модели")

        payload = {
            "model": self.model,
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": self._messages(text, history),
        }

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise DirectError(f"Модель вернула {resp.status_code}: {body[:300]}")

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece
        except DirectError:
            raise
        except Exception as exc:
            raise DirectError(f"Сбой прямого вызова: {exc}") from exc

    async def reply(self, text: str, history: Optional[List[dict]] = None) -> str:
        parts: List[str] = []
        async for piece in self.stream_reply(text, history):
            parts.append(piece)
        return "".join(parts).strip()

    async def warmup(self) -> None:
        """Один дешёвый запрос, чтобы прогреть соединение и маршрут у провайдера.

        Первый запрос к провайдеру стабильно медленнее последующих (замеры:
        6.9 с холодный против 0.8 с прогретый). За эту разницу не должен
        платить пользователь.
        """
        if not self.enabled:
            return
        try:
            payload = {
                "model": self.model,
                "stream": False,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "ok"}],
            }
            await self._client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
            )
            logger.info("Прямой режим: соединение прогрето (%s)", self.model)
        except Exception as exc:
            logger.warning("Прямой режим: прогрев не удался: %s", exc)
