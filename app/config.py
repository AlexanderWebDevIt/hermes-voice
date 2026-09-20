"""Настройки hermes-voice.

Приоритет: переменные окружения → config.json рядом с проектом → значения по умолчанию.
Все ключи бесплатные и локальные: faster-whisper (STT) и Piper (TTS).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"


def _load_file_config() -> Dict[str, Any]:
    """config.json рядом с проектом (необязателен)."""
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


_FILE = _load_file_config()


def _get(key: str, env: str, default: Any) -> Any:
    """env-переменная важнее файла, файл важнее дефолта."""
    value = os.getenv(env)
    if value is not None and value != "":
        return value
    if key in _FILE:
        return _FILE[key]
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return default


def _as_list(value: Any, default: list) -> list:
    """Из env приходит строкой: «Так…,Секунду…». Из config.json — уже списком."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(default)


# Отбивки вычисляем на уровне модуля: в датаклассе изменяемый дефолт запрещён,
# а default_factory не умеет читать аргументы — только замкнутые значения.
_FILLER_PHRASES = _as_list(
    _get("filler_phrases", "HV_FILLER_PHRASES", ["Так…", "Секунду…", "Сейчас.", "Ага…"]),
    ["Так…", "Секунду…", "Сейчас.", "Ага…"],
)


@dataclass(frozen=True)
class Settings:
    """Все параметры сервиса."""

    # --- HTTP-сервер ---
    host: str = _get("host", "HV_HOST", "0.0.0.0")
    port: int = _as_int(_get("port", "HV_PORT", 8643), 8643)

    # --- Hermes Agent (агент остаётся тем же, что в gateway) ---
    hermes_base_url: str = _get("hermes_base_url", "HV_HERMES_URL", "http://127.0.0.1:8642")
    hermes_api_key: str = _get("hermes_api_key", "HV_HERMES_KEY", "")
    hermes_model: str = _get("hermes_model", "HV_HERMES_MODEL", "hermes-agent")
    hermes_timeout: float = _as_float(_get("hermes_timeout", "HV_HERMES_TIMEOUT", 300), 300.0)

    # --- Бэкенд ответов ---
    # agent  — полный Hermes Agent с инструментами. Первый токен 5-17 с (замеры).
    # direct — прямой вызов модели без агентной обвязки. Первый токен 0.8-3.6 с,
    #          но без инструментов: не ищет в вебе, не читает файлы, не помнит сессии.
    # В приложении переключается на ходу командой config.
    voice_backend: str = _get("voice_backend", "HV_VOICE_BACKEND", "direct")
    # Явные настройки прямого режима. Если пусты — берутся из конфига Hermes
    # (config.yaml + .env), чтобы не дублировать модель руками.
    direct_base_url: str = _get("direct_base_url", "HV_DIRECT_URL", "")
    direct_api_key: str = _get("direct_api_key", "HV_DIRECT_KEY", "")
    direct_model: str = _get("direct_model", "HV_DIRECT_MODEL", "")
    direct_max_tokens: int = _as_int(_get("direct_max_tokens", "HV_DIRECT_MAX_TOKENS", 400), 400)
    direct_temperature: float = _as_float(_get("direct_temperature", "HV_DIRECT_TEMP", 0.6), 0.6)
    direct_warmup: bool = _as_bool(_get("direct_warmup", "HV_DIRECT_WARMUP", True), True)

    # --- STT: faster-whisper ---
    # Замеры на этой машине (Ryzen 5 3500U): base+float32 = 0.26x от длительности аудио,
    # small+float32 = 0.99x. Для живого диалога берём base; small — если важнее пунктуация.
    stt_model: str = _get("stt_model", "HV_STT_MODEL", "base")
    stt_device: str = _get("stt_device", "HV_STT_DEVICE", "cpu")
    # int8 на Zen+ медленнее float32 (нет AVX-512) — проверено бенчмарком.
    stt_compute: str = _get("stt_compute", "HV_STT_COMPUTE", "float32")
    # Больше потоков — быстрее; 8 соответствует числу логических ядер.
    stt_threads: int = _as_int(_get("stt_threads", "HV_STT_THREADS", 8), 8)
    stt_language: str = _get("stt_language", "HV_STT_LANGUAGE", "ru")
    stt_beam_size: int = _as_int(_get("stt_beam_size", "HV_STT_BEAM", 1), 1)
    # Подсказка модели: помогает с пунктуацией и именами. Пусто — не подсказывать.
    stt_initial_prompt: str = _get("stt_initial_prompt", "HV_STT_PROMPT", "")
    # Сколько секунд держать модель в памяти без запросов (0 — не выгружать).
    stt_idle_unload: float = _as_float(_get("stt_idle_unload", "HV_STT_IDLE_UNLOAD", 600), 600.0)
    # Прогревать модель при старте: иначе первая фраза ждёт загрузки ~4 секунды.
    stt_warmup: bool = _as_bool(_get("stt_warmup", "HV_STT_WARMUP", True), True)

    # --- TTS ---
    # piper — локальный ONNX-синтез: первый звук 0.24-0.91 с, ~9x realtime, без сети.
    # edge  — облачный edge-tts: голосов больше, но первый звук от 2 с и растёт с длиной
    #         текста (замеры: 15 симв. — 2.1 с, 160 симв. — 6.7 с). Только для длинных ответов.
    tts_engine: str = _get("tts_engine", "HV_TTS_ENGINE", "piper")
    tts_voice: str = _get("tts_voice", "HV_TTS_VOICE", "ru_RU-irina-medium")
    # Каталог с голосами piper (.onnx + .onnx.json).
    tts_voices_dir: str = _get("tts_voices_dir", "HV_TTS_VOICES_DIR", "voices")
    # Темп piper: 1.0 — как записан, 1.1 — на 10% быстрее.
    tts_piper_rate: float = _as_float(_get("tts_piper_rate", "HV_TTS_PIPER_RATE", 1.1), 1.1)
    # Параметры edge-tts (действуют только при tts_engine = edge).
    tts_rate: str = _get("tts_rate", "HV_TTS_RATE", "+10%")
    tts_volume: str = _get("tts_volume", "HV_TTS_VOLUME", "+0%")
    # Длина первого куска для озвучки: короткий старт = меньше задержка до первого звука.
    tts_first_chunk_chars: int = _as_int(_get("tts_first_chunk_chars", "HV_TTS_FIRST_CHUNK", 60), 60)
    tts_chunk_chars: int = _as_int(_get("tts_chunk_chars", "HV_TTS_CHUNK", 220), 220)
    # Прогревать голос piper при старте: иначе первый ответ платит ~4 с за загрузку модели.
    tts_warmup: bool = _as_bool(_get("tts_warmup", "HV_TTS_WARMUP", True), True)
    # Предзагрузить все голоса piper (~60 МБ и ~4 с на каждый) в фоне после старта.
    # Без этого первое переключение голоса стоит четыре секунды тишины.
    tts_preload_all: bool = _as_bool(_get("tts_preload_all", "HV_TTS_PRELOAD_ALL", True), True)

    # --- Отбивки: что сказать, пока модель думает ---
    # Тишина в 4 секунды превращает разговор в допрос. Короткое «так…» её закрывает.
    filler_enabled: bool = _as_bool(_get("filler_enabled", "HV_FILLER", True), True)
    # Сколько ждать первый кусок ответа, прежде чем вставить отбивку.
    # Меньше — успеет влезть даже при быстром ответе; больше — влезет реже.
    filler_delay_ms: int = _as_int(_get("filler_delay_ms", "HV_FILLER_DELAY_MS", 700), 700)
    filler_phrases: list = field(
        default_factory=lambda: _FILLER_PHRASES
    )

    # --- VAD: когда считать, что человек заговорил и когда закончил ---
    vad_engine: str = _get("vad_engine", "HV_VAD_ENGINE", "energy")
    # Во сколько раз RMS речи должен превышать уровень шума, чтобы это была речь.
    vad_start_ratio: float = _as_float(_get("vad_start_ratio", "HV_VAD_START_RATIO", 2.6), 2.6)
    # Тишина после речи, завершающая фразу (мс).
    # 650 мс оказалось мало: пауза после «Привет, Гермес!» рвала фразу на два куска.
    vad_silence_ms: int = _as_int(_get("vad_silence_ms", "HV_VAD_SILENCE_MS", 900), 900)
    # Минимальная длина фразы (мс) — короче считается шумом.
    vad_min_speech_ms: int = _as_int(_get("vad_min_speech_ms", "HV_VAD_MIN_SPEECH_MS", 300), 300)
    # Запас перед началом речи (мс), чтобы не срезать первое слово.
    vad_pre_roll_ms: int = _as_int(_get("vad_pre_roll_ms", "HV_VAD_PRE_ROLL_MS", 350), 350)
    # Сколько ждать начала речи после открытия соединения (мс); 0 — бесконечно.
    vad_initial_wait_ms: int = _as_int(_get("vad_initial_wait_ms", "HV_VAD_INITIAL_WAIT_MS", 0), 0)
    # Потолок длины одной фразы (мс) — защита от «залипшего» микрофона.
    vad_max_speech_ms: int = _as_int(_get("vad_max_speech_ms", "HV_VAD_MAX_SPEECH_MS", 30000), 30000)

    # --- Аудио на входе ---
    input_sample_rate: int = _as_int(_get("input_sample_rate", "HV_INPUT_SR", 16000), 16000)
    input_channels: int = _as_int(_get("input_channels", "HV_INPUT_CH", 1), 1)

    # --- Отладка ---
    debug: bool = _as_bool(_get("debug", "HV_DEBUG", False), False)


settings = Settings()
