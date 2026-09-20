"""Синтез речи. Два движка под одним интерфейсом.

* ``piper`` — **локальный** (ONNX, CPU). Замеры на Ryzen 5 3500U:
  первый звук 0.24 с на короткой фразе и 0.91 с на 145 символах,
  генерация ~9x быстрее реального времени. Без сети, без ключей, без лимитов.
  Отдаёт WAV (PCM s16 без сжатия).

* ``edge`` — облачный edge-tts (Microsoft). Голосов больше, интонация богаче,
  но первый звук от 2 с и **растёт с длиной текста** (2.1 с на 15 символах,
  6.7 с на 160) — это буферизация на стороне сервиса, канал тут ни при чём
  (ping 52 мс). Годится для длинных ответов, не для живого диалога.
  Отдаёт MP3.

Наружу движок отдаёт **готовые к проигрыванию файлы**: клиент просто пишет
полученные байты в файл и играет. Формат объявляется в hello, чтобы клиент
выбрал правильное расширение.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import struct
import threading
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)

# Границы, по которым безопасно резать: конец предложения или длинная пауза.
_SENTENCE_END = re.compile(r"[.!?…]+[\s\"'»)]*|\n{2,}|[;:]\s+")
_ABBREV = re.compile(r"\b(т\.\s*е|т\.\s*д|т\.\s*п|напр|см|рис|стр)\.$", re.IGNORECASE)


# --------------------------------------------------------------------- нарезка


def _cut_point(buffer: str, min_chars: int) -> Optional[int]:
    """Найти место разреза в буфере: конец предложения не раньше min_chars."""
    for match in _SENTENCE_END.finditer(buffer):
        end = match.end()
        if end < min_chars:
            continue
        if _ABBREV.search(buffer[:end].rstrip()):
            continue
        return end
    return None


async def text_chunks(
    deltas: AsyncIterator[str],
    first_chars: int = 60,
    chunk_chars: int = 220,
) -> AsyncIterator[str]:
    """Превратить поток дельт от агента в поток готовых к озвучке кусков.

    Первый кусок отдаётся как только набралось ``first_chars`` и найден конец
    предложения — это и есть выигрыш в задержке. Последний остаток отдаётся
    принудительно, когда поток дельт закончился.
    """
    buffer = ""
    first = True

    async for delta in deltas:
        buffer += delta
        limit = first_chars if first else chunk_chars

        # Жёсткий предохранитель: если предложение бесконечно длинное, режем по длине.
        while True:
            cut = _cut_point(buffer, limit)
            if cut is None:
                if len(buffer) >= max(limit * 3, 400):
                    cut = buffer.rfind(" ", 0, max(limit * 3, 400))
                    if cut <= 0:
                        break
                else:
                    break
            piece = buffer[:cut].strip()
            buffer = buffer[cut:]
            if piece:
                first = False
                yield piece

    tail = buffer.strip()
    if tail:
        yield tail


# ----------------------------------------------------------------------- WAV


def wav_bytes(pcm: bytes, rate: int, channels: int = 1, bits: int = 16) -> bytes:
    """Обернуть сырой PCM в WAV-контейнер.

    WAV выбран вместо MP3 сознательно: не нужен ни энкодер, ни ffmpeg, ни
    потеря качества, ни задержка на сжатие. Плата — ~44 КБ/с на канал.
    """
    byte_rate = rate * channels * bits // 8
    block_align = channels * bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack(
        "<IHHIIHH", 16, 1, channels, rate, byte_rate, block_align, bits
    )
    header += b"data" + struct.pack("<I", len(pcm))
    return header + pcm


# --------------------------------------------------------------------- piper


class PiperEngine:
    """Локальный синтез. Голоса — .onnx файлы в каталоге voices/."""

    format = "wav"

    def __init__(self, voices_dir: Path, default_voice: str = "", rate: float = 1.0) -> None:
        self.voices_dir = Path(voices_dir)
        self.default_voice = default_voice
        self.rate = rate
        self._cache: Dict[str, object] = {}
        self._lock = threading.Lock()
        self._executor_lock = asyncio.Lock()
        self.sample_rate = 22050
        self.channels = 1

    # ------------------------------------------------------------- голоса

    def available(self) -> List[str]:
        if not self.voices_dir.is_dir():
            return []
        return sorted(p.stem for p in self.voices_dir.glob("*.onnx"))

    def _resolve(self, voice: Optional[str]) -> str:
        name = (voice or self.default_voice or "").strip()
        known = self.available()
        if name in known:
            return name
        if known:
            if name:
                logger.warning("Piper: голос %r не найден, беру %r", name, known[0])
            return known[0]
        raise RuntimeError(
            f"Piper: в {self.voices_dir} нет ни одного .onnx. "
            "Скачай голоса (см. README, раздел «Голоса»)."
        )

    def _voice(self, name: str):
        """Загрузить модель один раз и держать в памяти (~60 МБ на голос)."""
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(name)
            if cached is not None:
                return cached
            from piper import PiperVoice

            path = self.voices_dir / f"{name}.onnx"
            logger.info("Piper: загружаю голос %s…", name)
            voice = PiperVoice.load(path)
            self.sample_rate = getattr(voice.config, "sample_rate", self.sample_rate)
            self._cache[name] = voice
            logger.info("Piper: голос %s готов (sr=%d)", name, self.sample_rate)
            return voice

    # ------------------------------------------------------------- синтез

    def _synth_sync(self, text: str, voice_name: str) -> bytes:
        from piper import SynthesisConfig

        voice = self._voice(voice_name)
        config = SynthesisConfig(length_scale=1.0 / max(self.rate, 0.1))
        pcm = bytearray()
        for chunk in voice.synthesize(text, syn_config=config):
            pcm += chunk.audio_int16_bytes
        return wav_bytes(bytes(pcm), self.sample_rate, self.channels)

    async def stream_audio(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """Отдать один цельный WAV на кусок текста.

        Синтез — блокирующий CPU, поэтому уходит в поток. Замок гарантирует,
        что два голоса не грузятся и не считаются одновременно: на 4 ядрах
        параллельный синтез только замедлил бы оба.
        """
        if not text.strip():
            return
        name = self._resolve(voice)
        async with self._executor_lock:
            data = await asyncio.to_thread(self._synth_sync, text, name)
        if data:
            yield data

    async def synthesize(self, text: str, voice: Optional[str] = None) -> bytes:
        """Один файл на весь текст."""
        if not text.strip():
            return b""
        name = self._resolve(voice)
        async with self._executor_lock:
            return await asyncio.to_thread(self._synth_sync, text, name)

    async def warmup(self, voice: Optional[str] = None) -> None:
        try:
            name = self._resolve(voice)
            await asyncio.to_thread(self._voice, name)
        except Exception as exc:  # прогрев не должен ронять старт сервиса
            logger.warning("Piper: прогрев не удался: %s", exc)

    async def warmup_all(self) -> None:
        """Загрузить все голоса заранее.

        Каждый голос — отдельная ONNX-модель (~60 МБ, ~4 с на загрузку).
        Без этого первое переключение голоса в разговоре стоит четыре секунды
        тишины. Грузим последовательно в фоне, чтобы не тормозить старт.
        """
        for name in self.available():
            if name in self._cache:
                continue
            try:
                await asyncio.to_thread(self._voice, name)
            except Exception as exc:
                logger.warning("Piper: не удалось прогреть %s: %s", name, exc)

    def describe(self) -> List[dict]:
        return [{"name": n, "locale": "ru-RU", "gender": "", "engine": "piper"} for n in self.available()]


# ---------------------------------------------------------------------- edge


class EdgeEngine:
    """Облачный edge-tts. Отдаёт mp3."""

    format = "mp3"

    # Имена голосов edge выглядят как ru-RU-SvetlanaNeural.
    _NAME = re.compile(r"^[a-z]{2}-[A-Z]{2}-\w+Neural$")

    def __init__(self, default_voice: str = "ru-RU-SvetlanaNeural", rate: str = "+10%", volume: str = "+0%") -> None:
        self.default_voice = default_voice
        self.rate = rate
        self.volume = volume

    def _resolve(self, voice: Optional[str]) -> str:
        """Не пускать в облако имя голоса piper (ru_RU-irina-medium)."""
        name = (voice or "").strip()
        if name and self._NAME.match(name):
            return name
        if name:
            logger.warning("TTS(edge): голос %r не похож на edge-голос, беру %r", name, self.default_voice)
        return self.default_voice

    async def stream_audio(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        import edge_tts

        communicate = edge_tts.Communicate(
            text,
            self._resolve(voice),
            rate=self.rate,
            volume=self.volume,
        )
        try:
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    yield chunk["data"]
        except Exception as exc:  # edge-tts ходит в сеть — падать нельзя
            logger.error("TTS(edge): сбой синтеза: %s", exc)

    async def warmup(self, voice: Optional[str] = None) -> None:
        return None

    async def synthesize(self, text: str, voice: Optional[str] = None) -> bytes:
        parts: List[bytes] = []
        async for piece in self.stream_audio(text, voice=voice):
            parts.append(piece)
        return b"".join(parts)

    @staticmethod
    async def describe(only_prefix: Optional[str] = "ru-") -> List[dict]:
        import edge_tts

        try:
            voices = await edge_tts.list_voices()
        except Exception as exc:
            logger.error("TTS(edge): не удалось получить список голосов: %s", exc)
            return []
        if only_prefix:
            voices = [v for v in voices if str(v.get("Locale", "")).startswith(only_prefix)]
        voices.sort(key=lambda v: (v.get("Locale", ""), v.get("ShortName", "")))
        return [
            {"name": v.get("ShortName"), "locale": v.get("Locale"), "gender": v.get("Gender"), "engine": "edge"}
            for v in voices
        ]


# ------------------------------------------------------------------- фасад


class TextToSpeech:
    """Единая точка входа: выбрать движок и синтезировать."""

    def __init__(
        self,
        engine: str = "piper",
        voice: str = "",
        rate: str = "+10%",
        volume: str = "+0%",
        piper_voices_dir: Path | str = "voices",
        piper_rate: float = 1.0,
        first_chunk_chars: int = 60,
        chunk_chars: int = 220,
    ) -> None:
        self.engine_name = (engine or "piper").strip().lower()
        self.first_chunk_chars = first_chunk_chars
        self.chunk_chars = chunk_chars

        self.piper = PiperEngine(piper_voices_dir, default_voice=voice, rate=piper_rate)
        # Дефолт edge не должен унаследовать имя голоса piper (ru_RU-irina-medium):
        # иначе при смене движка улетит заведомо неверное имя.
        edge_default = voice if EdgeEngine._NAME.match(voice or "") else "ru-RU-SvetlanaNeural"
        self.edge = EdgeEngine(default_voice=edge_default, rate=rate, volume=volume)

        if self.engine_name not in {"piper", "edge"}:
            logger.warning("TTS: неизвестный движок %r, беру piper", self.engine_name)
            self.engine_name = "piper"

        self._engine = self.piper if self.engine_name == "piper" else self.edge
        self._fillers: List[bytes] = []

    # ------------------------------------------------------------ отбивки

    async def prepare_fillers(self, phrases: List[str]) -> None:
        """Заранее озвучить короткие поддакивания.

        Пока модель думает, в эфире тишина — и разговор перестаёт быть разговором.
        Живой человек на такое место вставляет «так…», «сейчас». Синтезировать
        их в момент ожидания бессмысленно: это та же задержка. Поэтому готовим
        заранее и потом отдаём мгновенно.
        """
        self._fillers = []
        for phrase in phrases:
            text = phrase.strip()
            if not text:
                continue
            try:
                audio = await self._engine.synthesize(text)
            except Exception as exc:
                logger.warning("TTS: не удалось подготовить отбивку %r: %s", text, exc)
                continue
            if audio:
                self._fillers.append(audio)
        if self._fillers:
            logger.info("TTS: готово отбивок — %d", len(self._fillers))

    @property
    def has_fillers(self) -> bool:
        return bool(self._fillers)

    def pick_filler(self) -> Optional[bytes]:
        """Случайная отбивка. None, если их нет."""
        if not self._fillers:
            return None
        return random.choice(self._fillers)

    # ------------------------------------------------------------ свойства

    @property
    def format(self) -> str:
        return self._engine.format

    @property
    def sample_rate(self) -> int:
        return getattr(self._engine, "sample_rate", 24000)

    @property
    def channels(self) -> int:
        return getattr(self._engine, "channels", 1)

    def describe_output(self) -> dict:
        return {"format": self.format, "sample_rate": self.sample_rate, "channels": self.channels}

    # -------------------------------------------------------------- синтез

    async def stream_audio(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """Все байты, отданные до следующего куска, — один цельный файл."""
        async for data in self._engine.stream_audio(text, voice=voice):
            yield data

    async def synthesize(self, text: str, voice: Optional[str] = None) -> bytes:
        """Собрать весь ответ в один файл (для REST-эндпоинта)."""
        return await self._engine.synthesize(text, voice=voice)

    async def warmup(self, voice: Optional[str] = None) -> None:
        await self._engine.warmup(voice)

    async def warmup_all(self) -> None:
        """Прогреть все голоса движка (для piper — предзагрузка моделей)."""
        if hasattr(self._engine, "warmup_all"):
            await self._engine.warmup_all()

    def voice_names(self) -> List[str]:
        """Список голосов без обращения к сети (для hello)."""
        if self.engine_name == "piper":
            return self.piper.available()
        return []

    async def list_voices(self) -> List[dict]:
        if self.engine_name == "piper":
            return self.piper.describe()
        return await self.edge.describe()
