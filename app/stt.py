"""Распознавание речи: faster-whisper на CPU.

Модель грузится лениво (первый запрос платит за загрузку) и выгружается после
простоя, чтобы не держать память занятой, когда голосом не пользуются.
Инференс — блокирующий, поэтому уходит в отдельный поток и не морозит event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")


class SpeechToText:
    """Обёртка над faster-whisper с ленивой загрузкой и выгрузкой по простою."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "float32",
        language: str = "ru",
        beam_size: int = 1,
        cpu_threads: int = 8,
        initial_prompt: str = "",
        idle_unload_seconds: float = 600.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self.initial_prompt = initial_prompt
        self.idle_unload_seconds = idle_unload_seconds

        self._model = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._reaper: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ модель

    def _load_sync(self):
        """Загрузить модель (внутри уже захваченного лока)."""
        from faster_whisper import WhisperModel

        started = time.time()
        logger.info(
            "Загружаю faster-whisper '%s' (%s/%s, потоков %d)…",
            self.model_name,
            self.device,
            self.compute_type,
            self.cpu_threads,
        )
        model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )
        logger.info("Модель готова за %.1f с", time.time() - started)
        return model

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                self._model = self._load_sync()
        return self._model

    def unload(self) -> None:
        """Освободить память, занятую моделью."""
        with self._lock:
            if self._model is not None:
                logger.info("Выгружаю модель faster-whisper (простой)")
                self._model = None

    async def warmup(self) -> None:
        """Загрузить модель заранее, чтобы первая фраза не ждала загрузки."""
        loop = asyncio.get_running_loop()
        started = time.time()
        await loop.run_in_executor(_executor, self._ensure_model)
        self._last_used = time.time()
        logger.info("Прогрев STT завершён за %.1f с", time.time() - started)

    def _start_reaper(self) -> None:
        """Фоновый сторож: выгружает модель, если ей давно не пользовались."""
        if self.idle_unload_seconds <= 0 or self._reaper is not None:
            return

        async def loop() -> None:
            while True:
                await asyncio.sleep(30)
                idle = time.time() - self._last_used
                if self._model is not None and self._last_used and idle > self.idle_unload_seconds:
                    self.unload()

        self._reaper = asyncio.create_task(loop())

    # ------------------------------------------------------------------ инференс

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        model = self._ensure_model()
        segments, _info = model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,      # сегментацию уже сделал наш VAD
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt or None,
        )
        return "".join(segment.text for segment in segments).strip()

    async def transcribe(self, audio: np.ndarray) -> str:
        """Распознать PCM float32 моно 16 кГц."""
        if audio.size == 0:
            return ""

        self._start_reaper()
        loop = asyncio.get_running_loop()
        started = time.time()
        text = await loop.run_in_executor(_executor, self._transcribe_sync, audio)
        self._last_used = time.time()
        duration = audio.size / 16000.0
        logger.info(
            "STT: %.1f с аудио → %.1f с обработки → %r",
            duration,
            time.time() - started,
            text[:120],
        )
        return text

    # ------------------------------------------------------------------ файлы

    def _transcribe_file_sync(self, path: str, language: Optional[str]) -> str:
        """Распознать файл на диске: декодирование берёт на себя PyAV внутри faster-whisper."""
        model = self._ensure_model()
        segments, _info = model.transcribe(
            path,
            language=language or self.language,
            beam_size=self.beam_size,
            vad_filter=True,       # в файле тишину можно смело выбросить
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt or None,
        )
        return "".join(segment.text for segment in segments).strip()

    async def transcribe_file(self, path: str, language: Optional[str] = None) -> str:
        """Распознать аудиофайл по пути (wav, mp3, ogg, webm, m4a — что понимает PyAV)."""
        self._start_reaper()
        loop = asyncio.get_running_loop()
        started = time.time()
        text = await loop.run_in_executor(_executor, self._transcribe_file_sync, path, language)
        self._last_used = time.time()
        logger.info("STT (файл %s): %.1f с → %r", os.path.basename(path), time.time() - started, text[:120])
        return text

    async def transcribe_bytes(
        self,
        raw: bytes,
        filename: str = "audio.bin",
        language: Optional[str] = None,
    ) -> str:
        """Распознать аудио, пришедшее как байты (загрузка из приложения).

        Формат определяем по расширению имени файла — это то, что прислал клиент.
        """
        suffix = Path(filename).suffix or ".bin"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="hv_upload_")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
            return await self.transcribe_file(tmp_path, language=language)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------ утилиты

    @staticmethod
    def pcm16_to_float32(pcm: bytes) -> np.ndarray:
        """Сырой PCM 16-bit LE моно → float32 в диапазоне [-1, 1]."""
        if not pcm:
            return np.zeros(0, dtype=np.float32)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        return samples / 32768.0
