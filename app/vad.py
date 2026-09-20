"""Детектор речи (VAD) — определяет, когда человек начал и закончил фразу.

Без этого «голосовой режим» превращается в рацию с кнопкой: пришлось бы вручную
показывать, что фраза закончена. Здесь решение принимается по энергии сигнала
относительно адаптивного уровня шума — порог подстраивается под конкретный
микрофон и комнату, а не берётся с потолка.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VadResult:
    """Что произошло на очередном куске аудио."""

    speech_started: bool = False
    segment: Optional[np.ndarray] = None   # float32, моно 16 кГц — готовая фраза
    level: float = 0.0                     # текущая громкость (для индикатора в UI)


class StreamResampler:
    """Приведение потока PCM s16le к 16 кГц «на лету».

    Телефон не обязан отдавать 16 кГц: Android часто навязывает 48 кГц, и
    ``expo-audio`` честно сообщает фактическую частоту в каждом буфере. А whisper
    и VAD работают на 16 кГц — если скормить им 48 кГц как 16, получится и
    ускорение втрое, и мусор в распознавании.

    Передискретизация линейная, с сохранением дробной позиции между вызовами:
    буферы приходят произвольного размера, и без этого на каждой стыке была бы
    слышимая ступенька.
    """

    def __init__(self, src_rate: int, dst_rate: int = 16000) -> None:
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self.step = src_rate / float(dst_rate)
        self._buf = np.zeros(0, dtype=np.float32)
        self._cursor = 0.0          # позиция следующего выходного сэмпла в _buf

    @property
    def passthrough(self) -> bool:
        return abs(self.step - 1.0) < 1e-9

    def feed(self, pcm: bytes) -> bytes:
        """Отдать порцию PCM на целевой частоте (может быть пустой)."""
        if self.passthrough:
            return pcm

        incoming = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        if incoming.size:
            self._buf = np.concatenate([self._buf, incoming])

        out: List[float] = []
        while self._cursor + 1.0 < len(self._buf):
            i = int(self._cursor)
            frac = self._cursor - i
            out.append(self._buf[i] * (1.0 - frac) + self._buf[i + 1] * frac)
            self._cursor += self.step

        drop = int(self._cursor)
        if drop:
            self._buf = self._buf[drop:]
            self._cursor -= drop

        if not out:
            return b""
        return np.asarray(out, dtype="<i2").tobytes()

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._cursor = 0.0


class SpeechSegmenter:
    """Режет поток аудио на фразы по паузам."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        start_ratio: float = 2.6,
        silence_ms: int = 650,
        min_speech_ms: int = 300,
        max_speech_ms: int = 30000,
        pre_roll_ms: int = 240,
        noise_init: float = 0.006,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.start_ratio = start_ratio
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms

        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2
        self.silence_frames_needed = max(1, silence_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.max_speech_frames = max(1, max_speech_ms // frame_ms)
        self.pre_roll_frames = max(0, pre_roll_ms // frame_ms)

        self._pending = b""
        self._pre_roll: List[np.ndarray] = []
        self._speech: List[np.ndarray] = []
        self._noise = noise_init
        self._in_speech = False
        self._silence_run = 0
        self._frames_in_speech = 0

    # ------------------------------------------------------------------ служебное

    def reset(self) -> None:
        """Сбросить состояние (новое соединение или прерывание)."""
        self._pending = b""
        self._pre_roll.clear()
        self._speech.clear()
        self._in_speech = False
        self._silence_run = 0
        self._frames_in_speech = 0

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(frame))))

    # ------------------------------------------------------------------ основной цикл

    def feed(self, pcm16: bytes) -> List[VadResult]:
        """Скормить сырой PCM 16-bit LE моно. Вернуть произошедшие события."""
        results: List[VadResult] = []
        self._pending += pcm16

        while len(self._pending) >= self.frame_bytes:
            raw = self._pending[: self.frame_bytes]
            self._pending = self._pending[self.frame_bytes :]

            frame = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            level = self._rms(frame)
            is_speech = level > max(self._noise * self.start_ratio, 0.004)

            if not self._in_speech:
                # Копим пре-ролл, чтобы не отрезать начало слова.
                self._pre_roll.append(frame)
                if len(self._pre_roll) > self.pre_roll_frames:
                    self._pre_roll.pop(0)

                if is_speech:
                    self._in_speech = True
                    self._speech = list(self._pre_roll)
                    self._pre_roll.clear()
                    self._silence_run = 0
                    self._frames_in_speech = 0
                    results.append(VadResult(speech_started=True, level=level))
                else:
                    # Обновляем оценку шума только вне речи.
                    self._noise = 0.95 * self._noise + 0.05 * level
                continue

            # --- мы внутри фразы ---
            self._speech.append(frame)
            self._frames_in_speech += 1

            if is_speech:
                self._silence_run = 0
            else:
                self._silence_run += 1

            finished = (
                self._silence_run >= self.silence_frames_needed
                or self._frames_in_speech >= self.max_speech_frames
            )
            if not finished:
                results.append(VadResult(level=level))
                continue

            segment = np.concatenate(self._speech) if self._speech else np.zeros(0, dtype=np.float32)
            speech_frames = self._frames_in_speech - self._silence_run

            self._in_speech = False
            self._speech = []
            self._silence_run = 0
            self._frames_in_speech = 0

            if speech_frames >= self.min_speech_frames:
                results.append(VadResult(segment=segment, level=level))
            else:
                logger.debug("VAD: фраза слишком короткая (%d кадров) — пропускаю", speech_frames)
                results.append(VadResult(level=level))

        return results
