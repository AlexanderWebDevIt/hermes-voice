"""Замер скорости распознавания на этой машине.

Подбирает рабочую комбинацию модели, типа вычислений и числа потоков.
Правило простое: для живого диалога распознавание должно идти БЫСТРЕЕ
реального времени (коэффициент < 1.0), иначе человек ждёт ответа.

Запуск:
    .venv\\Scripts\\python.exe tools\\bench_stt.py
"""

from __future__ import annotations

import asyncio
import io
import time

import numpy as np

PHRASE = "Привет, Гермес! Ответь одним коротким предложением."
RATE = 16000


async def make_audio() -> np.ndarray:
    """Сгенерировать тестовую фразу через edge-tts и декодировать в float32."""
    import av
    import edge_tts

    communicate = edge_tts.Communicate(PHRASE, "ru-RU-SvetlanaNeural")
    chunks = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio" and chunk.get("data"):
            chunks.append(chunk["data"])
    mp3 = b"".join(chunks)

    container = av.open(io.BytesIO(mp3))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
    pcm = bytearray()
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            pcm += resampled.to_ndarray().tobytes()
    container.close()

    audio = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0
    print(f"Тестовое аудио: {audio.size / RATE:.2f} с\n")
    return audio


def bench(model_name: str, compute: str, threads: int, audio: np.ndarray) -> tuple[float, str]:
    from faster_whisper import WhisperModel

    started = time.time()
    model = WhisperModel(model_name, device="cpu", compute_type=compute, cpu_threads=threads)
    load_time = time.time() - started

    started = time.time()
    segments, _ = model.transcribe(audio, language="ru", beam_size=1, vad_filter=False)
    text = "".join(s.text for s in segments).strip()
    infer_time = time.time() - started

    duration = audio.size / RATE
    ratio = infer_time / duration
    print(
        f"{model_name:6} {compute:14} потоков={threads}  "
        f"загрузка {load_time:5.1f} с  распознавание {infer_time:5.2f} с  "
        f"(x{ratio:4.2f} от длительности)  {text[:48]!r}"
    )
    del model
    return ratio, text


async def main() -> None:
    audio = await make_audio()
    results = []

    for model_name in ("base", "small"):
        for compute in ("int8", "float32"):
            for threads in (4, 8):
                try:
                    ratio, _ = bench(model_name, compute, threads, audio)
                    results.append((ratio, model_name, compute, threads))
                except Exception as exc:
                    print(f"{model_name:6} {compute:14} потоков={threads}  ОШИБКА: {exc}")

    if results:
        results.sort()
        best = results[0]
        print(
            f"\nЛучший вариант: модель={best[1]} compute={best[2]} потоки={best[3]} "
            f"(x{best[0]:.2f} от длительности)"
        )
        print("Для реального времени нужен коэффициент меньше 1.0.")


if __name__ == "__main__":
    asyncio.run(main())
