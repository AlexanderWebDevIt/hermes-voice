"""Проверка передискретизации на сервере.

Телефон не обязан отдавать 16 кГц: Android часто навязывает 48. Клиент сообщает
фактическую частоту в команде `start`, сервер приводит поток к 16 кГц сам.
Здесь это проверяется от начала до конца: синтезируем фразу, поднимаем её до
48 кГц, отправляем как «микрофон 48 кГц» и смотрим, что распозналось.

Запуск (из корня проекта, внутри venv):
    .venv\\Scripts\\python.exe tools\\test_resample.py
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, ".")

BASE = "http://127.0.0.1:8643"
PHRASE = "Привет, Гермес. Проверка частоты дискретизации."


def decode_wav(blob: bytes) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(io.BytesIO(blob)) as wav:
        rate = wav.getframerate()
        pcm = wav.readframes(wav.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32), rate


def resample(samples: np.ndarray, src: int, dst: int) -> np.ndarray:
    count = int(len(samples) * dst / src)
    return np.interp(
        np.linspace(0, len(samples) - 1, count), np.arange(len(samples)), samples
    ).astype("<i2")


async def cycle(pcm16: bytes, declared_rate: int, label: str) -> None:
    import websockets

    url = BASE.replace("http://", "ws://") + "/v1/voice"
    chunk = 1280 * max(1, declared_rate // 16000)   # тот же шаг ~40 мс

    async with websockets.connect(url, max_size=None) as ws:
        await ws.recv()                                   # hello
        await ws.send(json.dumps({"type": "start", "sample_rate": declared_rate}))

        async def pump() -> None:
            for i in range(0, len(pcm16), chunk):
                await ws.send(pcm16[i : i + chunk])
                await asyncio.sleep(0.04)
            silence = b"\x00" * chunk
            for _ in range(60):
                await ws.send(silence)
                await asyncio.sleep(0.04)

        task = asyncio.create_task(pump())
        transcript = None
        audio = 0
        t0 = time.time()
        try:
            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=90)
                if isinstance(message, (bytes, bytearray)):
                    audio += len(message)
                    continue
                event = json.loads(message)
                if event["type"] == "transcript":
                    transcript = event["text"]
                elif event["type"] == "assistant_done":
                    await asyncio.sleep(0.8)
                    break
                elif event["type"] == "error":
                    print(f"  {label}: ОШИБКА {event['message']}")
                    break
        except asyncio.TimeoutError:
            print(f"  {label}: таймаут")
        finally:
            task.cancel()

        mark = "OK " if transcript and "частот" in transcript.lower() else "?? "
        print(f"  {mark}{label:26} → {transcript!r}  ({time.time() - t0:.1f} с, {audio} байт)")


async def main() -> None:
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(f"{BASE}/v1/audio/speech", json={"input": PHRASE})
        resp.raise_for_status()
        wav = resp.content

    samples, rate = decode_wav(wav)
    print(f"эталон  : {PHRASE!r}")
    print(f"источник: {rate} Гц, {len(samples) / rate:.1f} с\n")

    print("--- WebSocket ---")
    # 1. Как есть, 16 кГц — сервер должен пропустить без изменений.
    await cycle(resample(samples, rate, 16000).tobytes(), 16000, "16 кГц (passthrough)")
    # 2. 48 кГц — как отдаёт Android, если не послушался.
    await cycle(resample(samples, rate, 48000).tobytes(), 48000, "48 кГц (ресамплинг)")
    # 3. 44.1 кГц — нецелое отношение, самый неприятный случай.
    await cycle(resample(samples, rate, 44100).tobytes(), 44100, "44.1 кГц (нецелое)")


if __name__ == "__main__":
    asyncio.run(main())
