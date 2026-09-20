"""Самопроверка hermes-voice.

Прогоняет полный круг без микрофона и телефона:
    текст → TTS → аудио → faster-whisper → текст → Hermes → ответ

Запуск (из корня проекта, внутри venv):
    .venv\\Scripts\\python.exe tools\\smoke_test.py
    .venv\\Scripts\\python.exe tools\\smoke_test.py --ws    (плюс проверка WebSocket)
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time

import httpx

BASE = "http://127.0.0.1:8643"
PHRASE = "Привет, Гермес! Ответь одним коротким предложением."
WITH_WS = "--ws" in sys.argv


def audio_to_pcm16(blob: bytes, rate: int = 16000) -> bytes:
    """Декодировать аудио в сырой PCM 16 кГц моно — то, что клиент шлёт в сокет.

    Формат определяется по магическим байтам: сервис может отдавать WAV (piper)
    или MP3 (edge), и тест не должен об этом знать заранее.
    """
    if blob[:4] == b"RIFF":
        import wave

        import numpy as np

        with wave.open(io.BytesIO(blob)) as wav:
            assert wav.getsampwidth() == 2, "ожидается 16-битный PCM"
            assert wav.getnchannels() == 1, "ожидается моно"
            src_rate = wav.getframerate()
            pcm = wav.readframes(wav.getnframes())

        if src_rate == rate:
            return pcm
        # Piper отдаёт 22050 Гц, а VAD и микрофон телефона работают на 16 кГц.
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        count = int(len(samples) * rate / src_rate)
        resampled = np.interp(
            np.linspace(0, len(samples) - 1, count), np.arange(len(samples)), samples
        )
        return resampled.astype("<i2").tobytes()

    import av

    container = av.open(io.BytesIO(blob))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
    out = bytearray()
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            out += resampled.to_ndarray().tobytes()
    container.close()
    return bytes(out)


async def check_rest() -> bytes:
    """REST-эндпоинты: health, TTS, STT. Возвращает аудио для WS-теста."""
    async with httpx.AsyncClient(timeout=180) as client:
        health = (await client.get(f"{BASE}/health")).json()
        print("health       :", json.dumps(health, ensure_ascii=False))

        started = time.time()
        resp = await client.post(f"{BASE}/v1/audio/speech", json={"input": PHRASE})
        resp.raise_for_status()
        audio = resp.content
        kind = "WAV" if audio[:4] == b"RIFF" else "MP3"
        print(f"TTS          : {len(audio)} байт {kind} за {time.time() - started:.2f} с")

        started = time.time()
        resp = await client.post(
            f"{BASE}/v1/audio/transcriptions",
            files={"file": ("probe.wav", audio, "audio/wav")},
        )
        resp.raise_for_status()
        payload = resp.json()
        print(f"STT          : {time.time() - started:.2f} с → {payload['text']!r}")

        print(f"эталон       : {PHRASE!r}")
        return audio


async def check_ws(audio: bytes) -> None:
    """Полный голосовой цикл через WebSocket."""
    import websockets

    pcm = audio_to_pcm16(audio)
    print(f"PCM          : {len(pcm)} байт ({len(pcm) / 2 / 16000:.1f} с)")

    url = BASE.replace("http://", "ws://") + "/v1/voice"
    chunk = 1280  # 40 мс

    async with websockets.connect(url, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        print("hello        :", json.dumps(hello, ensure_ascii=False))

        await ws.send(json.dumps({"type": "start"}))

        async def pump() -> None:
            # Речь + хвост тишины, чтобы VAD закрыл фразу сам.
            for i in range(0, len(pcm), chunk):
                await ws.send(pcm[i : i + chunk])
                await asyncio.sleep(0.04)
            silence = b"\x00" * chunk
            for _ in range(50):   # 2 с тишины: VAD должен успеть закрыть фразу сам
                await ws.send(silence)
                await asyncio.sleep(0.04)

        pump_task = asyncio.create_task(pump())
        audio_bytes = 0
        started = time.time()
        phrase_closed_at = None        # момент, когда сервер закончил распознавание
        first_audio_at = None          # первый звук вообще (может быть отбивкой)
        answer_audio_at = None         # первый звук собственно ответа
        segments: list[dict] = []      # границы кусков речи по событию speech_chunk
        current: dict | None = None
        stray_bytes = 0                # аудио, пришедшее вне куска — так быть не должно

        try:
            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=90)
                if isinstance(message, (bytes, bytearray)):
                    audio_bytes += len(message)
                    if current is not None:
                        current["bytes"] += len(message)
                        if not current.get("filler") and answer_audio_at is None:
                            answer_audio_at = time.time() - started
                    else:
                        stray_bytes += len(message)
                    if first_audio_at is None:
                        first_audio_at = time.time() - started
                    continue
                event = json.loads(message)
                kind = event.get("type")
                if kind == "state":
                    print(f"  состояние  : {event['value']}")
                elif kind == "transcript":
                    phrase_closed_at = time.time() - started
                    print(f"  распознано : {event['text']!r}  (фраза закрыта на {phrase_closed_at:.2f} с)")
                elif kind == "assistant_delta":
                    sys.stdout.write(event["text"])
                    sys.stdout.flush()
                elif kind == "assistant_done":
                    print(f"\n  ответ собран ({len(event['text'])} симв.)")
                elif kind == "speech_chunk":
                    if event.get("filler"):
                        print(f"  отбивка    : (тишина закрыта)")
                        current = {"seq": -1, "text": "", "bytes": 0, "filler": True}
                    else:
                        current = {"seq": event.get("seq"), "text": event.get("text"), "bytes": 0}
                    segments.append(current)
                    print(f"  кусок #{current['seq']}: {current['text']!r}")
                elif kind == "error":
                    print(f"  ОШИБКА     : {event['message']}")
                elif kind in {"level"}:
                    continue
        except asyncio.TimeoutError:
            print("  (таймаут ожидания событий)")
        finally:
            pump_task.cancel()

        print(f"аудио с сервера: {audio_bytes} байт")
        if first_audio_at is not None and phrase_closed_at is not None:
            # Честная метрика: не от старта отправки, а от конца фразы.
            print(f"ТИШИНА ПОСЛЕ ФРАЗЫ: {first_audio_at - phrase_closed_at:.2f} с "
                  f"(до первого звука — отбивка или ответ)")
        if answer_audio_at is not None and phrase_closed_at is not None:
            print(f"ОТВЕТ НАЧАЛСЯ    : {answer_audio_at - phrase_closed_at:.2f} с "
                  f"(от конца фразы до звука самого ответа)")
        if segments:
            print(f"  кусков речи  : {len(segments)}")
            for seg in segments:
                mark = "отбивка" if seg.get("filler") else f"#{seg['seq']}"
                print(f"    {mark:>8} {seg['bytes']:>7} байт  {seg['text'][:48]!r}")
        if stray_bytes:
            print(f"  ВНИМАНИЕ: {stray_bytes} байт аудио пришло вне speech_chunk")


async def main() -> None:
    try:
        audio = await check_rest()
    except Exception as exc:
        print(f"REST не прошёл: {type(exc).__name__}: {exc}")
        return

    if WITH_WS:
        print("\n--- WebSocket ---")
        try:
            await check_ws(audio)
        except Exception as exc:
            print(f"WS не прошёл: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
