"""Разбор задержки первого звука: агент → первая фраза → первый байт TTS.

Запуск (из корня проекта, внутри venv):
    .venv\\Scripts\\python.exe tools\\bench_latency.py "Твой вопрос"
"""

from __future__ import annotations

import asyncio
import sys
import time

sys.path.insert(0, ".")

from app.config import ROOT, settings                 # noqa: E402
from app.hermes import HermesClient, HermesError      # noqa: E402
from app.tts import TextToSpeech, text_chunks         # noqa: E402

QUESTION = sys.argv[1] if len(sys.argv) > 1 else (
    "Расскажи в четырёх-пяти предложениях, что такое векторная база данных."
)


async def main() -> None:
    hermes = HermesClient(
        base_url=settings.hermes_base_url,
        api_key=settings.hermes_api_key,
        model=settings.hermes_model,
        timeout=settings.hermes_timeout,
    )
    tts = TextToSpeech(
        engine=settings.tts_engine,
        voice=settings.tts_voice,
        rate=settings.tts_rate,
        volume=settings.tts_volume,
        piper_voices_dir=ROOT / settings.tts_voices_dir,
        piper_rate=settings.tts_piper_rate,
    )

    # Прогрев: иначе в замер попадёт загрузка голоса (~4 с), которой в живом
    # сервисе нет — она сделана на старте.
    t0 = time.time()
    await tts.warmup()
    print(f"  прогрев голоса      : {time.time() - t0:6.2f} с")

    t0 = time.time()
    first_delta_at: float | None = None
    first_piece_at: float | None = None
    first_audio_at: float | None = None
    chars = 0

    queue: asyncio.Queue = asyncio.Queue()

    async def pump() -> None:
        nonlocal first_delta_at, chars
        try:
            async for delta in hermes.stream_reply(QUESTION, session_id="bench-latency"):
                if first_delta_at is None:
                    first_delta_at = time.time() - t0
                chars += len(delta)
                await queue.put(delta)
        except HermesError as exc:
            print("HermesError:", exc)
        finally:
            await queue.put(None)

    async def deltas():
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item

    async def speak() -> None:
        nonlocal first_piece_at, first_audio_at
        async for piece in text_chunks(deltas()):
            now = time.time() - t0
            if first_piece_at is None:
                first_piece_at = now
                print(f"  первый кусок текста : {now:6.2f} с  ({len(piece)} симв.) {piece[:70]!r}")
            else:
                print(f"  кусок               : {now:6.2f} с  ({len(piece)} симв.)")
            async for audio in tts.stream_audio(piece):
                if first_audio_at is None:
                    first_audio_at = time.time() - t0
                    print(f"  первый байт TTS     : {first_audio_at:6.2f} с")
                break   # для замера нужен только первый байт куска
        print(f"  конец потока        : {time.time() - t0:6.2f} с  (всего {chars} симв.)")

    await asyncio.gather(pump(), speak())

    print("\n--- итог ---")
    print(f"первая дельта агента : {first_delta_at:.2f} с" if first_delta_at else "агент не ответил")
    if first_piece_at:
        print(f"первая фраза целиком : {first_piece_at:.2f} с")
    if first_audio_at:
        print(f"ПЕРВЫЙ ЗВУК           : {first_audio_at:.2f} с")


if __name__ == "__main__":
    asyncio.run(main())
