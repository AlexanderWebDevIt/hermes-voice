"""hermes-voice — голосовой слой поверх Hermes Agent.

Сервис намеренно тонкий: он не подменяет агента и не хранит свою логику диалога.
Его работа — превратить речь в текст (faster-whisper), отдать текст в Hermes,
получить потоковый ответ и озвучить его (Piper) до того, как агент договорит.

Транспорт:
  WS   /v1/voice                    — realtime-диалог (основной режим)
  POST /v1/audio/transcriptions     — распознать файл (OpenAI-совместимо)
  POST /v1/audio/speech             — синтезировать текст (OpenAI-совместимо)
  GET  /v1/voices                   — список голосов
  GET  /health                      — живость сервиса
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from app.config import ROOT, settings
from app.direct import DirectLLM, DirectError, discover_model
from app.hermes import HermesClient, HermesError
from app.stt import SpeechToText
from app.tts import TextToSpeech, text_chunks
from app.vad import SpeechSegmenter, StreamResampler

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("hermes-voice")

app = FastAPI(title="hermes-voice", version="0.1.0")

stt = SpeechToText(
    model_name=settings.stt_model,
    device=settings.stt_device,
    compute_type=settings.stt_compute,
    language=settings.stt_language,
    beam_size=settings.stt_beam_size,
    cpu_threads=settings.stt_threads,
    initial_prompt=settings.stt_initial_prompt,
    idle_unload_seconds=settings.stt_idle_unload,
)
tts = TextToSpeech(
    engine=settings.tts_engine,
    voice=settings.tts_voice,
    rate=settings.tts_rate,
    volume=settings.tts_volume,
    piper_voices_dir=ROOT / settings.tts_voices_dir,
    piper_rate=settings.tts_piper_rate,
    first_chunk_chars=settings.tts_first_chunk_chars,
    chunk_chars=settings.tts_chunk_chars,
)
hermes = HermesClient(
    base_url=settings.hermes_base_url,
    api_key=settings.hermes_api_key,
    model=settings.hermes_model,
    timeout=settings.hermes_timeout,
)

# Прямой режим: модель без агентной обвязки. Явные настройки важнее —
# если их нет, берём модель из конфига Hermes, чтобы не дублировать руками.
_discovered = discover_model() or {}
direct = DirectLLM(
    base_url=settings.direct_base_url or _discovered.get("base_url", ""),
    api_key=settings.direct_api_key or _discovered.get("api_key", ""),
    model=settings.direct_model or _discovered.get("model", ""),
    max_tokens=settings.direct_max_tokens,
    temperature=settings.direct_temperature,
    timeout=settings.hermes_timeout,
)
if direct.enabled:
    logger.info("Прямой режим: %s (%s)", direct.model, direct.base_url)
else:
    logger.warning("Прямой режим недоступен: не нашёл модель. Останется только агент.")


@app.on_event("startup")
async def _warmup_on_startup() -> None:
    """Прогреть модели в фоне.

    Загрузка STT занимает около четырёх секунд, голоса Piper — примерно столько
    же. Если этого не сделать, за них заплатит первая же фраза пользователя —
    а в разговоре это заметно. Прямой режим дополнительно греет соединение с
    провайдером: холодный первый запрос стоит лишние секунды.
    """
    if settings.stt_warmup:
        asyncio.create_task(stt.warmup())
    if settings.tts_warmup:
        asyncio.create_task(tts.warmup())
        if settings.tts_preload_all:
            asyncio.create_task(tts.warmup_all())
    if settings.direct_warmup and direct.enabled:
        asyncio.create_task(direct.warmup())
    if settings.filler_enabled and settings.filler_phrases:
        asyncio.create_task(tts.prepare_fillers(settings.filler_phrases))


@app.on_event("shutdown")
async def _shutdown() -> None:
    await direct.aclose()


# ====================================================================== REST


@app.get("/health")
async def health() -> dict:
    """Живость сервиса и доступность Hermes."""
    return {
        "status": "ok",
        "hermes": await hermes.health(),
        "direct": direct.enabled,
        "backend": settings.voice_backend,
        "stt_model": settings.stt_model,
        "tts_engine": tts.engine_name,
        "tts_voice": settings.tts_voice,
        "tts_output": tts.describe_output(),
        "vad": settings.vad_engine,
    }


@app.get("/v1/voices")
async def voices() -> dict:
    """Голоса активного движка (piper — локальные, edge — облачные)."""
    return {
        "object": "list",
        "engine": tts.engine_name,
        "data": await tts.list_voices(),
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
) -> JSONResponse:
    """Распознать загруженный аудиофайл (wav, mp3, ogg, webm, m4a)."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Пустой файл")

    started = time.time()
    try:
        text = await stt.transcribe_bytes(raw, filename=file.filename or "audio.bin", language=language)
    except Exception as exc:
        logger.exception("STT: сбой распознавания файла")
        raise HTTPException(status_code=500, detail=f"Ошибка распознавания: {exc}") from exc

    return JSONResponse({"text": text, "elapsed": round(time.time() - started, 2)})


@app.post("/v1/audio/speech")
async def speech(payload: dict) -> Response:
    """Синтезировать текст (совместимо с OpenAI /v1/audio/speech).

    Формат ответа зависит от движка: piper — WAV, edge — MP3.
    """
    text = (payload.get("input") or payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Пустой текст")

    audio = await tts.synthesize(text, voice=payload.get("voice"))
    if not audio:
        raise HTTPException(status_code=502, detail="Синтез вернул пустой результат")
    media = "audio/wav" if tts.format == "wav" else "audio/mpeg"
    return Response(content=audio, media_type=media)


# ====================================================================== WebSocket


def _as_int(value, default: int) -> int:
    """Частота из JSON приходит числом, но клиент может прислать строкой."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class VoiceSession:
    """Один голосовой диалог: слушаю → думаю → говорю → слушаю."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.segmenter = SpeechSegmenter(
            sample_rate=settings.input_sample_rate,
            start_ratio=settings.vad_start_ratio,
            silence_ms=settings.vad_silence_ms,
            min_speech_ms=settings.vad_min_speech_ms,
            max_speech_ms=settings.vad_max_speech_ms,
            pre_roll_ms=settings.vad_pre_roll_ms,
        )
        self.voice = settings.tts_voice
        self.backend = settings.voice_backend if (settings.voice_backend != "direct" or direct.enabled) else "agent"
        self.session_id: Optional[str] = None
        self.history: list[dict] = []
        # Частота, которую реально отдаёт микрофон клиента. Может отличаться от
        # 16 кГц: Android часто навязывает 48 кГц. Узнаём из команды start.
        self.input_rate = settings.input_sample_rate
        self._resampler = StreamResampler(self.input_rate, settings.input_sample_rate)
        self._send_lock = asyncio.Lock()
        self._pipeline: Optional[asyncio.Task] = None
        self._active = False
        self._state = "idle"
        self._pending_segment = None
        self._last_level_sent = 0.0

    # -------------------------------------------------------------- отправка

    async def send_json(self, payload: dict) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(payload, ensure_ascii=False))

    async def send_audio(self, chunk: bytes) -> None:
        async with self._send_lock:
            await self.ws.send_bytes(chunk)

    async def set_state(self, value: str) -> None:
        self._state = value
        await self.send_json({"type": "state", "value": value})

    # -------------------------------------------------------------- конвейер

    def _cancel_pipeline(self) -> None:
        if self._pipeline and not self._pipeline.done():
            self._pipeline.cancel()
        self._pipeline = None

    async def _stream_answer(self, text: str):
        """Поток дельт ответа от выбранного бэкенда.

        direct — быстрый, без инструментов; agent — полный Hermes с инструментами.
        """
        if self.backend == "direct" and direct.enabled:
            async for delta in direct.stream_reply(text, history=self.history):
                yield delta
        else:
            async for delta in hermes.stream_reply(text, session_id=self.session_id):
                yield delta

    async def _respond(self, text: str) -> None:
        """Спросить бэкенд и озвучить ответ по мере генерации.

        Ключевая идея: текст не ждёт конца ответа. Как только набралось
        предложение — оно уходит в синтез и в сокет, пока модель продолжает
        печатать. Это и есть разница между «ответил через 10 секунд» и
        «начал говорить через 3».
        """
        queue: asyncio.Queue = asyncio.Queue()
        collected: list[str] = []

        async def pump_text() -> None:
            try:
                async for delta in self._stream_answer(text):
                    collected.append(delta)
                    await self.send_json({"type": "assistant_delta", "text": delta})
                    await queue.put(delta)
            except (HermesError, DirectError) as exc:
                logger.error("Бэкенд %s: %s", self.backend, exc)
                await self.send_json({"type": "error", "message": str(exc)})
            finally:
                await queue.put(None)

        async def speak() -> None:
            async def deltas():
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    yield item

            started_speaking = asyncio.Event()
            filler_done = asyncio.Event()

            async def filler() -> None:
                """Вставить отбивку, если ответ задерживается.

                Ждём первый кусок ответа. Не пришёл за filler_delay_ms — значит
                модель думает, и лучше сказать «так…», чем молчать. Пришёл
                раньше — отбивка не нужна и отменяется.
                """
                try:
                    await asyncio.wait_for(started_speaking.wait(), timeout=settings.filler_delay_ms / 1000)
                    return                       # ответ успел — тишины не было
                except asyncio.TimeoutError:
                    pass
                if not settings.filler_enabled:
                    return
                blob = tts.pick_filler()
                if blob is None:
                    return
                logger.info("Отбивка: модель думает дольше %d мс", settings.filler_delay_ms)
                # seq = -1 отделяет отбивку от нумерованных кусков ответа.
                await self.send_json({"type": "speech_chunk", "seq": -1, "text": "", "filler": True})
                await self.send_audio(blob)

            filler_task = asyncio.create_task(filler())

            first = True
            seq = 0
            try:
                async for piece in text_chunks(
                    deltas(),
                    first_chars=settings.tts_first_chunk_chars,
                    chunk_chars=settings.tts_chunk_chars,
                ):
                    if first:
                        await self.set_state("speaking")
                        started_speaking.set()
                        first = False
                    # Маркер границы: клиент играет каждый кусок отдельным файлом,
                    # поэтому ему нужно знать, где заканчивается предыдущий.
                    await self.send_json({"type": "speech_chunk", "seq": seq, "text": piece})
                    seq += 1
                    async for audio_chunk in tts.stream_audio(piece, voice=self.voice):
                        await self.send_audio(audio_chunk)
            finally:
                started_speaking.set()       # отпустить отбивку, если поток кончился
                filler_task.cancel()
                with suppress(asyncio.CancelledError):
                    await filler_task

        await asyncio.gather(pump_text(), speak())

        answer = "".join(collected).strip()
        if answer:
            logger.info("%s: %s", self.backend, answer[:200])
            self.history.extend([
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer},
            ])
            # Держим контекст ограниченным: голосовой диалог не должен расти бесконечно.
            self.history = self.history[-16:]
            await self.send_json({"type": "assistant_done", "text": answer})

    async def _run_pipeline(self, audio) -> None:
        """Фраза распознана → спросить модель → озвучить по мере генерации."""
        try:
            await self.set_state("thinking")

            text = await stt.transcribe(audio)
            if not text:
                await self.set_state("listening")
                return

            await self.send_json({"type": "transcript", "text": text, "final": True})
            logger.info("Пользователь: %s", text)

            await self._respond(text)
        except asyncio.CancelledError:
            logger.info("Конвейер прерван (barge-in)")
            raise
        except Exception as exc:
            logger.exception("Сбой конвейера")
            with suppress(Exception):
                await self.send_json({"type": "error", "message": str(exc)})
        finally:
            with suppress(Exception):
                await self.set_state("listening" if self._active else "idle")

            # Отложенная фраза: запускаем её только теперь, когда агент свободен.
            if self._pending_segment is not None and self._active:
                segment = self._pending_segment
                self._pending_segment = None
                self._pipeline = None      # чтобы не отменить самих себя
                self._start_pipeline(segment)

    def _start_pipeline(self, audio) -> None:
        self._cancel_pipeline()
        self._pipeline = asyncio.create_task(self._run_pipeline(audio))

    async def handle_text_message(self, raw: str) -> None:
        """Текстовые команды клиента."""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            await self.send_json({"type": "error", "message": "Некорректный JSON"})
            return

        kind = payload.get("type")

        if kind == "start":
            self._active = True
            self.segmenter.reset()
            if payload.get("voice"):
                self.voice = str(payload["voice"])
            if payload.get("session_id"):
                self.session_id = str(payload["session_id"])
            # Клиент сообщает фактическую частоту своего микрофона. Приводим поток
            # к 16 кГц сразу здесь, чтобы VAD и whisper получили то, что ждут.
            rate = _as_int(payload.get("sample_rate"), settings.input_sample_rate)
            if rate != self.input_rate:
                self.input_rate = rate
                self._resampler = StreamResampler(rate, settings.input_sample_rate)
                logger.info("Микрофон клиента: %d Гц → привожу к %d", rate, settings.input_sample_rate)
            await self.set_state("listening")

        elif kind == "stop":
            self._active = False
            self._cancel_pipeline()
            self.segmenter.reset()
            self._resampler.reset()
            await self.set_state("idle")

        elif kind == "interrupt":
            self._cancel_pipeline()
            self.segmenter.reset()
            if self._active:
                await self.set_state("listening")

        elif kind == "text":
            # content — основной ключ, text — алиас для удобства
            content = str(payload.get("content") or payload.get("text") or "").strip()
            if content:
                self._cancel_pipeline()
                self._pipeline = asyncio.create_task(self._answer_text(content))
            else:
                await self.send_json({"type": "error", "message": "Пустой текст"})

        elif kind == "ping":
            await self.send_json({"type": "pong"})

        elif kind == "config":
            changed = {}
            if payload.get("voice"):
                wanted = str(payload["voice"])
                known = tts.voice_names()
                # У piper голоса — это файлы, их список конечен и известен.
                # У edge список большой и тянется из сети, поэтому там не проверяем.
                if known and wanted not in known:
                    await self.send_json({
                        "type": "error",
                        "message": f"Неизвестный голос: {wanted}",
                        "voices": known,
                    })
                    return
                self.voice = wanted
                changed["voice"] = self.voice
            if payload.get("backend"):
                wanted = str(payload["backend"]).strip().lower()
                if wanted not in {"agent", "direct"}:
                    await self.send_json({"type": "error", "message": f"Неизвестный бэкенд: {wanted}"})
                    return
                if wanted == "direct" and not direct.enabled:
                    await self.send_json({"type": "error", "message": "Прямой режим не настроен на сервере"})
                    return
                self.backend = wanted
                changed["backend"] = self.backend
            if changed:
                await self.send_json({"type": "config", **changed})

        else:
            await self.send_json({"type": "error", "message": f"Неизвестная команда: {kind}"})

    async def _answer_text(self, content: str) -> None:
        """Ответ на текстовое сообщение (тот же конвейер, но без STT)."""
        try:
            await self.set_state("thinking")
            await self._respond(content)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Сбой текстового ответа")
            with suppress(Exception):
                await self.send_json({"type": "error", "message": str(exc)})
        finally:
            with suppress(Exception):
                await self.set_state("listening" if self._active else "idle")

    async def handle_audio(self, data: bytes) -> None:
        """Очередной кусок PCM: прогоняем через VAD и запускаем конвейер на готовой фразе."""
        if self._resampler is not None and not self._resampler.passthrough:
            data = self._resampler.feed(data)
            if not data:
                return       # накопили на один выходной сэмпл — ещё рано

        for event in self.segmenter.feed(data):
            busy = self._pipeline is not None and not self._pipeline.done()

            # Barge-in — только когда агент реально говорит. Если он ещё думает,
            # прерывать нечего: речь пользователя в этот момент не должна убивать
            # уже запущенный запрос (иначе теряется начало фразы).
            if event.speech_started and busy and self._state == "speaking":
                logger.info("Barge-in: прерываю озвучку")
                self._cancel_pipeline()
                await self.set_state("listening")

            if event.segment is not None and event.segment.size:
                if self._pipeline is not None and not self._pipeline.done():
                    # Не теряем фразу: обработаем её, как только агент освободится.
                    # Такое бывает, если человек сделал паузу длиннее порога VAD
                    # посреди мысли — фраза распалась на две.
                    logger.info("VAD: фраза пришла, пока агент занят — ставлю в очередь")
                    self._pending_segment = event.segment
                else:
                    self._start_pipeline(event.segment)

            # Индикатор громкости для «дышащей» кнопки в приложении — не чаще 10 раз в секунду.
            now = time.time()
            if now - self._last_level_sent > 0.1:
                self._last_level_sent = now
                await self.send_json({"type": "level", "value": round(min(event.level * 6, 1.0), 3)})

    async def close(self) -> None:
        self._active = False
        self._cancel_pipeline()


@app.websocket("/v1/voice")
async def voice_socket(ws: WebSocket) -> None:
    """Realtime-канал голосового диалога."""
    await ws.accept()
    session = VoiceSession(ws)
    await session.send_json({
        "type": "hello",
        "protocol": 1,
        "input": {
            "format": "pcm_s16le",
            "sample_rate": settings.input_sample_rate,
            "channels": settings.input_channels,
        },
        "output": {
            "format": tts.format,               # wav (piper) | mp3 (edge)
            "sample_rate": tts.sample_rate,
            "channels": tts.channels,
            "engine": tts.engine_name,
            "voices": tts.voice_names(),
        },
        "voice": session.voice,
        "backend": session.backend,
        "backends": ["agent"] + (["direct"] if direct.enabled else []),
        "stt_model": settings.stt_model,
    })
    logger.info("Голосовая сессия открыта")

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                await session.handle_audio(data)
            elif (text := message.get("text")) is not None:
                await session.handle_text_message(text)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Ошибка голосовой сессии")
    finally:
        await session.close()
        logger.info("Голосовая сессия закрыта")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level="debug" if settings.debug else "info",
    )
