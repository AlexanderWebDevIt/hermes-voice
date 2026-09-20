"""Регрессионный тест: согласование голоса между клиентом и сервером.

Ловит класс ошибок, который уже случался: клиент хранит голос, выбранный при
другом движке TTS, и сервер отвечает «Неизвестный голос» — озвучки нет вовсе.

Ожидаемое поведение:
  voice=''                     — молча проигнорирован (клиент «не выбирал»)
  voice=<имя другого движка>   — error + список допустимых голосов
  voice отсутствует            — молча проигнорирован
  voice=<верный голос>         — config с подтверждением

Запуск (сервис должен быть поднят):
  ./.venv/Scripts/python.exe tools/test_voice_config.py
"""

import asyncio
import json

import websockets

URL = "ws://127.0.0.1:8643/v1/voice"


async def probe(label: str, voice) -> None:
    async with websockets.connect(URL) as ws:
        hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
        known = hello["output"]["voices"]
        default = hello["voice"]

        payload = {"type": "config"}
        if voice is not None:
            payload["voice"] = voice
        await ws.send(json.dumps(payload))

        events = []
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.5)
                if isinstance(raw, (bytes, bytearray)):
                    continue
                data = json.loads(raw)
                events.append(data)
                if data.get("type") in ("config", "error"):
                    break
        except asyncio.TimeoutError:
            pass

        print(f"--- {label}: voice={voice!r}")
        print(f"    голосов на сервере: {len(known)}, по умолчанию: {default!r}")
        for event in events:
            print(f"    -> {json.dumps(event, ensure_ascii=False)}")
        if not events:
            print("    -> (ответа не было — команда проигнорирована молча)")
        print()


async def main() -> None:
    await probe("пустой голос (как теперь шлёт клиент)", "")
    await probe("голос от другого движка (edge-tts)", "ru-RU-SvetlanaNeural")
    await probe("голос без поля voice", None)
    await probe("верный голос Piper", "ru_RU-irina-medium")


asyncio.run(main())
