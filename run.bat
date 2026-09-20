@echo off
chcp 65001 >nul
cd /d "%~dp0"
title hermes-voice

echo.
echo  hermes-voice - голосовой слой для Hermes Agent
echo  Каталог: %CD%
echo  Адрес:   http://localhost:8643
echo.

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"

rem --- ищем uv: он быстрее и не требует pip внутри venv ---
set "UV="
where uv >nul 2>&1 && set "UV=uv"
if not defined UV if exist "%LOCALAPPDATA%\hermes\bin\uv.exe" set "UV=%LOCALAPPDATA%\hermes\bin\uv.exe"

rem --- виртуальное окружение ---
if not exist "%PY%" (
    if defined UV (
        echo  Создаю окружение через uv...
        "%UV%" venv "%VENV%"
    ) else (
        echo  Создаю окружение через python -m venv...
        python -m venv "%VENV%"
    )
    if errorlevel 1 (
        echo  [ОШИБКА] Не удалось создать окружение. Проверьте установку Python.
        pause
        exit /b 1
    )
)

rem --- зависимости ---
"%PY%" -c "import fastapi, faster_whisper, edge_tts, piper" >nul 2>&1
if errorlevel 1 (
    echo  Устанавливаю зависимости ^(первый раз это займёт несколько минут^)...
    if defined UV (
        set "VIRTUAL_ENV=%CD%\%VENV%"
        "%UV%" pip install -r requirements.txt
    ) else (
        "%PY%" -m pip install --upgrade pip
        "%PY%" -m pip install -r requirements.txt
    )
    if errorlevel 1 (
        echo  [ОШИБКА] Установка зависимостей не удалась.
        pause
        exit /b 1
    )
)

rem --- голоса piper: без них синтез не заработает ---
if not exist "voices\*.onnx" (
    echo  Скачиваю русские голоса Piper ^(~250 МБ^)...
    if not exist "voices" mkdir voices
    for %%v in (irina dmitri ruslan denis) do (
        echo    ru_RU-%%v-medium
        curl -sL -o "voices\ru_RU-%%v-medium.onnx" "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/%%v/medium/ru_RU-%%v-medium.onnx"
        curl -sL -o "voices\ru_RU-%%v-medium.onnx.json" "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/%%v/medium/ru_RU-%%v-medium.onnx.json"
    )
)

if not exist "config.json" (
    echo  [ВНИМАНИЕ] config.json не найден - ключ Hermes возьмётся из переменной HV_HERMES_KEY.
)

echo  Запускаю сервис. Для остановки - Ctrl+C.
echo.
"%PY%" -m uvicorn app.main:app --host 0.0.0.0 --port 8643

echo.
echo  Сервис остановлен.
pause
