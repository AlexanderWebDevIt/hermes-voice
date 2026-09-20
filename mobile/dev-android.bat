@echo off
setlocal

rem ---------------------------------------------------------------------------
rem Run the client on a USB-connected Android phone through Expo Go.
rem Metro (8081) and the voice server (8643) are forwarded to the device with
rem `adb reverse`, so neither Wi-Fi nor Tailscale is required.
rem ---------------------------------------------------------------------------

set "ADB=adb"
where adb >nul 2>nul || set "ADB=C:\Users\faktt\Documents\platform-tools\platform-tools\adb.exe"

"%ADB%" version >nul 2>nul
if errorlevel 1 (
  echo [X] adb not found. Install platform-tools or edit the ADB path in this file.
  exit /b 1
)

set "DEVICE="
for /f "skip=1 tokens=1,2" %%a in ('"%ADB%" devices') do (
  if not "%%b"=="" if "%DEVICE%"=="" set "DEVICE=%%a"
)

if "%DEVICE%"=="" (
  echo [X] No Android device detected.
  echo     Connect the phone by USB and enable USB debugging in Developer options.
  exit /b 1
)

echo [i] Device: %DEVICE%

"%ADB%" -s %DEVICE% reverse tcp:8081 tcp:8081
if errorlevel 1 exit /b 1
"%ADB%" -s %DEVICE% reverse tcp:8643 tcp:8643
if errorlevel 1 exit /b 1

echo [i] Forwarded: device 8081 -^> PC 8081 ^(Metro^), device 8643 -^> PC 8643 ^(voice^).
echo [i] In the app settings use host 127.0.0.1 and port 8643.
echo.

npx expo start --android --localhost
