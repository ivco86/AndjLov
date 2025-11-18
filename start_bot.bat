@echo off
REM Telegram Bot Startup Script for AI Gallery (Windows)

echo 🤖 Starting AI Gallery Telegram Bot...
echo.

REM Check if .env file exists
if not exist .env (
    echo ⚠️  No .env file found!
    echo Creating .env from example...
    copy .env.example .env
    echo.
    echo ❗ Please edit .env and add your TELEGRAM_BOT_TOKEN
    echo    Get token from: https://t.me/botfather
    echo.
    pause
    exit /b 1
)

REM Load environment variables from .env
for /f "delims=" %%i in ('type .env ^| findstr /v "^#"') do set %%i

REM Check if TELEGRAM_BOT_TOKEN is set
if "%TELEGRAM_BOT_TOKEN%"=="" (
    echo ❌ Error: TELEGRAM_BOT_TOKEN not set in .env file
    echo.
    echo How to get a token:
    echo 1. Open @BotFather in Telegram
    echo 2. Send /newbot
    echo 3. Follow instructions
    echo 4. Copy token to .env file
    echo.
    pause
    exit /b 1
)

REM Check if requirements are installed
echo 📦 Checking dependencies...
python -c "import telegram" 2>nul
if errorlevel 1 (
    echo Installing python-telegram-bot...
    pip install python-telegram-bot
)

REM Start bot
echo 🚀 Starting bot...
echo.
python telegram_bot.py

pause
