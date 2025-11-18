#!/bin/bash

# Telegram Bot Startup Script for AI Gallery

echo "🤖 Starting AI Gallery Telegram Bot..."
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found!"
    echo "Creating .env from example..."
    cp .env.example .env
    echo ""
    echo "❗ Please edit .env and add your TELEGRAM_BOT_TOKEN"
    echo "   Get token from: https://t.me/botfather"
    echo ""
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if TELEGRAM_BOT_TOKEN is set
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "❌ Error: TELEGRAM_BOT_TOKEN not set in .env file"
    echo ""
    echo "How to get a token:"
    echo "1. Open @BotFather in Telegram"
    echo "2. Send /newbot"
    echo "3. Follow instructions"
    echo "4. Copy token to .env file"
    echo ""
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import telegram" 2>/dev/null; then
    echo "Installing python-telegram-bot..."
    pip install python-telegram-bot
fi

# Check if app.py is running
if ! curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
    echo "⚠️  Warning: AI Gallery app.py doesn't seem to be running"
    echo "   Start it first with: python app.py"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start bot
echo "🚀 Starting bot..."
echo ""
python3 telegram_bot.py
