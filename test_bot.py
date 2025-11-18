#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script to check if Telegram bot can start
"""

import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("🔍 Telegram Bot Diagnostic Test")
print("=" * 60)
print()

# Check Python version
print(f"✓ Python version: {sys.version}")
print(f"✓ Platform: {sys.platform}")
print()

# Check required modules
print("Checking required modules...")
required_modules = [
    'telegram',
    'flask',
    'PIL',
    'requests'
]

missing_modules = []
for module in required_modules:
    try:
        __import__(module)
        print(f"  ✓ {module}")
    except ImportError as e:
        print(f"  ✗ {module} - MISSING!")
        missing_modules.append(module)

print()

if missing_modules:
    print("❌ Missing modules detected!")
    print("   Please install: pip install -r requirements.txt")
    print()
    sys.exit(1)

# Check environment variables
print("Checking environment variables...")
bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')

if bot_token:
    print(f"  ✓ TELEGRAM_BOT_TOKEN is set ({len(bot_token)} characters)")
else:
    print("  ⚠ TELEGRAM_BOT_TOKEN is NOT set")
    print("    You need to configure it in Settings or create .env file")

print(f"  • GALLERY_API_URL: {os.getenv('GALLERY_API_URL', 'http://localhost:5000')}")
print(f"  • PHOTOS_DIR: {os.getenv('PHOTOS_DIR', 'photos')}")
print(f"  • AUTO_ANALYZE: {os.getenv('AUTO_ANALYZE', 'true')}")
print(f"  • AI_STYLE: {os.getenv('AI_STYLE', 'classic')}")
print()

# Check directories
print("Checking directories...")
dirs_to_check = ['photos', 'data', 'data/thumbnails']
for dir_path in dirs_to_check:
    if os.path.exists(dir_path):
        print(f"  ✓ {dir_path}/")
    else:
        print(f"  ⚠ {dir_path}/ - does not exist (will be created)")
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"    ✓ Created {dir_path}/")
        except Exception as e:
            print(f"    ✗ Failed to create: {e}")

print()

# Test emoji printing
print("Testing emoji output...")
try:
    test_emojis = "🤖 📁 🌐 ✅ ❌ 🔍 ⚠️ 📊 🟢 🔴 🟡"
    print(f"  {test_emojis}")
    print("  ✓ Emoji output works!")
except Exception as e:
    print(f"  ✗ Emoji output failed: {e}")

print()

# Check if telegram bot can be imported
print("Testing Telegram bot import...")
try:
    from telegram_bot import TelegramGalleryBot
    print("  ✓ telegram_bot.py can be imported")

    if bot_token:
        print()
        print("Testing bot initialization...")
        try:
            bot = TelegramGalleryBot(bot_token)
            print("  ✓ Bot initialized successfully!")
            print()
            print("=" * 60)
            print("✅ ALL CHECKS PASSED!")
            print("=" * 60)
            print()
            print("You can now start the bot from the web UI.")
            print("If you still have issues, check the logs in Settings > View Logs")
        except Exception as e:
            print(f"  ✗ Bot initialization failed: {e}")
            print()
            print("Please check your bot token is valid.")
    else:
        print()
        print("=" * 60)
        print("⚠️  CONFIGURATION NEEDED")
        print("=" * 60)
        print()
        print("Steps to configure:")
        print("1. Start Flask app: python app.py")
        print("2. Open browser: http://localhost:5000")
        print("3. Click Settings (⚙️)")
        print("4. Get bot token from @BotFather on Telegram")
        print("5. Paste token and click Save Configuration")
        print("6. Click Start button")

except Exception as e:
    print(f"  ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()

print()
