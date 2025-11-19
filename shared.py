"""
Shared objects and configuration for AI Gallery
Initialized once and imported by all modules
"""

import os
import json
from database import Database
from ai_service import AIService

# Configuration from environment variables
PHOTOS_DIR = os.environ.get('PHOTOS_DIR', './photos')
DATA_DIR = os.environ.get('DATA_DIR', 'data')
LM_STUDIO_URL = os.environ.get('LM_STUDIO_URL', 'http://localhost:1234')
DATABASE_PATH = os.environ.get('DATABASE_PATH', 'data/gallery.db')
EXTERNAL_APPS_CONFIG = os.path.join(DATA_DIR, 'external_apps.json')

# Telegram bot configuration
TELEGRAM_BOT_CONFIG_PATH = '.env'
TELEGRAM_BOT_LOG_FILE = os.path.join(DATA_DIR, 'telegram_bot.log')

# Try to import telegram library
try:
    from telegram import Bot
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False
    print("Warning: python-telegram-bot not installed. Telegram photo sending will be disabled.")

# Initialize shared services
db = Database(DATABASE_PATH)
ai = AIService(LM_STUDIO_URL)

# Telegram bot process (will be managed by telegram routes)
telegram_bot_process = None


def load_external_apps():
    """Load external apps configuration from JSON file"""
    try:
        if os.path.exists(EXTERNAL_APPS_CONFIG):
            with open(EXTERNAL_APPS_CONFIG, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading external apps config: {e}")

    # Default configuration if file doesn't exist
    return {
        'image': [
            {'id': 'system', 'name': 'System Default', 'command': 'system', 'path': '', 'enabled': True}
        ],
        'video': [
            {'id': 'system', 'name': 'System Default', 'command': 'system', 'path': '', 'enabled': True}
        ]
    }


def save_external_apps(apps_config):
    """Save external apps configuration to JSON file"""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(EXTERNAL_APPS_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(apps_config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving external apps config: {e}")
        return False


# Load external apps configuration
EXTERNAL_APPS = load_external_apps()
