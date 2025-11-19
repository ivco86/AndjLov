"""
Telegram bot routes - bot management, configuration, sending photos
"""

import os
import sys
import time
import subprocess
import threading
import asyncio
from flask import Blueprint, jsonify, request
from shared import (
    db, TELEGRAM_BOT_CONFIG_PATH, TELEGRAM_BOT_LOG_FILE,
    HAS_TELEGRAM, DATA_DIR, telegram_bot_process
)

telegram_bp = Blueprint('telegram', __name__)


def log_bot_output(stream, stream_name, log_file):
    """Read bot output and log it"""
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            for line in iter(stream.readline, b''):
                if not line:
                    break
                decoded_line = line.decode('utf-8', errors='replace').rstrip()
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                log_line = f"[{timestamp}] [{stream_name}] {decoded_line}\n"
                f.write(log_line)
                f.flush()
                # Also print to console
                print(f"[BOT {stream_name}] {decoded_line}")
    except Exception as e:
        print(f"Error logging bot output: {e}")
    finally:
        stream.close()


def start_telegram_bot():
    """Start Telegram bot as subprocess"""
    import shared

    if shared.telegram_bot_process and shared.telegram_bot_process.poll() is None:
        return {'success': False, 'message': 'Bot is already running'}

    # Check if bot token is configured
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    if not bot_token:
        # Try to load from .env file
        if os.path.exists(TELEGRAM_BOT_CONFIG_PATH):
            with open(TELEGRAM_BOT_CONFIG_PATH, 'r') as f:
                for line in f:
                    if line.startswith('TELEGRAM_BOT_TOKEN='):
                        bot_token = line.split('=', 1)[1].strip()
                        break

    if not bot_token:
        return {'success': False, 'message': 'TELEGRAM_BOT_TOKEN not configured'}

    try:
        # Create log file
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(TELEGRAM_BOT_LOG_FILE, 'w') as f:
            f.write(f"=== Telegram Bot Log Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

        # Prepare environment variables
        bot_env = os.environ.copy()
        bot_env['TELEGRAM_BOT_TOKEN'] = bot_token
        bot_env['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering

        # Start bot as subprocess using current Python interpreter
        shared.telegram_bot_process = subprocess.Popen(
            [sys.executable, 'telegram_bot.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=bot_env,
            bufsize=0  # Unbuffered
        )

        # Start threads to capture output
        stdout_thread = threading.Thread(
            target=log_bot_output,
            args=(shared.telegram_bot_process.stdout, 'STDOUT', TELEGRAM_BOT_LOG_FILE),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=log_bot_output,
            args=(shared.telegram_bot_process.stderr, 'STDERR', TELEGRAM_BOT_LOG_FILE),
            daemon=True
        )

        stdout_thread.start()
        stderr_thread.start()

        # Wait a moment to check if bot starts successfully
        time.sleep(2)

        # Check if process is still running
        if shared.telegram_bot_process.poll() is not None:
            # Process exited immediately - probably an error
            return {'success': False, 'message': 'Bot exited immediately. Check logs for errors.'}

        print(f"✅ Telegram bot started (PID: {shared.telegram_bot_process.pid})")
        return {'success': True, 'message': f'Bot started (PID: {shared.telegram_bot_process.pid})'}
    except Exception as e:
        print(f"❌ Failed to start Telegram bot: {e}")
        return {'success': False, 'message': f'Failed to start bot: {str(e)}'}


def stop_telegram_bot():
    """Stop Telegram bot subprocess"""
    import shared

    if not shared.telegram_bot_process or shared.telegram_bot_process.poll() is not None:
        shared.telegram_bot_process = None
        return {'success': False, 'message': 'Bot is not running'}

    try:
        shared.telegram_bot_process.terminate()
        shared.telegram_bot_process.wait(timeout=5)
        pid = shared.telegram_bot_process.pid
        shared.telegram_bot_process = None

        print(f"✅ Telegram bot stopped (PID: {pid})")
        return {'success': True, 'message': f'Bot stopped (PID: {pid})'}
    except subprocess.TimeoutExpired:
        shared.telegram_bot_process.kill()
        shared.telegram_bot_process.wait()
        pid = shared.telegram_bot_process.pid
        shared.telegram_bot_process = None
        return {'success': True, 'message': f'Bot forcefully killed (PID: {pid})'}
    except Exception as e:
        print(f"❌ Failed to stop Telegram bot: {e}")
        return {'success': False, 'message': f'Failed to stop bot: {str(e)}'}


def get_telegram_bot_status():
    """Get Telegram bot status"""
    import shared

    is_running = shared.telegram_bot_process and shared.telegram_bot_process.poll() is None

    # Get bot configuration
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    if not bot_token and os.path.exists(TELEGRAM_BOT_CONFIG_PATH):
        with open(TELEGRAM_BOT_CONFIG_PATH, 'r') as f:
            for line in f:
                if line.startswith('TELEGRAM_BOT_TOKEN='):
                    bot_token = line.split('=', 1)[1].strip()
                    break

    auto_analyze = os.environ.get('AUTO_ANALYZE', 'true').lower() == 'true'
    ai_style = os.environ.get('AI_STYLE', 'classic')

    return {
        'running': is_running,
        'pid': shared.telegram_bot_process.pid if is_running else None,
        'configured': bool(bot_token),
        'auto_analyze': auto_analyze,
        'ai_style': ai_style
    }


@telegram_bp.route('/api/telegram/status', methods=['GET'])
def telegram_status():
    """Get Telegram bot status"""
    status = get_telegram_bot_status()
    return jsonify(status)


@telegram_bp.route('/api/telegram/start', methods=['POST'])
def telegram_start():
    """Start Telegram bot"""
    result = start_telegram_bot()
    return jsonify(result), 200 if result['success'] else 400


@telegram_bp.route('/api/telegram/stop', methods=['POST'])
def telegram_stop():
    """Stop Telegram bot"""
    result = stop_telegram_bot()
    return jsonify(result), 200 if result['success'] else 400


@telegram_bp.route('/api/telegram/config', methods=['GET', 'POST'])
def telegram_config():
    """Get or update Telegram bot configuration"""
    if request.method == 'GET':
        config = {}
        if os.path.exists(TELEGRAM_BOT_CONFIG_PATH):
            with open(TELEGRAM_BOT_CONFIG_PATH, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key] = value

        return jsonify({
            'config': config,
            'file_path': TELEGRAM_BOT_CONFIG_PATH
        })

    elif request.method == 'POST':
        data = request.json
        bot_token = data.get('bot_token', '')
        auto_analyze = data.get('auto_analyze', 'true')
        ai_style = data.get('ai_style', 'classic')

        # Update .env file
        config_lines = []
        if os.path.exists(TELEGRAM_BOT_CONFIG_PATH):
            with open(TELEGRAM_BOT_CONFIG_PATH, 'r') as f:
                config_lines = f.readlines()

        # Update or add configuration
        updated = {
            'TELEGRAM_BOT_TOKEN': False,
            'AUTO_ANALYZE': False,
            'AI_STYLE': False
        }

        for i, line in enumerate(config_lines):
            if line.startswith('TELEGRAM_BOT_TOKEN='):
                config_lines[i] = f"TELEGRAM_BOT_TOKEN={bot_token}\n"
                updated['TELEGRAM_BOT_TOKEN'] = True
            elif line.startswith('AUTO_ANALYZE='):
                config_lines[i] = f"AUTO_ANALYZE={auto_analyze}\n"
                updated['AUTO_ANALYZE'] = True
            elif line.startswith('AI_STYLE='):
                config_lines[i] = f"AI_STYLE={ai_style}\n"
                updated['AI_STYLE'] = True

        # Add missing configurations
        if not updated['TELEGRAM_BOT_TOKEN']:
            config_lines.append(f"TELEGRAM_BOT_TOKEN={bot_token}\n")
        if not updated['AUTO_ANALYZE']:
            config_lines.append(f"AUTO_ANALYZE={auto_analyze}\n")
        if not updated['AI_STYLE']:
            config_lines.append(f"AI_STYLE={ai_style}\n")

        # Write back
        with open(TELEGRAM_BOT_CONFIG_PATH, 'w') as f:
            f.writelines(config_lines)

        # Update environment variables
        os.environ['TELEGRAM_BOT_TOKEN'] = bot_token
        os.environ['AUTO_ANALYZE'] = auto_analyze
        os.environ['AI_STYLE'] = ai_style

        return jsonify({
            'success': True,
            'message': 'Configuration updated'
        })


@telegram_bp.route('/api/telegram/logs', methods=['GET'])
def telegram_logs():
    """Get Telegram bot logs"""
    lines = request.args.get('lines', 100, type=int)  # Get last N lines

    if not os.path.exists(TELEGRAM_BOT_LOG_FILE):
        return jsonify({
            'logs': '',
            'message': 'No log file found'
        })

    try:
        with open(TELEGRAM_BOT_LOG_FILE, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            # Get last N lines
            log_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            logs = ''.join(log_lines)

        return jsonify({
            'logs': logs,
            'total_lines': len(all_lines),
            'returned_lines': len(log_lines)
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'logs': ''
        }), 500


@telegram_bp.route('/api/telegram/send-photo', methods=['POST'])
def telegram_send_photo():
    """Send a photo from gallery to Telegram chat"""
    if not HAS_TELEGRAM:
        return jsonify({
            'success': False,
            'error': 'Telegram library not installed'
        }), 500

    data = request.json
    image_id = data.get('image_id')
    chat_id = data.get('chat_id')
    caption = data.get('caption', '')

    if not image_id:
        return jsonify({
            'success': False,
            'error': 'image_id is required'
        }), 400

    if not chat_id:
        return jsonify({
            'success': False,
            'error': 'chat_id is required'
        }), 400

    # Get bot token
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    if not bot_token and os.path.exists(TELEGRAM_BOT_CONFIG_PATH):
        with open(TELEGRAM_BOT_CONFIG_PATH, 'r') as f:
            for line in f:
                if line.startswith('TELEGRAM_BOT_TOKEN='):
                    bot_token = line.split('=', 1)[1].strip()
                    break

    if not bot_token:
        return jsonify({
            'success': False,
            'error': 'TELEGRAM_BOT_TOKEN not configured'
        }), 400

    # Get image from database
    image = db.get_image(image_id)
    if not image:
        return jsonify({
            'success': False,
            'error': 'Image not found'
        }), 404

    filepath = image['filepath']
    if not os.path.exists(filepath):
        return jsonify({
            'success': False,
            'error': 'Image file not found on disk'
        }), 404

    # Send photo or video using Telegram Bot API
    media_type = image.get('media_type', 'image')

    try:
        from telegram import Bot

        async def send_media_async():
            bot = Bot(token=bot_token)
            with open(filepath, 'rb') as media_file:
                if media_type == 'video':
                    message = await bot.send_video(
                        chat_id=int(chat_id),
                        video=media_file,
                        caption=caption if caption else None,
                        parse_mode='Markdown' if caption else None
                    )
                else:
                    message = await bot.send_photo(
                        chat_id=int(chat_id),
                        photo=media_file,
                        caption=caption if caption else None,
                        parse_mode='Markdown' if caption else None
                    )
            return message

        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            message = loop.run_until_complete(send_media_async())
            return jsonify({
                'success': True,
                'message_id': message.message_id,
                'chat_id': message.chat_id,
                'file_sent': os.path.basename(filepath),
                'media_type': media_type
            })
        finally:
            loop.close()

    except Exception as e:
        print(f"❌ Error sending {media_type} to Telegram: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
