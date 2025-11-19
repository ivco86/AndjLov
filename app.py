"""
AI Gallery - Flask Application
Main web server with REST API endpoints
"""

from flask import Flask, render_template, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import sys
import mimetypes
from PIL import Image, ImageDraw, ImageFont
import json
import subprocess
import signal
import atexit
import threading
import time
import io

# Try to import opencv for video frame extraction
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: opencv-python not installed. Video thumbnails will use placeholders.")

from database import Database
from ai_service import AIService
from privacy_service import get_privacy_service
from research_service import get_research_service
from pipeline_service import get_pipeline_service
from email_service import get_email_service
from scheduler_service import get_scheduler_service
from cloud_sync_service import get_cloud_sync_service

# Security helper function
def is_safe_path(filepath, base_dir):
    """
    Validate that a filepath is within the base directory.
    Handles Windows/Unix paths, symlinks, and relative paths.

    Args:
        filepath: The file path to validate
        base_dir: The base directory that should contain the file

    Returns:
        Tuple of (is_safe: bool, resolved_path: str)
    """
    try:
        # Normalize and resolve both paths to handle symlinks and relative paths
        # os.path.realpath resolves symlinks and normalizes path separators
        resolved_file = os.path.realpath(os.path.normpath(filepath))
        resolved_base = os.path.realpath(os.path.normpath(base_dir))

        # Ensure both use the same separator (important for cross-platform)
        resolved_file = os.path.normcase(resolved_file)
        resolved_base = os.path.normcase(resolved_base)

        # Check if the resolved file path is within the base directory
        # Use os.path.commonpath to ensure proper containment check
        try:
            common = os.path.commonpath([resolved_file, resolved_base])
            is_safe = os.path.normcase(common) == resolved_base
        except ValueError:
            # Paths are on different drives (Windows) or completely unrelated
            is_safe = False

        return is_safe, resolved_file
    except (ValueError, OSError) as e:
        # Invalid path or other error
        print(f"Path validation error: {e}")
        return False, None

# Helper functions for video processing
def extract_video_frame(video_path, output_path, time_sec=1.0):
    """Extract a frame from video using opencv if available"""
    if not HAS_OPENCV:
        return False

    try:
        cap = cv2.VideoCapture(video_path)

        # Set position to specified second
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(fps * time_sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read frame
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            return img
        return False
    except Exception as e:
        print(f"Error extracting video frame: {e}")
        return False

def create_video_placeholder(size=500):
    """Create a placeholder thumbnail for videos when opencv is not available"""
    img = Image.new('RGB', (size, int(size * 9/16)), color='#7b2cbf')

    # Add gradient effect
    draw = ImageDraw.Draw(img, 'RGBA')
    for i in range(img.height):
        alpha = int(255 * (1 - i / img.height))
        color = (255, 0, 110, alpha)
        draw.rectangle([(0, i), (img.width, i+1)], fill=color)

    # Add play icon
    center_x, center_y = img.width // 2, img.height // 2
    icon_size = 60

    # Draw white circle
    draw.ellipse(
        [(center_x - icon_size, center_y - icon_size),
         (center_x + icon_size, center_y + icon_size)],
        fill=(255, 255, 255, 230)
    )

    # Draw play triangle
    triangle = [
        (center_x - 20, center_y - 30),
        (center_x - 20, center_y + 30),
        (center_x + 30, center_y)
    ]
    draw.polygon(triangle, fill=(123, 44, 191))

    return img

def get_image_for_analysis(filepath, media_type='image'):
    """
    Get PIL Image for AI analysis
    For images: open directly
    For videos: extract frame at 1 second
    Returns: PIL Image object or None
    """
    if media_type == 'video':
        # Try to extract frame from video
        img = extract_video_frame(filepath, None, time_sec=1.0)
        if not img:
            # Fallback to placeholder if extraction fails
            print(f"Warning: Could not extract frame from video {filepath}, using placeholder")
            img = create_video_placeholder(800)
        return img
    else:
        # Regular image
        try:
            return Image.open(filepath)
        except Exception as e:
            print(f"Error opening image {filepath}: {e}")
            return None

# Configuration
PHOTOS_DIR = os.environ.get('PHOTOS_DIR', './photos')
DATA_DIR = os.environ.get('DATA_DIR', 'data')
LM_STUDIO_URL = os.environ.get('LM_STUDIO_URL', 'http://localhost:1234')
DATABASE_PATH = os.environ.get('DATABASE_PATH', 'data/gallery.db')

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
VIDEO_FORMATS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.m4v'}
ALL_MEDIA_FORMATS = SUPPORTED_FORMATS | VIDEO_FORMATS

# External applications configuration
EXTERNAL_APPS = {
    'image': [
        {'id': 'gimp', 'name': 'GIMP', 'command': 'gimp'},
        {'id': 'photoshop', 'name': 'Photoshop', 'command': 'photoshop'},
        {'id': 'krita', 'name': 'Krita', 'command': 'krita'},
        {'id': 'inkscape', 'name': 'Inkscape', 'command': 'inkscape'},
        {'id': 'illustrator', 'name': 'Illustrator', 'command': 'illustrator'},
        {'id': 'affinity', 'name': 'Affinity Photo', 'command': 'affinity-photo'},
        {'id': 'system', 'name': 'System Default', 'command': 'xdg-open'},
    ],
    'video': [
        {'id': 'vlc', 'name': 'VLC Player', 'command': 'vlc'},
        {'id': 'mpv', 'name': 'MPV Player', 'command': 'mpv'},
        {'id': 'kdenlive', 'name': 'Kdenlive', 'command': 'kdenlive'},
        {'id': 'davinci', 'name': 'DaVinci Resolve', 'command': 'davinci-resolve'},
        {'id': 'premiere', 'name': 'Premiere Pro', 'command': 'premiere'},
        {'id': 'ffmpeg', 'name': 'FFplay', 'command': 'ffplay'},
        {'id': 'system', 'name': 'System Default', 'command': 'xdg-open'},
    ]
}

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize services
db = Database(DATABASE_PATH)
ai = AIService(LM_STUDIO_URL)
research = get_research_service(db)
email = get_email_service(db)
pipeline = get_pipeline_service(db, ai, get_privacy_service(), research, email)
scheduler = get_scheduler_service(db, pipeline['executor'])
cloud_sync = get_cloud_sync_service(db, DATA_DIR)

# Load scheduled pipelines on startup
if scheduler:
    scheduler.reload_scheduled_pipelines()

# Check if auto-backup is needed
if cloud_sync:
    cloud_sync.auto_backup_if_needed()

# Telegram Bot Management
telegram_bot_process = None
telegram_bot_config_path = '.env'
telegram_bot_log_file = os.path.join(DATA_DIR, 'telegram_bot.log')

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
    global telegram_bot_process

    if telegram_bot_process and telegram_bot_process.poll() is None:
        return {'success': False, 'message': 'Bot is already running'}

    # Check if bot token is configured
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    if not bot_token:
        # Try to load from .env file
        if os.path.exists(telegram_bot_config_path):
            with open(telegram_bot_config_path, 'r') as f:
                for line in f:
                    if line.startswith('TELEGRAM_BOT_TOKEN='):
                        bot_token = line.split('=', 1)[1].strip()
                        break

    if not bot_token:
        return {'success': False, 'message': 'TELEGRAM_BOT_TOKEN not configured'}

    try:
        # Create log file
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(telegram_bot_log_file, 'w') as f:
            f.write(f"=== Telegram Bot Log Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

        # Prepare environment variables
        bot_env = os.environ.copy()
        bot_env['TELEGRAM_BOT_TOKEN'] = bot_token
        bot_env['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering

        # Start bot as subprocess using current Python interpreter
        telegram_bot_process = subprocess.Popen(
            [sys.executable, 'telegram_bot.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=bot_env,
            bufsize=0  # Unbuffered
        )

        # Start threads to capture output
        stdout_thread = threading.Thread(
            target=log_bot_output,
            args=(telegram_bot_process.stdout, 'STDOUT', telegram_bot_log_file),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=log_bot_output,
            args=(telegram_bot_process.stderr, 'STDERR', telegram_bot_log_file),
            daemon=True
        )

        stdout_thread.start()
        stderr_thread.start()

        # Wait a moment to check if bot starts successfully
        time.sleep(2)

        # Check if process is still running
        if telegram_bot_process.poll() is not None:
            # Process exited immediately - probably an error
            return {'success': False, 'message': 'Bot exited immediately. Check logs for errors.'}

        print(f"✅ Telegram bot started (PID: {telegram_bot_process.pid})")
        return {'success': True, 'message': f'Bot started (PID: {telegram_bot_process.pid})'}
    except Exception as e:
        print(f"❌ Failed to start Telegram bot: {e}")
        return {'success': False, 'message': f'Failed to start bot: {str(e)}'}

def stop_telegram_bot():
    """Stop Telegram bot subprocess"""
    global telegram_bot_process

    if not telegram_bot_process or telegram_bot_process.poll() is not None:
        telegram_bot_process = None
        return {'success': False, 'message': 'Bot is not running'}

    try:
        telegram_bot_process.terminate()
        telegram_bot_process.wait(timeout=5)
        pid = telegram_bot_process.pid
        telegram_bot_process = None

        print(f"✅ Telegram bot stopped (PID: {pid})")
        return {'success': True, 'message': f'Bot stopped (PID: {pid})'}
    except subprocess.TimeoutExpired:
        telegram_bot_process.kill()
        telegram_bot_process.wait()
        pid = telegram_bot_process.pid
        telegram_bot_process = None
        return {'success': True, 'message': f'Bot forcefully killed (PID: {pid})'}
    except Exception as e:
        print(f"❌ Failed to stop Telegram bot: {e}")
        return {'success': False, 'message': f'Failed to stop bot: {str(e)}'}

def get_telegram_bot_status():
    """Get Telegram bot status"""
    global telegram_bot_process

    is_running = telegram_bot_process and telegram_bot_process.poll() is None

    # Get bot configuration
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    if not bot_token and os.path.exists(telegram_bot_config_path):
        with open(telegram_bot_config_path, 'r') as f:
            for line in f:
                if line.startswith('TELEGRAM_BOT_TOKEN='):
                    bot_token = line.split('=', 1)[1].strip()
                    break

    auto_analyze = os.environ.get('AUTO_ANALYZE', 'true').lower() == 'true'
    ai_style = os.environ.get('AI_STYLE', 'classic')

    return {
        'running': is_running,
        'pid': telegram_bot_process.pid if is_running else None,
        'configured': bool(bot_token),
        'auto_analyze': auto_analyze,
        'ai_style': ai_style
    }

# Cleanup on exit
def cleanup_telegram_bot():
    """Cleanup Telegram bot on exit"""
    global telegram_bot_process
    if telegram_bot_process and telegram_bot_process.poll() is None:
        print("🛑 Stopping Telegram bot...")
        telegram_bot_process.terminate()
        telegram_bot_process.wait(timeout=5)

atexit.register(cleanup_telegram_bot)

# ============ FRONTEND ROUTES ============

@app.route('/')
def index():
    """Serve main application page"""
    return render_template('index.html')

# ============ SYSTEM API ============

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check system health and AI connection"""
    ai_connected, ai_message = ai.check_connection()
    stats = db.get_stats()
    
    return jsonify({
        'status': 'ok',
        'ai_connected': ai_connected,
        'ai_message': ai_message,
        'database': 'connected',
        'stats': stats
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify({
        'photos_dir': PHOTOS_DIR,
        'lm_studio_url': LM_STUDIO_URL,
        'supported_formats': list(SUPPORTED_FORMATS)
    })

@app.route('/api/ai/styles', methods=['GET'])
def get_ai_styles():
    """Get available AI description styles"""
    return jsonify({
        'styles': ai.get_available_styles()
    })

@app.route('/api/external-apps', methods=['GET'])
def get_external_apps():
    """Get list of external applications for opening images/videos"""
    return jsonify({
        'apps': EXTERNAL_APPS
    })

@app.route('/api/images/<int:image_id>/open-with', methods=['POST'])
def open_with_external_app(image_id):
    """Open image/video with external application"""
    import subprocess

    try:
        image = db.get_image(image_id)
        if not image:
            return jsonify({'error': 'Image not found'}), 404

        filepath = image['filepath']

        # Security: Validate filepath is within PHOTOS_DIR
        is_safe, resolved_path = is_safe_path(filepath, PHOTOS_DIR)

        if not is_safe:
            print(f"Security: Path traversal attempt blocked in open-with: {filepath}")
            return jsonify({'error': 'Invalid file path'}), 403

        if not os.path.exists(resolved_path):
            return jsonify({'error': 'File not found on disk'}), 404

        data = request.get_json() or {}
        app_id = data.get('app_id')

        if not app_id:
            return jsonify({'error': 'app_id is required'}), 400

        # Get media type
        media_type = image.get('media_type', 'image')

        # Find the application
        app_list = EXTERNAL_APPS.get(media_type, [])
        app = next((a for a in app_list if a['id'] == app_id), None)

        if not app:
            return jsonify({'error': f'Application {app_id} not found for {media_type}'}), 404

        # Check for custom path
        custom_paths_json = db.get_setting('external_app_paths')
        custom_paths = json.loads(custom_paths_json) if custom_paths_json else {}

        # Use custom path if configured, otherwise use default command
        app_command = custom_paths.get(media_type, {}).get(app_id, app['command'])

        # Launch application in background
        command = [app_command, resolved_path]

        print(f"[OPEN_WITH] Opening {resolved_path} with {app['name']} ({app_command})")

        # Start process in background (detached)
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )

        return jsonify({
            'success': True,
            'app': app['name'],
            'file': image['filename']
        })

    except FileNotFoundError:
        return jsonify({'error': f'Application not found. Make sure {app["command"]} is installed.'}), 404
    except Exception as e:
        print(f"Error opening with external app: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to open file: {str(e)}'}), 500

# ============ EXTERNAL APPS CONFIG API ============

@app.route('/api/external-apps/config', methods=['GET'])
def get_external_apps_config():
    """Get external apps configuration with custom paths"""
    try:
        # Get custom paths from settings
        custom_paths_json = db.get_setting('external_app_paths')
        custom_paths = json.loads(custom_paths_json) if custom_paths_json else {}

        # Merge with default apps and add custom paths
        config = {
            'image': [],
            'video': []
        }

        for media_type in ['image', 'video']:
            for app in EXTERNAL_APPS.get(media_type, []):
                app_config = app.copy()
                # Add custom path if configured
                custom_path = custom_paths.get(media_type, {}).get(app['id'])
                if custom_path:
                    app_config['custom_path'] = custom_path
                config[media_type].append(app_config)

        return jsonify({
            'success': True,
            'config': config,
            'custom_paths': custom_paths
        })
    except Exception as e:
        print(f"Error getting external apps config: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/external-apps/config', methods=['POST'])
def save_external_apps_config():
    """Save custom paths for external apps"""
    try:
        data = request.get_json() or {}
        custom_paths = data.get('custom_paths', {})

        # Validate structure
        for media_type in custom_paths:
            if media_type not in ['image', 'video']:
                return jsonify({'error': f'Invalid media type: {media_type}'}), 400

        # Save to database
        db.set_setting('external_app_paths', json.dumps(custom_paths))

        return jsonify({
            'success': True,
            'message': 'External app paths saved successfully'
        })
    except Exception as e:
        print(f"Error saving external apps config: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============ TELEGRAM BOT API ============

@app.route('/api/telegram/status', methods=['GET'])
def telegram_status():
    """Get Telegram bot status"""
    status = get_telegram_bot_status()
    return jsonify(status)

@app.route('/api/telegram/start', methods=['POST'])
def telegram_start():
    """Start Telegram bot"""
    result = start_telegram_bot()
    return jsonify(result), 200 if result['success'] else 400

@app.route('/api/telegram/stop', methods=['POST'])
def telegram_stop():
    """Stop Telegram bot"""
    result = stop_telegram_bot()
    return jsonify(result), 200 if result['success'] else 400

@app.route('/api/telegram/config', methods=['GET', 'POST'])
def telegram_config():
    """Get or update Telegram bot configuration"""
    if request.method == 'GET':
        config = {}
        if os.path.exists(telegram_bot_config_path):
            with open(telegram_bot_config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key] = value

        return jsonify({
            'config': config,
            'file_path': telegram_bot_config_path
        })

    elif request.method == 'POST':
        data = request.json
        bot_token = data.get('bot_token', '')
        auto_analyze = data.get('auto_analyze', 'true')
        ai_style = data.get('ai_style', 'classic')

        # Update .env file
        config_lines = []
        if os.path.exists(telegram_bot_config_path):
            with open(telegram_bot_config_path, 'r') as f:
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
        with open(telegram_bot_config_path, 'w') as f:
            f.writelines(config_lines)

        # Update environment variables
        os.environ['TELEGRAM_BOT_TOKEN'] = bot_token
        os.environ['AUTO_ANALYZE'] = auto_analyze
        os.environ['AI_STYLE'] = ai_style

        return jsonify({
            'success': True,
            'message': 'Configuration updated'
        })

@app.route('/api/telegram/logs', methods=['GET'])
def telegram_logs():
    """Get Telegram bot logs"""
    lines = request.args.get('lines', 100, type=int)  # Get last N lines

    if not os.path.exists(telegram_bot_log_file):
        return jsonify({
            'logs': '',
            'message': 'No log file found'
        })

    try:
        with open(telegram_bot_log_file, 'r', encoding='utf-8') as f:
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

# ============ IMAGE API ============

@app.route('/api/images', methods=['GET'])
def get_images():
    """Get all images with optional filters"""
    limit = request.args.get('limit', 1000, type=int)
    offset = request.args.get('offset', 0, type=int)
    favorites_only = request.args.get('favorites', 'false').lower() == 'true'
    
    media_type = request.args.get('media_type')
    if media_type:
        media_type = media_type.strip().lower()
        if media_type in ('all', 'any'):
            media_type = None
    
    analyzed_param = request.args.get('analyzed')
    analyzed = None
    if analyzed_param is not None:
        analyzed_param = analyzed_param.strip().lower()
        if analyzed_param in ('true', '1'):
            analyzed = True
        elif analyzed_param in ('false', '0'):
            analyzed = False

    images = db.get_all_images(
        limit=limit,
        offset=offset,
        favorites_only=favorites_only,
        media_type=media_type,
        analyzed=analyzed
    )

    return jsonify({
        'images': images,
        'count': len(images),
        'offset': offset,
        'limit': limit
    })

@app.route('/api/images/<int:image_id>', methods=['GET'])
def get_image(image_id):
    """Get single image details"""
    image = db.get_image(image_id)
    
    if not image:
        return jsonify({'error': 'Image not found'}), 404
    
    # Get boards containing this image
    boards = db.get_image_boards(image_id)
    image['boards'] = boards
    
    return jsonify(image)

@app.route('/api/images/<int:image_id>/file', methods=['GET'])
def serve_image(image_id):
    """Serve actual image file"""
    image = db.get_image(image_id)

    if not image:
        return jsonify({'error': 'Image not found'}), 404

    filepath = image['filepath']

    # Security: Validate filepath is within PHOTOS_DIR
    is_safe, resolved_path = is_safe_path(filepath, PHOTOS_DIR)

    if not is_safe:
        print(f"Security: Path traversal attempt blocked: {filepath}")
        return jsonify({'error': 'Invalid file path'}), 403

    if not os.path.exists(resolved_path):
        return jsonify({'error': 'File not found on disk'}), 404

    # Additional check: ensure it's actually a file, not a directory
    if not os.path.isfile(resolved_path):
        return jsonify({'error': 'Invalid file'}), 403

    return send_file(resolved_path, mimetype=mimetypes.guess_type(resolved_path)[0])

@app.route('/api/images/<int:image_id>/thumbnail', methods=['GET'])
def serve_thumbnail(image_id):
    """Serve thumbnail (resized image for grid) with caching"""
    size = request.args.get('size', 300, type=int)
    size = min(size, 1000)  # Prevent abuse

    image = db.get_image(image_id)

    if not image:
        return jsonify({'error': 'Image not found'}), 404

    filepath = image['filepath']

    # Security: Validate filepath is within PHOTOS_DIR
    is_safe, resolved_path = is_safe_path(filepath, PHOTOS_DIR)

    if not is_safe:
        print(f"Security: Path traversal attempt blocked: {filepath}")
        return jsonify({'error': 'Invalid file path'}), 403

    if not os.path.exists(resolved_path):
        return jsonify({'error': 'File not found on disk'}), 404

    if not os.path.isfile(resolved_path):
        return jsonify({'error': 'Invalid file'}), 403

    # Thumbnail caching
    thumbnail_cache_dir = os.path.join(DATA_DIR, 'thumbnails')
    os.makedirs(thumbnail_cache_dir, exist_ok=True)

    # Check if this is a video
    is_video = image.get('media_type') == 'video'

    # Generate cache key from image ID, size, and modification time
    try:
        mtime = int(os.path.getmtime(resolved_path))
        cache_filename = f"{image_id}_{size}_{mtime}.jpg"
        cache_path = os.path.join(thumbnail_cache_dir, cache_filename)

        # Check if cached thumbnail exists
        if os.path.exists(cache_path):
            return send_file(cache_path, mimetype='image/jpeg')

        # Generate and cache thumbnail
        if is_video:
            # Try to extract frame from video
            img = extract_video_frame(resolved_path, cache_path, time_sec=1.0)

            if not img:
                # Fallback to placeholder if opencv not available or extraction failed
                img = create_video_placeholder(size)
        else:
            # Regular image processing
            img = Image.open(resolved_path)

        # Resize thumbnail
        img.thumbnail((size, size), Image.Resampling.LANCZOS)

        # Higher quality for better visual appearance (92 is a good balance)
        img.save(cache_path, 'JPEG', quality=92, optimize=True)

        # Clean up old thumbnails for this image
        for old_file in os.listdir(thumbnail_cache_dir):
            if old_file.startswith(f"{image_id}_") and old_file != cache_filename:
                try:
                    os.remove(os.path.join(thumbnail_cache_dir, old_file))
                except:
                    pass

        return send_file(cache_path, mimetype='image/jpeg')
    except Exception as e:
        print(f"Error generating thumbnail: {e}")

        # For videos, try to return placeholder
        if is_video:
            try:
                img = create_video_placeholder(size)
                img.save(cache_path, 'JPEG', quality=92)
                return send_file(cache_path, mimetype='image/jpeg')
            except:
                pass

        # Fallback to original file
        return send_file(resolved_path, mimetype=mimetypes.guess_type(resolved_path)[0])

@app.route('/api/images/<int:image_id>/favorite', methods=['POST'])
def toggle_favorite(image_id):
    """Toggle favorite status"""
    new_status = db.toggle_favorite(image_id)
    
    return jsonify({
        'success': True,
        'image_id': image_id,
        'is_favorite': new_status
    })

@app.route('/api/images/<int:image_id>/rename', methods=['POST'])
def rename_image(image_id):
    """Rename image file"""
    data = request.json
    new_filename = data.get('new_filename')
    
    if not new_filename:
        return jsonify({'error': 'new_filename is required'}), 400
    
    # Get current image
    image = db.get_image(image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404
    
    old_path = image['filepath']
    
    if not os.path.exists(old_path):
        return jsonify({'error': 'File not found on disk'}), 404
    
    # Sanitize filename
    new_filename = secure_filename(new_filename)
    
    # Keep same directory
    directory = os.path.dirname(old_path)
    new_path = os.path.join(directory, new_filename)
    
    # Check if target exists
    if os.path.exists(new_path):
        return jsonify({'error': 'File with that name already exists'}), 409
    
    try:
        # Rename file on disk
        os.rename(old_path, new_path)
        
        # Update database
        db.rename_image(image_id, new_path, new_filename)
        
        return jsonify({
            'success': True,
            'image_id': image_id,
            'old_filename': image['filename'],
            'new_filename': new_filename,
            'new_filepath': new_path
        })
    except Exception as e:
        return jsonify({'error': f'Failed to rename: {str(e)}'}), 500

@app.route('/api/images/<int:image_id>', methods=['PATCH'])
def update_image(image_id):
    """Update image description and tags"""
    data = request.json

    # Get current image
    image = db.get_image(image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404

    description = data.get('description', '')
    tags = data.get('tags', [])

    try:
        # Use the database method to update image analysis
        # Clean up tags - remove empty strings
        clean_tags = [tag.strip() for tag in tags if tag and tag.strip()]

        # Update using the proper database method
        db.update_image_analysis(image_id, description, clean_tags)

        # Get updated image
        updated_image = db.get_image(image_id)

        return jsonify({
            'success': True,
            'image': updated_image
        })
    except Exception as e:
        return jsonify({'error': f'Failed to update: {str(e)}'}), 500

@app.route('/api/images/<int:image_id>/analyze', methods=['POST'])
def analyze_image(image_id):
    """Analyze single image/video with AI and optionally auto-rename"""
    temp_image_path = None
    try:
        image = db.get_image(image_id)

        if not image:
            return jsonify({'error': 'Image not found'}), 404

        filepath = image['filepath']
        media_type = image.get('media_type', 'image')

        # Security: Validate filepath is within PHOTOS_DIR
        is_safe, resolved_path = is_safe_path(filepath, PHOTOS_DIR)

        if not is_safe:
            print(f"Security: Path traversal attempt blocked in AI analyze: {filepath}")
            return jsonify({'error': 'Invalid file path'}), 403

        if not os.path.exists(resolved_path):
            return jsonify({'error': 'File not found on disk'}), 404

        # Check AI connection
        connected, message = ai.check_connection()
        if not connected:
            return jsonify({'error': f'AI not available: {message}'}), 503

        # Get style and custom prompt from request
        data = request.get_json() or {}
        style = data.get('style', 'classic')
        custom_prompt = data.get('custom_prompt', None)

        # For videos, extract frame first
        analysis_path = resolved_path
        if media_type == 'video':
            print(f"[ANALYZE] Extracting frame from video {image_id} for AI analysis...")

            # Get frame as PIL Image
            frame_img = get_image_for_analysis(resolved_path, media_type='video')

            if not frame_img:
                return jsonify({'error': 'Could not extract frame from video for analysis'}), 500

            # Save frame to temporary file
            temp_dir = os.path.join(DATA_DIR, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, f"video_frame_{image_id}.jpg")
            frame_img.save(temp_image_path, 'JPEG', quality=95)
            analysis_path = temp_image_path

            print(f"[ANALYZE] Video frame extracted to {temp_image_path}")

        # Analyze image with specified style
        print(f"[ANALYZE] Analyzing image {image_id} with style '{style}'...")
        result = ai.analyze_image(analysis_path, style=style, custom_prompt=custom_prompt)

        if result:
            print(f"[ANALYZE] AI analysis complete for image {image_id}")
            print(f"[ANALYZE] Description: {result['description'][:100]}...")
            print(f"[ANALYZE] Tags: {result['tags']}")

            # Update database with analysis
            print(f"[ANALYZE] Updating database for image {image_id}...")
            db.update_image_analysis(
                image_id,
                result['description'],
                result['tags']
            )
            print(f"[ANALYZE] ✅ Database updated successfully for image {image_id}")
            
            # Auto-rename if AI suggested a filename
            new_filename = None
            renamed = False
            
            if result.get('suggested_filename'):
                suggested = result['suggested_filename'].strip()
                
                if suggested and len(suggested) > 0:
                    # Sanitize filename
                    suggested = secure_filename(suggested)
                    
                    # Get original extension
                    old_ext = Path(filepath).suffix
                    
                    # Build new filename
                    new_filename = f"{suggested}{old_ext}"
                    
                    # Get directory
                    directory = os.path.dirname(filepath)
                    new_filepath = os.path.join(directory, new_filename)
                    
                    # Check if different from current name
                    if new_filepath != filepath:
                        # Check if target exists
                        if not os.path.exists(new_filepath):
                            try:
                                # Rename file on disk
                                os.rename(filepath, new_filepath)
                                
                                # Update database
                                db.rename_image(image_id, new_filepath, new_filename)
                                
                                renamed = True
                                print(f"Auto-renamed: {image['filename']} → {new_filename}")
                            except Exception as e:
                                print(f"Auto-rename failed: {e}")
                                renamed = False
                        else:
                            # File exists, add counter
                            counter = 1
                            base_name = suggested
                            while os.path.exists(new_filepath) and counter < 100:
                                new_filename = f"{base_name}_{counter}{old_ext}"
                                new_filepath = os.path.join(directory, new_filename)
                                counter += 1
                            
                            if not os.path.exists(new_filepath):
                                try:
                                    os.rename(filepath, new_filepath)
                                    db.rename_image(image_id, new_filepath, new_filename)
                                    renamed = True
                                    print(f"Auto-renamed: {image['filename']} → {new_filename}")
                                except Exception as e:
                                    print(f"Auto-rename failed: {e}")
            
            return jsonify({
                'success': True,
                'image_id': image_id,
                'description': result['description'],
                'tags': result['tags'],
                'renamed': renamed,
                'new_filename': new_filename if renamed else image['filename'],
                'suggested_filename': result.get('suggested_filename', '')
            })
        else:
            return jsonify({'error': 'Analysis failed - AI returned no result'}), 500
            
    except Exception as e:
        print(f"Error analyzing image {image_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500
    finally:
        # Clean up temporary video frame if created
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                print(f"[ANALYZE] Cleaned up temporary frame file: {temp_image_path}")
            except Exception as e:
                print(f"[ANALYZE] Warning: Could not delete temp file {temp_image_path}: {e}")

@app.route('/api/images/<int:image_id>/similar', methods=['GET'])
def get_similar_images(image_id):
    """Get similar images based on shared tags"""
    try:
        limit = int(request.args.get('limit', 6))
        similar = db.get_similar_images(image_id, limit)

        return jsonify({
            'image_id': image_id,
            'similar': similar,
            'count': len(similar)
        })

    except Exception as e:
        print(f"Error finding similar images for {image_id}: {str(e)}")
        return jsonify({'error': f'Failed to find similar images: {str(e)}'}), 500

@app.route('/api/images/search', methods=['GET'])
def search_images():
    """Search images by query"""
    query = request.args.get('q', '').strip()

    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400

    results = db.search_images(query)

    return jsonify({
        'query': query,
        'results': results,
        'count': len(results)
    })

# ============ TAG ENDPOINTS ============

@app.route('/api/tags', methods=['GET'])
def get_tags():
    """Get all tags with usage statistics"""
    try:
        tags = db.get_all_tags()
        return jsonify({
            'tags': tags,
            'count': len(tags)
        })
    except Exception as e:
        print(f"Error getting tags: {str(e)}")
        return jsonify({'error': f'Failed to get tags: {str(e)}'}), 500

@app.route('/api/tags/suggestions', methods=['GET'])
def get_tag_suggestions():
    """Get tag suggestions for autocomplete"""
    try:
        prefix = request.args.get('prefix', '')
        limit = int(request.args.get('limit', 10))

        suggestions = db.get_tag_suggestions(prefix, limit)
        return jsonify({
            'suggestions': suggestions,
            'count': len(suggestions)
        })
    except Exception as e:
        print(f"Error getting tag suggestions: {str(e)}")
        return jsonify({'error': f'Failed to get tag suggestions: {str(e)}'}), 500

@app.route('/api/tags/<tag>/related', methods=['GET'])
def get_related_tags(tag):
    """Get tags that frequently appear with the given tag"""
    try:
        limit = int(request.args.get('limit', 10))
        related = db.get_related_tags(tag, limit)
        return jsonify({
            'tag': tag,
            'related': related,
            'count': len(related)
        })
    except Exception as e:
        print(f"Error getting related tags: {str(e)}")
        return jsonify({'error': f'Failed to get related tags: {str(e)}'}), 500

# ============ OTHER ENDPOINTS ============

@app.route('/api/scan', methods=['POST'])
def scan_directory():
    """Scan photos directory for new images and videos"""
    if not os.path.exists(PHOTOS_DIR):
        return jsonify({'error': f'Photos directory not found: {PHOTOS_DIR}'}), 404

    found_media = []
    new_media = []
    skipped = 0

    # Walk through directory
    for root, dirs, files in os.walk(PHOTOS_DIR):
        for filename in files:
            ext = Path(filename).suffix.lower()

            if ext in ALL_MEDIA_FORMATS:
                filepath = os.path.join(root, filename)
                found_media.append(filepath)

                try:
                    file_size = os.path.getsize(filepath)
                    width = None
                    height = None
                    media_type = 'video' if ext in VIDEO_FORMATS else 'image'

                    # Get dimensions for images only
                    if media_type == 'image':
                        img = Image.open(filepath)
                        width, height = img.size
                        img.close()

                    # Try to add to database
                    image_id = db.add_image(
                        filepath=filepath,
                        filename=filename,
                        width=width,
                        height=height,
                        file_size=file_size,
                        media_type=media_type
                    )

                    if image_id:
                        new_media.append({
                            'id': image_id,
                            'filename': filename,
                            'filepath': filepath,
                            'media_type': media_type
                        })
                    else:
                        skipped += 1

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    skipped += 1

    return jsonify({
        'success': True,
        'found': len(found_media),
        'new': len(new_media),
        'skipped': skipped,
        'images': new_media
    })

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Upload image or video file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALL_MEDIA_FORMATS:
        return jsonify({'error': f'Unsupported format: {ext}'}), 400

    try:
        # Sanitize filename
        filename = secure_filename(file.filename)

        # Ensure photos directory exists
        os.makedirs(PHOTOS_DIR, exist_ok=True)

        # Save file
        filepath = os.path.join(PHOTOS_DIR, filename)

        # Handle duplicates
        counter = 1
        base_name = Path(filename).stem
        while os.path.exists(filepath):
            filename = f"{base_name}_{counter}{ext}"
            filepath = os.path.join(PHOTOS_DIR, filename)
            counter += 1

        file.save(filepath)

        # Get file info
        file_size = os.path.getsize(filepath)
        width = None
        height = None
        media_type = 'video' if ext in VIDEO_FORMATS else 'image'

        # Get dimensions for images only
        if media_type == 'image':
            img = Image.open(filepath)
            width, height = img.size
            img.close()

        # Add to database
        image_id = db.add_image(
            filepath=filepath,
            filename=filename,
            width=width,
            height=height,
            file_size=file_size,
            media_type=media_type
        )

        return jsonify({
            'success': True,
            'image_id': image_id,
            'filename': filename,
            'filepath': filepath,
            'media_type': media_type
        })

    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/analyze-batch', methods=['POST'])
def batch_analyze():
    """Analyze all unanalyzed images with auto-rename"""
    limit = request.args.get('limit', 10, type=int)
    
    # Check AI connection
    connected, message = ai.check_connection()
    if not connected:
        return jsonify({'error': f'AI not available: {message}'}), 503
    
    # Get unanalyzed images
    images = db.get_unanalyzed_images(limit=limit)
    
    if not images:
        return jsonify({
            'success': True,
            'message': 'No unanalyzed images',
            'analyzed': 0
        })
    
    analyzed_count = 0
    failed_count = 0
    renamed_count = 0
    
    for image in images:
        filepath = image['filepath']
        image_id = image['id']

        # Security: Validate filepath is within PHOTOS_DIR
        is_safe, resolved_path = is_safe_path(filepath, PHOTOS_DIR)

        if not is_safe or not os.path.exists(resolved_path):
            failed_count += 1
            continue

        result = ai.analyze_image(resolved_path)
        
        if result:
            # Update analysis
            db.update_image_analysis(
                image_id,
                result['description'],
                result['tags']
            )
            analyzed_count += 1
            
            # Auto-rename if AI suggested a filename
            if result.get('suggested_filename'):
                suggested = result['suggested_filename'].strip()
                
                if suggested and len(suggested) > 0:
                    # Sanitize filename
                    suggested = secure_filename(suggested)
                    
                    # Get original extension
                    old_ext = Path(filepath).suffix
                    
                    # Build new filename
                    new_filename = f"{suggested}{old_ext}"
                    
                    # Get directory
                    directory = os.path.dirname(filepath)
                    new_filepath = os.path.join(directory, new_filename)
                    
                    # Check if different from current name
                    if new_filepath != filepath:
                        # Check if target exists
                        if not os.path.exists(new_filepath):
                            try:
                                # Rename file on disk
                                os.rename(filepath, new_filepath)
                                
                                # Update database
                                db.rename_image(image_id, new_filepath, new_filename)
                                
                                renamed_count += 1
                                print(f"Batch auto-renamed: {image['filename']} → {new_filename}")
                            except Exception as e:
                                print(f"Batch auto-rename failed for {image['filename']}: {e}")
                        else:
                            # File exists, add counter
                            counter = 1
                            base_name = suggested
                            while os.path.exists(new_filepath) and counter < 100:
                                new_filename = f"{base_name}_{counter}{old_ext}"
                                new_filepath = os.path.join(directory, new_filename)
                                counter += 1
                            
                            if not os.path.exists(new_filepath):
                                try:
                                    os.rename(filepath, new_filepath)
                                    db.rename_image(image_id, new_filepath, new_filename)
                                    renamed_count += 1
                                    print(f"Batch auto-renamed: {image['filename']} → {new_filename}")
                                except Exception as e:
                                    print(f"Batch auto-rename failed for {image['filename']}: {e}")
        else:
            failed_count += 1
    
    return jsonify({
        'success': True,
        'total': len(images),
        'analyzed': analyzed_count,
        'renamed': renamed_count,
        'failed': failed_count
    })

# ============ BOARD API ============

@app.route('/api/boards', methods=['GET', 'POST'])
def boards():
    """Get all boards or create new board"""
    if request.method == 'GET':
        all_boards = db.get_all_boards()

        # Add image count for each board
        conn = db.get_connection()
        cursor = conn.cursor()
        for board in all_boards:
            cursor.execute(
                'SELECT COUNT(*) as count FROM board_images WHERE board_id = ?',
                (board['id'],)
            )
            result = cursor.fetchone()
            board['image_count'] = result['count'] if result else 0
        conn.close()

        # Organize into hierarchy
        top_level = []
        boards_map = {board['id']: board for board in all_boards}

        for board in all_boards:
            board['sub_boards'] = []

        for board in all_boards:
            if board['parent_id'] is None:
                top_level.append(board)
            else:
                parent = boards_map.get(board['parent_id'])
                if parent:
                    parent['sub_boards'].append(board)

        return jsonify({
            'boards': top_level,
            'total': len(all_boards)
        })

    elif request.method == 'POST':
        data = request.json
        name = data.get('name')
        description = data.get('description')
        parent_id = data.get('parent_id')

        if not name:
            return jsonify({'error': 'Board name is required'}), 400

        board_id = db.create_board(name, description, parent_id)

        return jsonify({
            'success': True,
            'board_id': board_id,
            'name': name
        }), 201

@app.route('/api/boards/<int:board_id>', methods=['GET', 'PUT', 'DELETE'])
def board_detail(board_id):
    """Get, update, or delete board"""
    if request.method == 'GET':
        board = db.get_board(board_id)
        
        if not board:
            return jsonify({'error': 'Board not found'}), 404
        
        # Get sub-boards
        board['sub_boards'] = db.get_sub_boards(board_id)
        
        # Get images in board
        board['images'] = db.get_board_images(board_id)
        
        return jsonify(board)
    
    elif request.method == 'PUT':
        data = request.json
        name = data.get('name')
        description = data.get('description')
        
        db.update_board(board_id, name, description)
        
        return jsonify({
            'success': True,
            'board_id': board_id
        })
    
    elif request.method == 'DELETE':
        data = request.json or {}
        delete_sub_boards = data.get('delete_sub_boards', False)

        db.delete_board(board_id, delete_sub_boards=delete_sub_boards)

        return jsonify({
            'success': True,
            'board_id': board_id,
            'deleted_sub_boards': delete_sub_boards
        })

@app.route('/api/boards/<int:board_id>/merge', methods=['POST'])
def merge_board(board_id):
    """Merge this board into another board"""
    data = request.json
    target_board_id = data.get('target_board_id')
    delete_source = data.get('delete_source', True)

    if not target_board_id:
        return jsonify({'error': 'target_board_id is required'}), 400

    if board_id == target_board_id:
        return jsonify({'error': 'Cannot merge board into itself'}), 400

    try:
        moved_count = db.merge_boards(board_id, target_board_id, delete_source)

        return jsonify({
            'success': True,
            'source_board_id': board_id,
            'target_board_id': target_board_id,
            'images_moved': moved_count,
            'source_deleted': delete_source
        })
    except Exception as e:
        print(f"Error merging boards: {e}")
        return jsonify({'error': f'Failed to merge boards: {str(e)}'}), 500

@app.route('/api/boards/<int:board_id>/images', methods=['POST', 'DELETE'])
def board_images(board_id):
    """Add or remove image from board"""
    data = request.json
    image_id = data.get('image_id')
    
    if not image_id:
        return jsonify({'error': 'image_id is required'}), 400
    
    if request.method == 'POST':
        db.add_image_to_board(board_id, image_id)
        
        return jsonify({
            'success': True,
            'board_id': board_id,
            'image_id': image_id,
            'action': 'added'
        })
    
    elif request.method == 'DELETE':
        db.remove_image_from_board(board_id, image_id)
        
        return jsonify({
            'success': True,
            'board_id': board_id,
            'image_id': image_id,
            'action': 'removed'
        })

# ============ PRIVACY OPERATIONS ============

@app.route('/api/images/<int:image_id>/privacy/analyze', methods=['POST'])
def analyze_image_privacy(image_id):
    """
    Analyze image for privacy concerns (faces, license plates, NSFW)
    """
    image = db.get_image(image_id)

    if not image:
        return jsonify({'error': 'Image not found'}), 404

    # Get image path
    image_path = image['filepath']

    # Security: Validate filepath is within PHOTOS_DIR
    is_safe, resolved_path = is_safe_path(image_path, PHOTOS_DIR)

    if not is_safe:
        print(f"Security: Path traversal attempt blocked in privacy analyze: {image_path}")
        return jsonify({'error': 'Invalid file path'}), 403

    if not os.path.exists(resolved_path):
        return jsonify({'error': 'Image file not found'}), 404

    try:
        # Perform privacy analysis
        privacy_service = get_privacy_service()
        result = privacy_service.analyze_image_privacy(resolved_path)

        # Update database
        db.update_privacy_analysis(
            image_id=image_id,
            has_faces=result['has_faces'],
            has_plates=result['has_plates'],
            is_nsfw=result['is_nsfw'],
            privacy_zones=result['privacy_zones']
        )

        return jsonify({
            'success': True,
            'image_id': image_id,
            'has_faces': result['has_faces'],
            'has_plates': result['has_plates'],
            'is_nsfw': result['is_nsfw'],
            'privacy_zones': result['privacy_zones'],
            'zones_count': len(result['privacy_zones'])
        })

    except Exception as e:
        print(f"Error analyzing privacy: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/images/<int:image_id>/privacy', methods=['GET'])
def get_image_privacy(image_id):
    """
    Get privacy analysis data for an image
    """
    privacy_data = db.get_privacy_data(image_id)

    if privacy_data is None:
        return jsonify({
            'analyzed': False,
            'message': 'Image not analyzed for privacy yet'
        })

    return jsonify({
        'analyzed': True,
        'has_faces': privacy_data['has_faces'],
        'has_plates': privacy_data['has_plates'],
        'is_nsfw': privacy_data['is_nsfw'],
        'privacy_zones': privacy_data['privacy_zones'],
        'analyzed_at': privacy_data['privacy_analyzed_at']
    })

@app.route('/api/images/<int:image_id>/thumbnail/blur', methods=['GET'])
def get_blurred_thumbnail(image_id):
    """
    Get thumbnail with privacy zones blurred
    """
    # Get size parameter
    size = request.args.get('size', 500, type=int)
    blur_strength = request.args.get('blur', 30, type=int)

    image = db.get_image(image_id)

    if not image:
        return jsonify({'error': 'Image not found'}), 404

    image_path = image['filepath']

    # Security: Validate filepath is within PHOTOS_DIR
    is_safe, resolved_path = is_safe_path(image_path, PHOTOS_DIR)

    if not is_safe:
        print(f"Security: Path traversal attempt blocked in blurred thumbnail: {image_path}")
        return jsonify({'error': 'Invalid file path'}), 403

    if not os.path.exists(resolved_path):
        return jsonify({'error': 'Image file not found'}), 404

    # Get privacy zones
    privacy_data = db.get_privacy_data(image_id)
    zones = privacy_data['privacy_zones'] if privacy_data else []

    try:
        # Generate blurred thumbnail
        privacy_service = get_privacy_service()
        blurred_img = privacy_service.generate_privacy_thumbnail(
            resolved_path,
            zones,
            size=size,
            blur_strength=blur_strength
        )

        # Convert to bytes
        img_io = io.BytesIO()
        blurred_img.save(img_io, 'JPEG', quality=90)
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        print(f"Error generating blurred thumbnail: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/privacy/batch-analyze', methods=['POST'])
def batch_analyze_privacy():
    """
    Batch analyze images for privacy
    """
    data = request.get_json()
    image_ids = data.get('image_ids', [])

    if not image_ids:
        return jsonify({'error': 'No image IDs provided'}), 400

    results = {
        'success': [],
        'failed': [],
        'total': len(image_ids)
    }

    privacy_service = get_privacy_service()

    for image_id in image_ids:
        try:
            image = db.get_image(image_id)

            if not image:
                results['failed'].append({
                    'id': image_id,
                    'error': 'Image not found'
                })
                continue

            # Security: Validate filepath is within PHOTOS_DIR
            is_safe, resolved_path = is_safe_path(image['filepath'], PHOTOS_DIR)

            if not is_safe or not os.path.exists(resolved_path):
                results['failed'].append({
                    'id': image_id,
                    'error': 'Invalid or missing file'
                })
                continue

            # Analyze
            result = privacy_service.analyze_image_privacy(resolved_path)

            # Update DB
            db.update_privacy_analysis(
                image_id=image_id,
                has_faces=result['has_faces'],
                has_plates=result['has_plates'],
                is_nsfw=result['is_nsfw'],
                privacy_zones=result['privacy_zones']
            )

            results['success'].append({
                'id': image_id,
                'has_faces': result['has_faces'],
                'has_plates': result['has_plates'],
                'zones_count': len(result['privacy_zones'])
            })

        except Exception as e:
            results['failed'].append({
                'id': image_id,
                'error': str(e)
            })

    return jsonify(results)

@app.route('/api/privacy/stats', methods=['GET'])
def get_privacy_stats():
    """
    Get privacy statistics
    """
    # Get counts
    images_with_faces = db.get_images_with_faces(limit=10000)
    nsfw_images = db.get_nsfw_images(limit=10000)
    unanalyzed = db.get_unanalyzed_privacy_images(limit=10000)

    return jsonify({
        'images_with_faces': len(images_with_faces),
        'nsfw_images': len(nsfw_images),
        'unanalyzed_images': len(unanalyzed)
    })

# ============ EXPORT/IMPORT DATA ============

@app.route('/api/export', methods=['GET'])
def export_data():
    """
    Export gallery data in various formats

    Query parameters:
        format: json|markdown|csv (default: json)
        include_images: true|false (default: true)
        include_boards: true|false (default: true)
    """
    format_type = request.args.get('format', 'json').lower()
    include_images = request.args.get('include_images', 'true').lower() == 'true'
    include_boards = request.args.get('include_boards', 'true').lower() == 'true'

    try:
        if format_type == 'json':
            data = db.export_data_json(include_images=include_images, include_boards=include_boards)

            # Create response with proper headers for file download
            from flask import Response
            response = Response(data, mimetype='application/json')
            response.headers['Content-Disposition'] = 'attachment; filename=gallery_export.json'
            return response

        elif format_type == 'markdown':
            data = db.export_data_markdown(include_images=include_images, include_boards=include_boards)

            from flask import Response
            response = Response(data, mimetype='text/markdown')
            response.headers['Content-Disposition'] = 'attachment; filename=gallery_export.md'
            return response

        elif format_type == 'csv':
            csv_data = db.export_data_csv()

            # Create a ZIP file with multiple CSV files
            import zipfile
            from io import BytesIO

            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                if 'images_csv' in csv_data:
                    zip_file.writestr('images.csv', csv_data['images_csv'])
                if 'boards_csv' in csv_data:
                    zip_file.writestr('boards.csv', csv_data['boards_csv'])
                if 'board_images_csv' in csv_data:
                    zip_file.writestr('board_images.csv', csv_data['board_images_csv'])

            zip_buffer.seek(0)

            from flask import Response
            response = Response(zip_buffer.getvalue(), mimetype='application/zip')
            response.headers['Content-Disposition'] = 'attachment; filename=gallery_export.zip'
            return response

        else:
            return jsonify({'error': f'Unsupported format: {format_type}'}), 400

    except Exception as e:
        print(f"Export error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/import', methods=['POST'])
def import_data():
    """
    Import gallery data from JSON export

    Request body (JSON):
        {
            "data": "<json_export_data>",
            "import_boards": true,
            "import_board_assignments": true,
            "update_existing": false
        }
    """
    try:
        request_data = request.get_json()

        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'Missing data field in request'}), 400

        json_data = request_data['data']
        import_boards = request_data.get('import_boards', True)
        import_board_assignments = request_data.get('import_board_assignments', True)
        update_existing = request_data.get('update_existing', False)

        # Perform import
        result = db.import_data_json(
            json_data=json_data,
            import_boards=import_boards,
            import_board_assignments=import_board_assignments,
            update_existing=update_existing
        )

        if result.get('success'):
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        print(f"Import error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

# ============ RESEARCH & EDUCATION ENDPOINTS ============

@app.route('/api/annotations', methods=['POST'])
def add_annotation():
    """Add annotation to an image"""
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        class_name = data.get('class_name')
        x = data.get('x')
        y = data.get('y')
        width = data.get('width')
        height = data.get('height')
        class_id = data.get('class_id')
        confidence = data.get('confidence', 1.0)
        notes = data.get('notes')

        if not all([image_id, class_name, x is not None, y is not None, width, height]):
            return jsonify({'error': 'Missing required fields'}), 400

        annotation_id = db.add_annotation(
            image_id=image_id,
            class_name=class_name,
            x=x,
            y=y,
            width=width,
            height=height,
            class_id=class_id,
            confidence=confidence,
            notes=notes
        )

        return jsonify({
            'success': True,
            'annotation_id': annotation_id
        })

    except Exception as e:
        print(f"Error adding annotation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/images/<int:image_id>/annotations', methods=['GET'])
def get_image_annotations(image_id):
    """Get all annotations for an image"""
    try:
        annotations = db.get_annotations(image_id)
        return jsonify({
            'success': True,
            'annotations': annotations,
            'count': len(annotations)
        })
    except Exception as e:
        print(f"Error getting annotations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/annotations/<int:annotation_id>', methods=['PUT'])
def update_annotation(annotation_id):
    """Update an annotation"""
    try:
        data = request.get_json()
        db.update_annotation(
            annotation_id=annotation_id,
            class_name=data.get('class_name'),
            x=data.get('x'),
            y=data.get('y'),
            width=data.get('width'),
            height=data.get('height'),
            class_id=data.get('class_id'),
            confidence=data.get('confidence'),
            notes=data.get('notes')
        )

        return jsonify({'success': True})

    except Exception as e:
        print(f"Error updating annotation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/annotations/<int:annotation_id>', methods=['DELETE'])
def delete_annotation(annotation_id):
    """Delete an annotation"""
    try:
        db.delete_annotation(annotation_id)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting annotation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset/classes', methods=['GET'])
def get_dataset_classes():
    """Get all dataset classes"""
    try:
        classes = db.get_dataset_classes()
        return jsonify({
            'success': True,
            'classes': classes
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset/classes', methods=['POST'])
def add_dataset_class():
    """Add a new dataset class"""
    try:
        data = request.get_json()
        name = data.get('name')
        color = data.get('color', '#FF5722')
        description = data.get('description')

        if not name:
            return jsonify({'error': 'Class name is required'}), 400

        class_id = db.add_dataset_class(name, color, description)

        return jsonify({
            'success': True,
            'class_id': class_id
        })

    except Exception as e:
        print(f"Error adding class: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset/stats', methods=['GET'])
def get_dataset_stats():
    """Get dataset statistics"""
    try:
        stats = db.get_dataset_statistics()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/citation', methods=['POST'])
def generate_citation():
    """Generate citation for an image"""
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        format_type = data.get('format', 'apa').lower()

        if not image_id:
            return jsonify({'error': 'image_id is required'}), 400

        image = db.get_image(image_id)
        if not image:
            return jsonify({'error': 'Image not found'}), 404

        citation = research.generate_citation(image, format_type)

        return jsonify({
            'success': True,
            'citation': citation,
            'format': format_type
        })

    except Exception as e:
        print(f"Error generating citation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/citation/batch', methods=['POST'])
def generate_batch_citations():
    """Generate citations for multiple images"""
    try:
        data = request.get_json()
        image_ids = data.get('image_ids', [])
        format_type = data.get('format', 'apa').lower()

        if not image_ids:
            return jsonify({'error': 'image_ids is required'}), 400

        citations = research.generate_batch_citations(image_ids, format_type)

        return jsonify({
            'success': True,
            'citations': citations,
            'count': len(citations)
        })

    except Exception as e:
        print(f"Error generating batch citations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset/export', methods=['GET'])
def export_dataset():
    """Export dataset in various formats"""
    try:
        format_type = request.args.get('format', 'coco').lower()
        include_images = request.args.get('include_images', 'false').lower() == 'true'
        split_dataset = request.args.get('split', 'false').lower() == 'true'

        # Get image IDs from query params or use all annotated images
        image_ids_param = request.args.get('image_ids')
        if image_ids_param:
            image_ids = [int(id.strip()) for id in image_ids_param.split(',')]
        else:
            # Get all images with annotations
            all_annotations = db.get_all_annotations()
            image_ids = list(set(a['image_id'] for a in all_annotations))

        if format_type == 'coco':
            coco_data = research.export_dataset_coco(image_ids)
            return jsonify(coco_data)

        elif format_type == 'yolo':
            yolo_data = research.export_dataset_yolo(image_ids)
            # Return as JSON with file contents
            return jsonify({
                'success': True,
                'files': yolo_data['files'],
                'images': yolo_data['images'],
                'classes': yolo_data['classes']
            })

        elif format_type == 'csv':
            csv_data = research.export_dataset_csv(image_ids)
            return csv_data, 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': f'attachment; filename="dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
            }

        elif format_type == 'zip':
            # Export as ZIP file
            export_format = request.args.get('export_format', 'coco').lower()
            zip_data = research.export_dataset_zip(export_format, image_ids, include_images, split_dataset)

            return zip_data, 200, {
                'Content-Type': 'application/zip',
                'Content-Disposition': f'attachment; filename="dataset_{export_format}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip"'
            }

        else:
            return jsonify({'error': f'Unsupported format: {format_type}'}), 400

    except Exception as e:
        print(f"Error exporting dataset: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/annotations/import-from-privacy', methods=['POST'])
def import_annotations_from_privacy():
    """Import privacy zones as annotations"""
    try:
        data = request.get_json()
        image_id = data.get('image_id')

        if not image_id:
            return jsonify({'error': 'image_id is required'}), 400

        # Get privacy data
        privacy_data = db.get_privacy_data(image_id)
        if not privacy_data or not privacy_data.get('privacy_zones'):
            return jsonify({'error': 'No privacy zones found'}), 404

        # Import privacy zones as annotations
        zones = privacy_data['privacy_zones']
        imported_count = 0

        for zone in zones:
            zone_type = zone.get('type')
            if zone_type in ['face', 'plate']:
                class_name = 'face' if zone_type == 'face' else 'license_plate'

                db.add_annotation(
                    image_id=image_id,
                    class_name=class_name,
                    x=zone['x'],
                    y=zone['y'],
                    width=zone['w'],
                    height=zone['h'],
                    confidence=0.8,
                    notes='Imported from privacy detection'
                )
                imported_count += 1

        return jsonify({
            'success': True,
            'imported': imported_count
        })

    except Exception as e:
        print(f"Error importing from privacy: {e}")
        return jsonify({'error': str(e)}), 500

# ============ WORKFLOW AUTOMATION ENDPOINTS ============

@app.route('/api/pipelines', methods=['GET'])
def get_pipelines():
    """Get all pipelines"""
    try:
        enabled_only = request.args.get('enabled_only', 'false').lower() == 'true'
        pipelines = db.get_all_pipelines(enabled_only=enabled_only)

        return jsonify({
            'success': True,
            'pipelines': pipelines,
            'count': len(pipelines)
        })
    except Exception as e:
        print(f"Error getting pipelines: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pipelines', methods=['POST'])
def create_pipeline():
    """Create a new pipeline"""
    try:
        data = request.get_json()
        name = data.get('name')
        description = data.get('description', '')
        trigger_type = data.get('trigger_type')
        trigger_config = data.get('trigger_config', {})
        actions = data.get('actions', [])
        enabled = data.get('enabled', True)

        if not name or not trigger_type or not actions:
            return jsonify({'error': 'name, trigger_type, and actions are required'}), 400

        pipeline_id = db.create_pipeline(
            name=name,
            description=description,
            trigger_type=trigger_type,
            trigger_config=trigger_config,
            actions=actions,
            enabled=enabled
        )

        return jsonify({
            'success': True,
            'pipeline_id': pipeline_id
        })

    except Exception as e:
        print(f"Error creating pipeline: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/pipelines/<int:pipeline_id>', methods=['GET'])
def get_pipeline(pipeline_id):
    """Get pipeline by ID"""
    try:
        pipeline_data = db.get_pipeline(pipeline_id)

        if not pipeline_data:
            return jsonify({'error': 'Pipeline not found'}), 404

        return jsonify({
            'success': True,
            'pipeline': pipeline_data
        })
    except Exception as e:
        print(f"Error getting pipeline: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pipelines/<int:pipeline_id>', methods=['PUT'])
def update_pipeline(pipeline_id):
    """Update a pipeline"""
    try:
        data = request.get_json()

        db.update_pipeline(
            pipeline_id=pipeline_id,
            name=data.get('name'),
            description=data.get('description'),
            trigger_type=data.get('trigger_type'),
            trigger_config=data.get('trigger_config'),
            actions=data.get('actions'),
            enabled=data.get('enabled')
        )

        return jsonify({'success': True})

    except Exception as e:
        print(f"Error updating pipeline: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pipelines/<int:pipeline_id>', methods=['DELETE'])
def delete_pipeline(pipeline_id):
    """Delete a pipeline"""
    try:
        db.delete_pipeline(pipeline_id)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting pipeline: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pipelines/<int:pipeline_id>/execute', methods=['POST'])
def execute_pipeline(pipeline_id):
    """Execute a pipeline"""
    try:
        data = request.get_json() or {}
        image_ids = data.get('image_ids', [])

        if not image_ids:
            return jsonify({'error': 'image_ids is required'}), 400

        result = pipeline['executor'].execute_pipeline(
            pipeline_id=pipeline_id,
            image_ids=image_ids,
            trigger_source='manual'
        )

        return jsonify(result)

    except Exception as e:
        print(f"Error executing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/pipelines/actions', methods=['GET'])
def get_available_actions():
    """Get list of available actions"""
    try:
        actions = pipeline['actions'].get_available_actions()

        return jsonify({
            'success': True,
            'actions': actions
        })
    except Exception as e:
        print(f"Error getting actions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pipelines/templates', methods=['GET'])
def get_pipeline_templates():
    """Get pipeline templates"""
    try:
        templates = pipeline['templates'].get_all_templates()

        return jsonify({
            'success': True,
            'templates': templates
        })
    except Exception as e:
        print(f"Error getting templates: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pipelines/templates/<template_id>', methods=['POST'])
def create_from_template(template_id):
    """Create pipeline from template"""
    try:
        pipeline_id = pipeline['templates'].create_from_template(template_id, db)

        if not pipeline_id:
            return jsonify({'error': 'Template not found'}), 404

        return jsonify({
            'success': True,
            'pipeline_id': pipeline_id
        })

    except Exception as e:
        print(f"Error creating from template: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pipelines/<int:pipeline_id>/executions', methods=['GET'])
def get_pipeline_executions(pipeline_id):
    """Get execution history for a pipeline"""
    try:
        limit = request.args.get('limit', 50, type=int)
        executions = db.get_pipeline_execution_history(pipeline_id, limit=limit)

        return jsonify({
            'success': True,
            'executions': executions,
            'count': len(executions)
        })
    except Exception as e:
        print(f"Error getting executions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/executions/recent', methods=['GET'])
def get_recent_executions():
    """Get recent pipeline executions"""
    try:
        limit = request.args.get('limit', 100, type=int)
        executions = db.get_recent_executions(limit=limit)

        return jsonify({
            'success': True,
            'executions': executions,
            'count': len(executions)
        })
    except Exception as e:
        print(f"Error getting recent executions: {e}")
        return jsonify({'error': str(e)}), 500

# ============ EMAIL SETTINGS ============

@app.route('/api/settings/email', methods=['GET'])
def get_email_settings():
    """Get email configuration"""
    try:
        config = email.smtp_config
        # Don't send password to client
        safe_config = {**config}
        if 'password' in safe_config:
            safe_config['password'] = '***' if safe_config['password'] else ''

        return jsonify({
            'success': True,
            'config': safe_config
        })
    except Exception as e:
        print(f"Error getting email settings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings/email', methods=['POST'])
def save_email_settings():
    """Save email configuration"""
    try:
        data = request.json

        # Only update password if it's not the placeholder
        if data.get('password') == '***':
            data['password'] = email.smtp_config.get('password', '')

        email.save_config(data)

        return jsonify({
            'success': True,
            'message': 'Email settings saved'
        })
    except Exception as e:
        print(f"Error saving email settings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings/email/test', methods=['POST'])
def test_email_connection():
    """Test email connection"""
    try:
        result = email.test_connection()
        return jsonify(result)
    except Exception as e:
        print(f"Error testing email: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/email/send', methods=['POST'])
def send_test_email():
    """Send a test email"""
    try:
        data = request.json
        to_email = data.get('to_email')
        subject = data.get('subject', 'Test Email from AI Gallery')
        body = data.get('body', 'This is a test email from AI Gallery.')

        if not to_email:
            return jsonify({'success': False, 'error': 'Recipient email required'}), 400

        result = email.send_email(to_email, subject, body)
        return jsonify(result)
    except Exception as e:
        print(f"Error sending email: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ============ SCHEDULER ============

@app.route('/api/scheduler/pipelines', methods=['GET'])
def get_scheduled_pipelines_list():
    """Get all scheduled pipelines"""
    try:
        if not scheduler:
            return jsonify({'success': False, 'error': 'Scheduler not available'}), 503

        jobs = scheduler.get_scheduled_pipelines()
        return jsonify({
            'success': True,
            'jobs': jobs,
            'count': len(jobs)
        })
    except Exception as e:
        print(f"Error getting scheduled pipelines: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/scheduler/pipelines/<int:pipeline_id>/schedule', methods=['POST'])
def schedule_pipeline_execution(pipeline_id):
    """Schedule a pipeline for automatic execution"""
    try:
        if not scheduler:
            return jsonify({'success': False, 'error': 'Scheduler not available'}), 503

        data = request.json
        schedule_config = data.get('schedule_config', {})

        result = scheduler.schedule_pipeline(pipeline_id, schedule_config)
        return jsonify(result)
    except Exception as e:
        print(f"Error scheduling pipeline: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/scheduler/pipelines/<int:pipeline_id>/unschedule', methods=['DELETE'])
def unschedule_pipeline_execution(pipeline_id):
    """Remove scheduled execution for a pipeline"""
    try:
        if not scheduler:
            return jsonify({'success': False, 'error': 'Scheduler not available'}), 503

        result = scheduler.unschedule_pipeline(pipeline_id)
        return jsonify(result)
    except Exception as e:
        print(f"Error unscheduling pipeline: {e}")
        return jsonify({'error': str(e)}), 500

# ============ WEBHOOKS ============

def verify_webhook_token(token: str) -> bool:
    """Verify webhook authentication token"""
    # Get webhook token from settings
    webhook_config = db.get_setting('webhook_config')
    if not webhook_config:
        return False

    config = json.loads(webhook_config)
    return config.get('enabled') and config.get('token') == token

@app.route('/api/webhooks/trigger/<int:pipeline_id>', methods=['POST'])
def webhook_trigger_pipeline(pipeline_id):
    """Trigger a pipeline via webhook"""
    try:
        # Check authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization'}), 401

        token = auth_header.replace('Bearer ', '')
        if not verify_webhook_token(token):
            return jsonify({'error': 'Invalid webhook token'}), 403

        # Get pipeline
        pipeline_obj = db.get_pipeline(pipeline_id)
        if not pipeline_obj:
            return jsonify({'error': 'Pipeline not found'}), 404

        if pipeline_obj['trigger_type'] != 'webhook' and pipeline_obj['trigger_type'] != 'manual':
            return jsonify({'error': 'Pipeline not configured for webhook trigger'}), 400

        # Get image filter from request
        data = request.json or {}
        image_ids = data.get('image_ids', [])

        # If no image IDs provided, use filter from request or pipeline config
        if not image_ids:
            image_filter = data.get('image_filter', {})
            if not image_filter:
                trigger_config = json.loads(pipeline_obj.get('trigger_config', '{}'))
                image_filter = trigger_config.get('image_filter', {})

            # Apply filter to get images
            if image_filter.get('type') == 'all':
                images = db.get_all_images(limit=10000)
            elif image_filter.get('type') == 'recent':
                days = image_filter.get('days', 7)
                # Get recent images - simplified
                images = db.get_all_images(limit=1000)
            elif image_filter.get('type') == 'board':
                board_id = image_filter.get('board_id')
                images = db.get_board_images(board_id) if board_id else []
            else:
                images = db.get_all_images(limit=100)

            image_ids = [img['id'] for img in images]

        if not image_ids:
            return jsonify({'error': 'No images to process'}), 400

        # Execute pipeline
        result = pipeline['executor'].execute_pipeline(
            pipeline_id=pipeline_id,
            image_ids=image_ids,
            trigger_source='webhook'
        )

        return jsonify(result)

    except Exception as e:
        print(f"Error in webhook trigger: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/webhooks/config', methods=['GET'])
def get_webhook_config():
    """Get webhook configuration"""
    try:
        config_str = db.get_setting('webhook_config')
        if config_str:
            config = json.loads(config_str)
            # Don't send token to client in full
            if config.get('token'):
                config['token_preview'] = config['token'][:8] + '...'
                del config['token']
        else:
            config = {
                'enabled': False,
                'token': None
            }

        return jsonify({
            'success': True,
            'config': config
        })
    except Exception as e:
        print(f"Error getting webhook config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/webhooks/config', methods=['POST'])
def save_webhook_config():
    """Save webhook configuration"""
    try:
        data = request.json
        enabled = data.get('enabled', False)
        token = data.get('token')

        # Generate token if enabled but no token provided
        if enabled and not token:
            import secrets
            token = secrets.token_urlsafe(32)

        config = {
            'enabled': enabled,
            'token': token
        }

        db.set_setting('webhook_config', json.dumps(config))

        return jsonify({
            'success': True,
            'token': token,
            'webhook_url': f"{request.host_url}api/webhooks/trigger/<pipeline_id>"
        })
    except Exception as e:
        print(f"Error saving webhook config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/webhooks/regenerate-token', methods=['POST'])
def regenerate_webhook_token():
    """Regenerate webhook token"""
    try:
        import secrets
        new_token = secrets.token_urlsafe(32)

        config_str = db.get_setting('webhook_config')
        config = json.loads(config_str) if config_str else {}

        config['token'] = new_token
        db.set_setting('webhook_config', json.dumps(config))

        return jsonify({
            'success': True,
            'token': new_token
        })
    except Exception as e:
        print(f"Error regenerating token: {e}")
        return jsonify({'error': str(e)}), 500

# ============ CLOUD SYNC & BACKUP ============

@app.route('/api/backup/create', methods=['POST'])
def create_backup():
    """Create a backup"""
    try:
        data = request.json or {}
        include_images = data.get('include_images', True)

        result = cloud_sync.create_backup(include_images=include_images)
        return jsonify(result)
    except Exception as e:
        print(f"Error creating backup: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/backup/list', methods=['GET'])
def list_backups():
    """List all backups"""
    try:
        backups = cloud_sync.get_backups_list()
        return jsonify({
            'success': True,
            'backups': backups,
            'count': len(backups)
        })
    except Exception as e:
        print(f"Error listing backups: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/backup/<backup_name>', methods=['DELETE'])
def delete_backup(backup_name):
    """Delete a backup"""
    try:
        result = cloud_sync.delete_backup(backup_name)
        return jsonify(result)
    except Exception as e:
        print(f"Error deleting backup: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/backup/<backup_name>/download', methods=['GET'])
def download_backup(backup_name):
    """Download a backup file"""
    try:
        backup_file = cloud_sync.backup_dir / f'{backup_name}.zip'
        if not backup_file.exists():
            return jsonify({'error': 'Backup not found'}), 404

        return send_file(
            backup_file,
            as_attachment=True,
            download_name=f'{backup_name}.zip'
        )
    except Exception as e:
        print(f"Error downloading backup: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cloud-sync/config', methods=['GET'])
def get_cloud_sync_config():
    """Get cloud sync configuration"""
    try:
        config = cloud_sync.get_sync_config()
        return jsonify({
            'success': True,
            'config': config
        })
    except Exception as e:
        print(f"Error getting cloud sync config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cloud-sync/config', methods=['POST'])
def save_cloud_sync_config():
    """Save cloud sync configuration"""
    try:
        data = request.json
        result = cloud_sync.save_sync_config(data)
        return jsonify(result)
    except Exception as e:
        print(f"Error saving cloud sync config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cloud-sync/remotes', methods=['GET'])
def list_rclone_remotes():
    """List rclone remotes"""
    try:
        result = cloud_sync.list_rclone_remotes()
        return jsonify(result)
    except Exception as e:
        print(f"Error listing remotes: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cloud-sync/sync', methods=['POST'])
def sync_to_cloud():
    """Sync backup to cloud"""
    try:
        data = request.json
        remote = data.get('remote')

        if not remote:
            return jsonify({'error': 'Remote name required'}), 400

        result = cloud_sync.sync_to_cloud(remote)
        return jsonify(result)
    except Exception as e:
        print(f"Error syncing to cloud: {e}")
        return jsonify({'error': str(e)}), 500

# ============ STATS & ANALYTICS ============

@app.route('/api/stats/overview', methods=['GET'])
def get_stats_overview():
    """Get overview statistics"""
    try:
        # Get total counts
        all_images = db.get_all_images(limit=100000)
        total_images = len(all_images)

        favorites = [img for img in all_images if img.get('is_favorite')]
        analyzed = [img for img in all_images if img.get('analyzed_at')]
        with_faces = [img for img in all_images if img.get('has_faces')]

        # Calculate total storage
        total_size = sum(img.get('file_size', 0) for img in all_images)

        # Get boards count
        boards = db.get_all_boards()

        # Get tags distribution
        all_tags = {}
        for img in all_images:
            tags = json.loads(img.get('tags', '[]'))
            for tag in tags:
                all_tags[tag] = all_tags.get(tag, 0) + 1

        top_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:10]

        return jsonify({
            'success': True,
            'stats': {
                'total_images': total_images,
                'favorites': len(favorites),
                'analyzed': len(analyzed),
                'with_faces': len(with_faces),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / 1024 / 1024, 2),
                'total_boards': len(boards),
                'total_tags': len(all_tags),
                'top_tags': [{'tag': tag, 'count': count} for tag, count in top_tags]
            }
        })
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats/timeline', methods=['GET'])
def get_timeline_stats():
    """Get timeline statistics (images per day/month)"""
    try:
        from datetime import datetime, timedelta
        from collections import defaultdict

        all_images = db.get_all_images(limit=100000)

        # Group by date
        by_day = defaultdict(int)
        by_month = defaultdict(int)
        by_year = defaultdict(int)

        for img in all_images:
            created_at = img.get('created_at')
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    by_day[dt.strftime('%Y-%m-%d')] += 1
                    by_month[dt.strftime('%Y-%m')] += 1
                    by_year[dt.strftime('%Y')] += 1
                except:
                    pass

        # Convert to sorted lists
        timeline_daily = [{'date': date, 'count': count} for date, count in sorted(by_day.items())]
        timeline_monthly = [{'month': month, 'count': count} for month, count in sorted(by_month.items())]
        timeline_yearly = [{'year': year, 'count': count} for year, count in sorted(by_year.items())]

        return jsonify({
            'success': True,
            'timeline': {
                'daily': timeline_daily[-90:],  # Last 90 days
                'monthly': timeline_monthly[-24:],  # Last 24 months
                'yearly': timeline_yearly
            }
        })
    except Exception as e:
        print(f"Error getting timeline stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats/storage', methods=['GET'])
def get_storage_stats():
    """Get storage statistics by type"""
    try:
        all_images = db.get_all_images(limit=100000)

        # Group by media type
        by_type = {}
        for img in all_images:
            media_type = img.get('media_type', 'image')
            if media_type not in by_type:
                by_type[media_type] = {'count': 0, 'size': 0}

            by_type[media_type]['count'] += 1
            by_type[media_type]['size'] += img.get('file_size', 0)

        # Convert to list
        storage_by_type = [
            {
                'type': media_type,
                'count': data['count'],
                'size_mb': round(data['size'] / 1024 / 1024, 2)
            }
            for media_type, data in by_type.items()
        ]

        return jsonify({
            'success': True,
            'storage': storage_by_type
        })
    except Exception as e:
        print(f"Error getting storage stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats/activity', methods=['GET'])
def get_activity_stats():
    """Get recent activity statistics"""
    try:
        from datetime import datetime, timedelta

        # Get images from last 7 days
        all_images = db.get_all_images(limit=100000)
        now = datetime.now()

        recent_images = []
        for img in all_images:
            created_at = img.get('created_at')
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    if (now - dt).days <= 7:
                        recent_images.append(img)
                except:
                    pass

        # Get recent pipeline executions
        recent_executions = db.get_recent_executions(limit=20)

        return jsonify({
            'success': True,
            'activity': {
                'images_last_7_days': len(recent_images),
                'recent_executions': recent_executions[:5]
            }
        })
    except Exception as e:
        print(f"Error getting activity stats: {e}")
        return jsonify({'error': str(e)}), 500

# ============ STATIC FILES ============

@app.route('/favicon.ico')
def favicon():
    """Serve favicon"""
    # Return a simple emoji as SVG favicon
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <text y="75" font-size="75">🖼️</text>
    </svg>'''
    return svg, 200, {'Content-Type': 'image/svg+xml'}

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# ============ ERROR HANDLERS ============

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============ MAIN ============

if __name__ == '__main__':
    # Ensure photos directory exists
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    print(f"""
╔══════════════════════════════════════╗
║       AI Gallery Starting...         ║
╚══════════════════════════════════════╝

📁 Photos Directory: {PHOTOS_DIR}
🤖 LM Studio URL: {LM_STUDIO_URL}
💾 Database: {DATABASE_PATH}

🌐 Open: http://localhost:5000

Press Ctrl+C to stop
    """)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )