"""
AI Gallery - Flask Application
Main web server with REST API endpoints
"""

from flask import Flask, render_template, jsonify, request, send_file, send_from_directory, after_this_request
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
from pdf_catalog import PDFCatalogGenerator
from export_utils import MetadataExporter, BoardExporter

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
        if not os.path.exists(filepath):
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

        # Get absolute path
        abs_filepath = os.path.abspath(filepath)

        # Launch application in background
        command = [app['command'], abs_filepath]

        print(f"[OPEN_WITH] Opening {abs_filepath} with {app['name']} ({app['command']})")

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
    abs_filepath = os.path.abspath(filepath)
    abs_photos_dir = os.path.abspath(PHOTOS_DIR)

    if not abs_filepath.startswith(abs_photos_dir):
        print(f"Security: Path traversal attempt blocked: {filepath}")
        return jsonify({'error': 'Invalid file path'}), 403

    if not os.path.exists(abs_filepath):
        return jsonify({'error': 'File not found on disk'}), 404

    # Additional check: ensure it's actually a file, not a directory
    if not os.path.isfile(abs_filepath):
        return jsonify({'error': 'Invalid file'}), 403

    return send_file(abs_filepath, mimetype=mimetypes.guess_type(abs_filepath)[0])

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
    abs_filepath = os.path.abspath(filepath)
    abs_photos_dir = os.path.abspath(PHOTOS_DIR)

    if not abs_filepath.startswith(abs_photos_dir):
        print(f"Security: Path traversal attempt blocked: {filepath}")
        return jsonify({'error': 'Invalid file path'}), 403

    if not os.path.exists(abs_filepath):
        return jsonify({'error': 'File not found on disk'}), 404

    if not os.path.isfile(abs_filepath):
        return jsonify({'error': 'Invalid file'}), 403

    # Thumbnail caching
    thumbnail_cache_dir = os.path.join(DATA_DIR, 'thumbnails')
    os.makedirs(thumbnail_cache_dir, exist_ok=True)

    # Check if this is a video
    is_video = image.get('media_type') == 'video'

    # Generate cache key from image ID, size, and modification time
    try:
        mtime = int(os.path.getmtime(abs_filepath))
        cache_filename = f"{image_id}_{size}_{mtime}.jpg"
        cache_path = os.path.join(thumbnail_cache_dir, cache_filename)

        # Check if cached thumbnail exists
        if os.path.exists(cache_path):
            return send_file(cache_path, mimetype='image/jpeg')

        # Generate and cache thumbnail
        if is_video:
            # Try to extract frame from video
            img = extract_video_frame(abs_filepath, cache_path, time_sec=1.0)

            if not img:
                # Fallback to placeholder if opencv not available or extraction failed
                img = create_video_placeholder(size)
        else:
            # Regular image processing
            img = Image.open(abs_filepath)

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
        return send_file(abs_filepath, mimetype=mimetypes.guess_type(abs_filepath)[0])

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

        if not os.path.exists(filepath):
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
        analysis_path = filepath
        if media_type == 'video':
            print(f"[ANALYZE] Extracting frame from video {image_id} for AI analysis...")

            # Get frame as PIL Image
            frame_img = get_image_for_analysis(filepath, media_type='video')

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
        
        if not os.path.exists(filepath):
            failed_count += 1
            continue
        
        result = ai.analyze_image(filepath)
        
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

# ============ EXPORT ENDPOINTS ============

@app.route('/api/export/images/csv', methods=['POST'])
def export_images_csv():
    """Export images metadata to CSV"""
    data = request.get_json() or {}
    image_ids = data.get('image_ids', [])

    if not image_ids:
        return jsonify({'error': 'No image IDs provided'}), 400

    # Get images
    images = []
    for image_id in image_ids:
        image = db.get_image_by_id(image_id)
        if image:
            images.append(image)

    if not images:
        return jsonify({'error': 'No valid images found'}), 404

    # Generate CSV
    csv_content = MetadataExporter.to_csv(images)

    # Return as downloadable file
    return csv_content, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': f'attachment; filename="image_metadata.csv"'
    }

@app.route('/api/export/images/json', methods=['POST'])
def export_images_json():
    """Export images metadata to JSON"""
    data = request.get_json() or {}
    image_ids = data.get('image_ids', [])
    include_summary = data.get('include_summary', True)

    if not image_ids:
        return jsonify({'error': 'No image IDs provided'}), 400

    # Get images
    images = []
    for image_id in image_ids:
        image = db.get_image_by_id(image_id)
        if image:
            images.append(image)

    if not images:
        return jsonify({'error': 'No valid images found'}), 404

    # Generate JSON
    if include_summary:
        json_content = MetadataExporter.to_json_with_summary(images)
    else:
        json_content = MetadataExporter.to_json(images)

    # Return as downloadable file
    return json_content, 200, {
        'Content-Type': 'application/json',
        'Content-Disposition': f'attachment; filename="image_metadata.json"'
    }

@app.route('/api/export/boards/<int:board_id>/csv', methods=['GET'])
def export_board_csv(board_id):
    """Export board images metadata to CSV"""
    # Get board info
    board = db.get_board_by_id(board_id)
    if not board:
        return jsonify({'error': 'Board not found'}), 404

    # Get board images
    images = db.get_board_images(board_id)

    if not images:
        return jsonify({'error': 'No images in board'}), 404

    # Generate CSV
    csv_content = MetadataExporter.to_csv(images)

    # Return as downloadable file
    board_name = board['name'].replace(' ', '_')
    return csv_content, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': f'attachment; filename="{board_name}_metadata.csv"'
    }

@app.route('/api/export/boards/<int:board_id>/json', methods=['GET'])
def export_board_json(board_id):
    """Export board images metadata to JSON"""
    # Get board info
    board = db.get_board_by_id(board_id)
    if not board:
        return jsonify({'error': 'Board not found'}), 404

    # Get board images
    images = db.get_board_images(board_id)

    if not images:
        return jsonify({'error': 'No images in board'}), 404

    # Generate JSON with summary
    json_content = MetadataExporter.to_json_with_summary(images, board_info=board)

    # Return as downloadable file
    board_name = board['name'].replace(' ', '_')
    return json_content, 200, {
        'Content-Type': 'application/json',
        'Content-Disposition': f'attachment; filename="{board_name}_metadata.json"'
    }

@app.route('/api/export/boards/<int:board_id>/pdf', methods=['POST'])
def export_board_pdf(board_id):
    """Generate PDF catalog for a board"""
    # Get board info
    board = db.get_board_by_id(board_id)
    if not board:
        return jsonify({'error': 'Board not found'}), 404

    # Get board images
    images = db.get_board_images(board_id)

    if not images:
        return jsonify({'error': 'No images in board'}), 404

    # Get request options
    data = request.get_json() or {}
    page_size = data.get('page_size', 'A4')  # A4 or letter
    orientation = data.get('orientation', 'portrait')  # portrait or landscape

    # Map page size string to reportlab constant
    from reportlab.lib.pagesizes import A4, letter
    page_size_map = {
        'A4': A4,
        'letter': letter
    }
    page_size_obj = page_size_map.get(page_size, A4)

    # Create temporary PDF file
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_path = temp_file.name
    temp_file.close()

    try:
        # Generate PDF
        generator = PDFCatalogGenerator(page_size=page_size_obj, orientation=orientation)

        # Get thumbnail directory
        thumbnail_dir = os.path.join(DATA_DIR, 'thumbnails')

        generator.generate_board_catalog(
            board_info=board,
            images=images,
            output_path=temp_path,
            data_dir=PHOTOS_DIR,
            thumbnail_dir=thumbnail_dir
        )

        # Send file
        board_name = board['name'].replace(' ', '_')
        filename = f"{board_name}_catalog.pdf"

        @after_this_request
        def remove_file(response):
            try:
                os.remove(temp_path)
            except Exception:
                pass
            return response

        return send_file(
            temp_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        # Clean up temp file on error
        try:
            os.remove(temp_path)
        except:
            pass
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500

@app.route('/api/export/images/pdf', methods=['POST'])
def export_images_pdf():
    """Generate PDF catalog from selected images"""
    data = request.get_json() or {}
    image_ids = data.get('image_ids', [])
    title = data.get('title', 'Image Catalog')
    subtitle = data.get('subtitle', None)
    page_size = data.get('page_size', 'A4')
    orientation = data.get('orientation', 'portrait')

    if not image_ids:
        return jsonify({'error': 'No image IDs provided'}), 400

    # Get images
    images = []
    for image_id in image_ids:
        image = db.get_image_by_id(image_id)
        if image:
            images.append(image)

    if not images:
        return jsonify({'error': 'No valid images found'}), 404

    # Map page size string to reportlab constant
    from reportlab.lib.pagesizes import A4, letter
    page_size_map = {
        'A4': A4,
        'letter': letter
    }
    page_size_obj = page_size_map.get(page_size, A4)

    # Create temporary PDF file
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_path = temp_file.name
    temp_file.close()

    try:
        # Generate PDF
        generator = PDFCatalogGenerator(page_size=page_size_obj, orientation=orientation)

        # Get thumbnail directory
        thumbnail_dir = os.path.join(DATA_DIR, 'thumbnails')

        generator.generate_catalog(
            images=images,
            output_path=temp_path,
            title=title,
            subtitle=subtitle,
            data_dir=PHOTOS_DIR,
            thumbnail_dir=thumbnail_dir
        )

        # Send file
        filename = f"{title.replace(' ', '_')}.pdf"

        @after_this_request
        def remove_file(response):
            try:
                os.remove(temp_path)
            except Exception:
                pass
            return response

        return send_file(
            temp_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        # Clean up temp file on error
        try:
            os.remove(temp_path)
        except:
            pass
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500

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