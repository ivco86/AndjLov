"""
System routes - health checks, configuration, external apps
"""

import os
import subprocess
from flask import Blueprint, jsonify, request
from shared import db, ai, PHOTOS_DIR, LM_STUDIO_URL, EXTERNAL_APPS, save_external_apps
from utils import SUPPORTED_FORMATS, get_full_filepath

system_bp = Blueprint('system', __name__)


@system_bp.route('/api/health', methods=['GET'])
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


@system_bp.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify({
        'photos_dir': PHOTOS_DIR,
        'lm_studio_url': LM_STUDIO_URL,
        'supported_formats': list(SUPPORTED_FORMATS)
    })


@system_bp.route('/api/ai/styles', methods=['GET'])
def get_ai_styles():
    """Get available AI description styles"""
    return jsonify({
        'styles': ai.get_available_styles()
    })


@system_bp.route('/api/external-apps', methods=['GET'])
def get_external_apps():
    """Get list of external applications for opening images/videos"""
    return jsonify({
        'apps': EXTERNAL_APPS
    })


@system_bp.route('/api/settings/external-apps', methods=['POST'])
def add_external_app():
    """Add new external application"""
    global EXTERNAL_APPS

    data = request.get_json() or {}
    media_type = data.get('media_type')  # 'image' or 'video'
    app_data = data.get('app')

    if not media_type or media_type not in ['image', 'video']:
        return jsonify({'error': 'media_type must be "image" or "video"'}), 400

    if not app_data or not all(k in app_data for k in ['id', 'name', 'command']):
        return jsonify({'error': 'app must have id, name, and command'}), 400

    # Check if ID already exists
    existing = next((a for a in EXTERNAL_APPS.get(media_type, []) if a['id'] == app_data['id']), None)
    if existing:
        return jsonify({'error': f'App with id "{app_data["id"]}" already exists'}), 409

    # Add defaults
    app_data.setdefault('path', '')
    app_data.setdefault('enabled', True)

    # Add to list
    if media_type not in EXTERNAL_APPS:
        EXTERNAL_APPS[media_type] = []
    EXTERNAL_APPS[media_type].append(app_data)

    # Save to file
    if save_external_apps(EXTERNAL_APPS):
        return jsonify({'success': True, 'app': app_data})
    else:
        return jsonify({'error': 'Failed to save configuration'}), 500


@system_bp.route('/api/settings/external-apps/<media_type>/<app_id>', methods=['PUT'])
def update_external_app(media_type, app_id):
    """Update external application"""
    global EXTERNAL_APPS

    if media_type not in ['image', 'video']:
        return jsonify({'error': 'media_type must be "image" or "video"'}), 400

    data = request.get_json() or {}

    # Find app
    app_list = EXTERNAL_APPS.get(media_type, [])
    app_index = next((i for i, a in enumerate(app_list) if a['id'] == app_id), None)

    if app_index is None:
        return jsonify({'error': f'App "{app_id}" not found'}), 404

    # Update fields
    allowed_fields = ['name', 'command', 'path', 'enabled']
    for field in allowed_fields:
        if field in data:
            EXTERNAL_APPS[media_type][app_index][field] = data[field]

    # Save to file
    if save_external_apps(EXTERNAL_APPS):
        return jsonify({'success': True, 'app': EXTERNAL_APPS[media_type][app_index]})
    else:
        return jsonify({'error': 'Failed to save configuration'}), 500


@system_bp.route('/api/settings/external-apps/<media_type>/<app_id>', methods=['DELETE'])
def delete_external_app(media_type, app_id):
    """Delete external application"""
    global EXTERNAL_APPS

    if media_type not in ['image', 'video']:
        return jsonify({'error': 'media_type must be "image" or "video"'}), 400

    # Find and remove app
    app_list = EXTERNAL_APPS.get(media_type, [])
    app_index = next((i for i, a in enumerate(app_list) if a['id'] == app_id), None)

    if app_index is None:
        return jsonify({'error': f'App "{app_id}" not found'}), 404

    removed_app = EXTERNAL_APPS[media_type].pop(app_index)

    # Save to file
    if save_external_apps(EXTERNAL_APPS):
        return jsonify({'success': True, 'removed': removed_app})
    else:
        return jsonify({'error': 'Failed to save configuration'}), 500


@system_bp.route('/api/images/<int:image_id>/open-with', methods=['POST'])
def open_with_external_app(image_id):
    """Open image/video with external application"""
    try:
        image = db.get_image(image_id)
        if not image:
            return jsonify({'error': 'Image not found'}), 404

        filepath = get_full_filepath(image['filepath'], PHOTOS_DIR)
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

        # Check if app is enabled
        if not app.get('enabled', True):
            return jsonify({'error': f'Application {app["name"]} is disabled'}), 400

        # Get absolute path
        abs_filepath = os.path.abspath(filepath)

        # Determine executable path and build command
        app_path = app.get('path', '').strip()
        app_command = app.get('command', '').strip()

        if app_path:
            # Use custom path if specified
            command = [app_path, abs_filepath]
        elif app_command == 'system':
            # System default - use OS-specific opener
            import platform
            system = platform.system()
            if system == 'Windows':
                # On Windows, use 'start' command with empty string
                command = ['cmd', '/c', 'start', '', abs_filepath]
            elif system == 'Darwin':
                command = ['open', abs_filepath]
            else:
                command = ['xdg-open', abs_filepath]
        else:
            # Use command name (assumes it's in PATH)
            command = [app_command, abs_filepath]

        print(f"[OPEN_WITH] Opening {abs_filepath} with {app['name']}")
        print(f"[OPEN_WITH] Command: {' '.join(command)}")

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
