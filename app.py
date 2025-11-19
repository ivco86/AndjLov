"""
AI Gallery - Flask Application
Main web server with REST API endpoints

This is the main entry point. All routes are organized into Blueprints in the routes/ folder.
"""

import os
import atexit
from flask import Flask, render_template, jsonify, send_from_directory

# Import shared configuration and services
from shared import PHOTOS_DIR, LM_STUDIO_URL, DATABASE_PATH

# Import all blueprints
from routes import system_bp, telegram_bp, images_bp, boards_bp, export_bp, ai_bp

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Register all blueprints
app.register_blueprint(system_bp)
app.register_blueprint(telegram_bp)
app.register_blueprint(images_bp)
app.register_blueprint(boards_bp)
app.register_blueprint(export_bp)
app.register_blueprint(ai_bp)


# ============ FRONTEND ROUTES ============

@app.route('/')
def index():
    """Serve main application page"""
    return render_template('index.html')


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


# ============ CLEANUP ============

def cleanup_telegram_bot():
    """Cleanup Telegram bot on exit"""
    import shared
    if shared.telegram_bot_process and shared.telegram_bot_process.poll() is None:
        print("🛑 Stopping Telegram bot...")
        shared.telegram_bot_process.terminate()
        shared.telegram_bot_process.wait(timeout=5)


atexit.register(cleanup_telegram_bot)


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
