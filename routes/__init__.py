"""
Flask Blueprints for AI Gallery
This package contains all route modules organized by functionality
"""

from .system import system_bp
from .telegram import telegram_bp
from .images import images_bp
from .boards import boards_bp
from .export import export_bp

__all__ = [
    'system_bp',
    'telegram_bp',
    'images_bp',
    'boards_bp',
    'export_bp'
]
