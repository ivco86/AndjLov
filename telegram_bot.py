#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram Bot for AI Gallery
Automatically saves photos from Telegram groups to the gallery and analyzes them with AI
"""

import os
import sys
import asyncio
import logging
import mimetypes
from pathlib import Path
from datetime import datetime
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters
import requests
from database import Database

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
GALLERY_API_URL = os.getenv('GALLERY_API_URL', 'http://localhost:5000')
PHOTOS_DIR = os.getenv('PHOTOS_DIR', 'photos')
AUTO_ANALYZE = os.getenv('AUTO_ANALYZE', 'true').lower() == 'true'
AI_STYLE = os.getenv('AI_STYLE', 'classic')  # classic, detailed, tags, custom

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.m4v'}

# Allowed group/user IDs (empty = allow all)
ALLOWED_CHATS = os.getenv('ALLOWED_CHATS', '').split(',') if os.getenv('ALLOWED_CHATS') else []

# Ensure photos directory exists
os.makedirs(PHOTOS_DIR, exist_ok=True)

# Database instance
db = Database()


class TelegramGalleryBot:
    """Telegram bot for saving photos to AI Gallery"""

    def __init__(self, token: str):
        self.token = token
        self.app = None
    
    def _sanitize_token(self, value: str) -> str:
        return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in value)
    
    def _build_filename(self, update: Update, original_name: str, default_ext: str,
                        allowed_exts=None) -> str:
        """
        Build a safe filename using the original name when available.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        user = update.effective_user
        username = (user.username or user.first_name or "user") if user else "user"
        safe_username = self._sanitize_token(username) or "user"
        
        base_name = Path(original_name).name if original_name else ""
        stem = Path(base_name).stem if base_name else ""
        ext = Path(base_name).suffix.lower() if base_name else ""
        
        if not stem:
            stem = f"telegram_{safe_username}_{timestamp}"
        
        if not ext:
            ext = default_ext
        elif allowed_exts and ext not in allowed_exts:
            ext = default_ext
        
        safe_stem = self._sanitize_token(stem)
        return f"{safe_stem}{ext}"
    
    def _guess_extension(self, mime_type: str, fallback: str) -> str:
        if mime_type:
            guessed = mimetypes.guess_extension(mime_type)
            if guessed:
                return guessed
        return fallback
    
    async def _run_analysis(self, update: Update, image_id: int):
        await update.message.reply_text("🤖 Analyzing with AI...")
        try:
            response = requests.post(
                f"{GALLERY_API_URL}/api/images/{image_id}/analyze",
                json={'style': AI_STYLE},
                timeout=120
            )
            
            if response.ok:
                data = response.json()
                description = data.get('description', 'No description')
                tags = data.get('tags', [])
                tags_text = ', '.join(tags[:5]) if tags else 'No tags'
                
                await update.message.reply_text(
                    f"✨ *Analysis Complete!*\n\n"
                    f"📝 {description}\n\n"
                    f"🏷️ Tags: {tags_text}",
                    parse_mode='Markdown'
                )
            else:
                logger.error(f"Analysis API error: {response.status_code}")
                await update.message.reply_text("⚠️ AI analysis failed (API error)")
        except requests.Timeout:
            logger.error("Analysis timeout")
            await update.message.reply_text("⚠️ AI analysis timeout (will retry later)")
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            await update.message.reply_text(f"⚠️ AI analysis error: {str(e)}")
    
    async def _process_media(self, update: Update, telegram_file, filename: str,
                             media_type: str, width=None, height=None, file_size=None):
        filepath = os.path.join(PHOTOS_DIR, filename)
        await telegram_file.download_to_drive(filepath)
        
        chat = update.effective_chat
        chat_name = chat.title or chat.username or chat.id
        media_label = "Video" if media_type == 'video' else "Image"
        logger.info(f"Downloaded {media_label.lower()} from {chat_name}: {filename}")
        
        image_id = db.add_image(
            filepath,
            filename=filename,
            width=width,
            height=height,
            file_size=file_size,
            media_type=media_type
        )
        
        if image_id:
            await update.message.reply_text(
                f"✅ {media_label} saved to gallery!\n"
                f"📝 File: `{filename}`\n"
                f"🆔 Image ID: {image_id}",
                parse_mode='Markdown'
            )
            
            if AUTO_ANALYZE:
                await self._run_analysis(update, image_id)
        else:
            await update.message.reply_text(f"❌ Failed to add {media_label.lower()} to gallery")
        
        return image_id

    async def send_photo_to_chat(self, chat_id: int, filepath: str, caption: str = None):
        """
        Send a photo from the gallery to a Telegram chat

        Args:
            chat_id: Telegram chat ID (group or user)
            filepath: Path to the photo file
            caption: Optional caption for the photo

        Returns:
            dict: Result with success status and message
        """
        try:
            if not os.path.exists(filepath):
                return {'success': False, 'error': 'File not found'}

            # Check if file is an image
            ext = Path(filepath).suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                return {'success': False, 'error': 'Not a valid image file'}

            # Send photo using the application's bot
            with open(filepath, 'rb') as photo_file:
                message = await self.app.bot.send_photo(
                    chat_id=chat_id,
                    photo=photo_file,
                    caption=caption,
                    parse_mode='Markdown' if caption else None
                )

            logger.info(f"Photo sent to chat {chat_id}: {filepath}")
            return {
                'success': True,
                'message_id': message.message_id,
                'chat_id': chat_id
            }

        except Exception as e:
            logger.error(f"Error sending photo to chat {chat_id}: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        chat_id = update.effective_chat.id
        chat_type = update.effective_chat.type

        welcome_msg = (
            "🖼️ *AI Gallery Bot*\n\n"
            "I automatically save photos to your AI Gallery and analyze them with AI!\n\n"
            "*How to use:*\n"
            "• Send me a photo or add me to a group\n"
            "• I'll automatically save and analyze all photos\n"
            "• Use /status to check my status\n\n"
            f"*Current chat:*\n"
            f"• ID: `{chat_id}`\n"
            f"• Type: {chat_type}\n\n"
            f"*Settings:*\n"
            f"• Auto-analyze: {'✅' if AUTO_ANALYZE else '❌'}\n"
            f"• AI Style: {AI_STYLE}\n"
        )

        await update.message.reply_text(welcome_msg, parse_mode='Markdown')

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            # Get gallery stats
            response = requests.get(f"{GALLERY_API_URL}/api/health", timeout=5)
            if response.ok:
                data = response.json()
                stats = data.get('stats', {})

                status_msg = (
                    "📊 *Gallery Status*\n\n"
                    f"• Total Images: {stats.get('total_images', 0)}\n"
                    f"• Analyzed: {stats.get('analyzed_images', 0)}\n"
                    f"• Unanalyzed: {stats.get('unanalyzed_images', 0)}\n"
                    f"• Favorites: {stats.get('favorite_images', 0)}\n"
                    f"• Boards: {stats.get('total_boards', 0)}\n\n"
                    f"• AI Status: {'🟢 Connected' if data.get('ai_connected') else '🔴 Offline'}\n"
                    f"• Bot Status: 🟢 Active\n"
                )
            else:
                status_msg = "⚠️ Cannot connect to gallery API"

        except Exception as e:
            logger.error(f"Error getting status: {e}")
            status_msg = f"❌ Error: {str(e)}"

        await update.message.reply_text(status_msg, parse_mode='Markdown')

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_msg = (
            "🤖 *Bot Commands*\n\n"
            "/start - Show welcome message\n"
            "/status - Check gallery statistics\n"
            "/help - Show this help message\n\n"
            "*Features:*\n"
            "• Automatic photo saving\n"
            "• AI analysis with descriptions and tags\n"
            "• Works in groups and private chats\n"
            "• Supports JPG, PNG, WebP formats\n"
        )

        await update.message.reply_text(help_msg, parse_mode='Markdown')

    def is_chat_allowed(self, chat_id: int) -> bool:
        """Check if chat is allowed to use the bot"""
        if not ALLOWED_CHATS:
            return True  # Allow all if no restrictions
        return str(chat_id) in ALLOWED_CHATS

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming photos"""
        chat_id = update.effective_chat.id
        chat_name = update.effective_chat.title or update.effective_chat.username or "Unknown"

        # Check if chat is allowed
        if not self.is_chat_allowed(chat_id):
            logger.warning(f"Ignoring photo from unauthorized chat: {chat_id}")
            return

        try:
            # Get the largest photo
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            filename = self._build_filename(update, None, '.jpg', IMAGE_EXTENSIONS)

            await self._process_media(
                update,
                file,
                filename,
                media_type='image',
                width=getattr(photo, 'width', None),
                height=getattr(photo, 'height', None),
                file_size=getattr(photo, 'file_size', None)
            )

        except Exception as e:
            logger.error(f"Error handling photo: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def handle_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming videos"""
        chat_id = update.effective_chat.id

        if not self.is_chat_allowed(chat_id):
            logger.warning(f"Ignoring video from unauthorized chat: {chat_id}")
            return

        try:
            video = update.message.video
            if not video:
                return

            file = await context.bot.get_file(video.file_id)
            default_ext = self._guess_extension(video.mime_type, '.mp4')
            filename = self._build_filename(update, video.file_name, default_ext, VIDEO_EXTENSIONS)

            await self._process_media(
                update,
                file,
                filename,
                media_type='video',
                width=getattr(video, 'width', None),
                height=getattr(video, 'height', None),
                file_size=getattr(video, 'file_size', None)
            )
        except Exception as e:
            logger.error(f"Error handling video: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming documents (uncompressed photos or videos)"""
        chat_id = update.effective_chat.id

        # Check if chat is allowed
        if not self.is_chat_allowed(chat_id):
            return

        document = update.message.document

        mime_type = document.mime_type or ''
        media_type = None
        allowed_exts = None
        default_ext = '.jpg'

        if mime_type.startswith('image/'):
            media_type = 'image'
            allowed_exts = IMAGE_EXTENSIONS
            default_ext = self._guess_extension(mime_type, '.jpg')
        elif mime_type.startswith('video/'):
            media_type = 'video'
            allowed_exts = VIDEO_EXTENSIONS
            default_ext = self._guess_extension(mime_type, '.mp4')

        if not media_type:
            logger.info(f"Ignoring unsupported document type: {mime_type}")
            return

        try:
            file = await context.bot.get_file(document.file_id)
            filename = self._build_filename(update, document.file_name, default_ext, allowed_exts)

            await self._process_media(
                update,
                file,
                filename,
                media_type=media_type,
                file_size=getattr(document, 'file_size', None)
            )
        except Exception as e:
            logger.error(f"Error handling document: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")

    def run(self):
        """Start the bot"""
        if not self.token:
            logger.error("TELEGRAM_BOT_TOKEN not set!")
            print("❌ Error: TELEGRAM_BOT_TOKEN environment variable is not set")
            print("\nHow to set it:")
            print("  export TELEGRAM_BOT_TOKEN='your_token_here'")
            print("\nOr create a .env file with:")
            print("  TELEGRAM_BOT_TOKEN=your_token_here")
            return

        logger.info("Starting Telegram Gallery Bot...")

        # Create application
        self.app = Application.builder().token(self.token).build()

        # Add handlers
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("status", self.status_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        self.app.add_handler(MessageHandler(filters.VIDEO, self.handle_video))
        self.app.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))

        # Start bot
        print("🤖 Telegram Gallery Bot is running...")
        print(f"📁 Photos directory: {PHOTOS_DIR}")
        print(f"🌐 Gallery API: {GALLERY_API_URL}")
        print(f"🤖 Auto-analyze: {AUTO_ANALYZE}")
        print(f"🎨 AI Style: {AI_STYLE}")
        if ALLOWED_CHATS:
            print(f"🔒 Allowed chats: {', '.join(ALLOWED_CHATS)}")
        else:
            print("🌍 Accepting photos from all chats")
        print("\nPress Ctrl+C to stop")

        self.app.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Main entry point"""
    bot = TelegramGalleryBot(TELEGRAM_BOT_TOKEN)
    bot.run()


if __name__ == '__main__':
    main()
