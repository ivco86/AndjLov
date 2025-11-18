# 🖼️ AI Gallery

A powerful local photo gallery application that uses AI to automatically describe and tag your images.

## ✨ Features

- 🤖 **AI-Powered**: Automatic image analysis using LM Studio (local AI)
- 📁 **Smart Organization**: Create boards and sub-boards to organize your photos
- 🔍 **Full-Text Search**: Search across filenames, descriptions, and tags
- ⭐ **Favorites**: Mark and filter your favorite images
- ✏️ **File Management**: Rename files directly from the UI
- 🌐 **Fully Offline**: No cloud dependencies, everything runs locally
- 🎨 **Beautiful UI**: Clean, dark-themed interface optimized for photo viewing
- 📱 **Responsive**: Works on desktop, tablet, and mobile
- 📲 **Telegram Bot**: Automatically save photos from Telegram groups to your gallery

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- [LM Studio](https://lmstudio.ai/) with a vision-capable model (e.g., LLaVA)

### Installation

1. **Clone or download this repository**

```bash
git clone <repository-url>
cd ai-gallery
```

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure your photos directory**

Edit the `PHOTOS_DIR` variable in `app.py` or set an environment variable:

```bash
export PHOTOS_DIR="/path/to/your/photos"
```

4. **Start LM Studio**

- Open LM Studio
- Load a vision-capable model (e.g., llava-v1.6-vicuna-7b)
- Start the local server (usually on port 1234)

5. **Run the application**

```bash
python app.py
```

6. **Open in browser**

Navigate to: http://localhost:5000

## 📖 Usage Guide

### First Time Setup

1. **Scan Your Photos**
   - Click the "📂 Scan Directory" button
   - The app will find all images in your photos folder
   - Images are indexed into the database

2. **Analyze Images**
   - Click "🤖 Analyze Batch" to analyze multiple images
   - Or click the analyze button on individual images
   - AI will generate descriptions and tags

3. **Create Boards**
   - Click the "+" button next to "Boards"
   - Create collections to organize your photos
   - You can create sub-boards for hierarchical organization

4. **Add Images to Boards**
   - Click any image to open details
   - Click "📋 Add to Board"
   - Select which boards to add the image to

### Key Features

#### Search
- Type in the search box to find images
- Searches across filenames, descriptions, and tags
- Results update in real-time

#### Favorites
- Click the star icon to mark images as favorites
- Click "Favorites" in sidebar to view only favorites

#### Rename Files
- Click any image, then click ✏️ (rename button)
- Enter new filename
- File is renamed on disk and database is updated

#### Board Organization
- Boards can contain multiple images
- Images can belong to multiple boards
- Create sub-boards for nested organization
- Click any board to view its images

## 🛠️ Configuration

### Environment Variables

```bash
PHOTOS_DIR=./photos              # Directory containing your photos
LM_STUDIO_URL=http://localhost:1234  # LM Studio API endpoint
DATABASE_PATH=data/gallery.db    # SQLite database location
SERVER_HOST=0.0.0.0             # Server bind address
SERVER_PORT=5000                # Server port
```

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)
- BMP (.bmp)

## 🏗️ Architecture

```
ai-gallery/
├── app.py              # Flask application & API endpoints
├── database.py         # Database layer (SQLite operations)
├── ai_service.py       # LM Studio integration
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html     # Main HTML template
├── static/
│   ├── css/
│   │   └── styles.css # Application styles
│   └── js/
│       └── app.js     # Frontend JavaScript
└── data/
    └── gallery.db     # SQLite database (auto-created)
```

## 📊 Database Schema

### Images Table
- Stores image metadata, descriptions, and tags
- Full-text search index for fast queries

### Boards Table
- Hierarchical board structure (supports sub-boards)
- Optional cover images and descriptions

### Board-Images Relationship
- Many-to-many relationship
- Images can belong to multiple boards

## 🔧 API Endpoints

### Images
- `GET /api/images` - List all images
- `GET /api/images/:id` - Get image details
- `GET /api/images/:id/file` - Serve image file
- `GET /api/images/:id/thumbnail` - Serve thumbnail
- `POST /api/images/:id/favorite` - Toggle favorite
- `POST /api/images/:id/analyze` - Analyze image with AI
- `POST /api/images/:id/rename` - Rename file
- `GET /api/images/search?q=query` - Search images

### System
- `GET /api/health` - Check system health & AI status
- `POST /api/scan` - Scan directory for new images
- `POST /api/analyze-batch` - Batch analyze images

### Boards
- `GET /api/boards` - List all boards
- `POST /api/boards` - Create board
- `GET /api/boards/:id` - Get board with images
- `PUT /api/boards/:id` - Update board
- `DELETE /api/boards/:id` - Delete board
- `POST /api/boards/:id/images` - Add image to board
- `DELETE /api/boards/:id/images` - Remove image from board

## 🐛 Troubleshooting

### AI Not Connected
- Make sure LM Studio is running
- Check that a vision model is loaded
- Verify the URL in config (default: http://localhost:1234)

### Photos Not Found
- Check that PHOTOS_DIR path is correct
- Make sure the directory exists and is readable
- Use absolute paths if relative paths don't work

### Database Locked
- Close any other instances of the app
- Check file permissions on data/gallery.db
- Restart the application

### Images Not Loading
- Verify files exist on disk
- Check file permissions
- Try rescanning the directory

## 🎯 Performance Tips

- **For 1,000+ images**: Enable pagination (already implemented)
- **Slow AI analysis**: Reduce batch size or use smaller model
- **Thumbnails**: Generated on-the-fly, cached by browser
- **Database**: Regularly vacuum SQLite for optimal performance

## 🔐 Security Notes

- **Local Use Only**: By default, binds to 0.0.0.0 (accessible on network)
- **No Authentication**: Not recommended for public internet access
- **File Safety**: Validates file paths to prevent directory traversal
- **For Network Use**: Consider adding authentication or firewall rules

## 📲 Telegram Bot Integration

Save photos from Telegram groups directly to your gallery with automatic AI analysis!

### Quick Setup

1. **Create a bot with [@BotFather](https://t.me/botfather)**
   ```
   /newbot
   ```

2. **Configure the bot**
   ```bash
   cp .env.example .env
   # Edit .env and add your TELEGRAM_BOT_TOKEN
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the bot**
   ```bash
   ./start_bot.sh    # Linux/Mac
   start_bot.bat     # Windows
   ```

### Features

- ✨ Auto-save photos from groups
- 🤖 Automatic AI analysis
- 📊 Gallery statistics via `/status`
- 🔒 Optional access control by chat ID
- 👥 Works in groups and private chats

For detailed setup instructions, see [TELEGRAM_BOT.md](TELEGRAM_BOT.md)

## 📝 Future Enhancements

### Planned Features
- Drag-and-drop image management
- Bulk operations (select multiple images)
- Export boards as ZIP archives
- Video support
- Face detection and recognition
- Duplicate image detection
- Timeline view by date
- Map view using GPS data

### Advanced Features
- CLIP embeddings for visual similarity
- Background folder watching
- Mobile app (Electron/Tauri)
- Cloud backup integration
- Multi-user support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

MIT License - feel free to use this project for personal or commercial purposes.

## 💡 Tips & Tricks

1. **Keyboard Shortcuts**
   - `Ctrl/Cmd + K` - Focus search
   - `ESC` - Close modals

2. **Efficient Workflow**
   - Scan directory periodically for new images
   - Analyze images in batches (10-20 at a time)
   - Create board structure before adding images
   - Use descriptive board names for easy navigation

3. **AI Prompts**
   - Default prompt can be customized in `ai_service.py`
   - Adjust temperature for more/less creative descriptions
   - Model choice affects quality and speed

4. **Performance**
   - Keep thumbnail size reasonable (300px default)
   - Use FTS (full-text search) for fast queries
   - Regular database maintenance improves speed

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section

---

**Built with ❤️ using Flask, SQLite, and LM Studio**

Enjoy organizing your photos with AI! 🎉
#   P u s h  
 #   r e i m a g i n e d - m e m e  
 #   A n d j e l a  
 #   A n d l o v  
 #   r e i m a g i n e d - m e m e  
 #   A n d j L o v  
 #   A n d j L o v  
 #   A n d j L o v  
 