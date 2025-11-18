# 🖼️ AI Gallery

**Enterprise-Grade Local Photo Management with AI-Powered Automation**

A powerful, fully-featured photo gallery application that combines AI analysis, workflow automation, privacy protection, and research tools - all running locally on your machine.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🌟 Key Features

### 🤖 **AI-Powered Intelligence**
- Automatic image analysis using LM Studio (local AI)
- Multiple analysis styles: Classic, Detailed, Tags-focused
- Batch processing with progress tracking
- Privacy-aware AI analysis (local, no cloud)

### 📊 **Analytics & Insights**
- **Interactive Stats Dashboard** with Chart.js
- Timeline charts (daily/monthly/yearly)
- Top tags distribution
- Storage analytics by media type
- Recent activity tracking

### ⚡ **Workflow Automation**
- **Visual Pipeline Builder** - Create custom workflows
- **13 Built-in Actions**: AI analysis, tagging, boards, privacy, research, email
- **Scheduled Triggers**: Cron-like scheduling (daily, weekly, custom)
- **Conditional Logic**: If/Then/Else actions
- **Webhooks**: Zapier, n8n, Make.com integration
- **4 Pre-built Templates**: Auto-organize, Research workflow, Privacy-first, Favorites backup

### 🔒 **Privacy & Protection**
- Face detection (OpenCV)
- License plate detection
- NSFW content detection
- Privacy zones with blur/pixelate
- Batch privacy analysis
- Privacy mode for sensitive images

### 🎓 **Research & Education Mode**
- **Annotation Tools**: Bounding box drawing on canvas
- **Citation Generator**: APA, MLA, BibTeX, Chicago formats
- **Dataset Export**: COCO, YOLO, CSV formats
- **ML Training Ready**: Train/val/test splits (70/15/15)
- **Dataset Classes**: Color-coded annotation categories

### 📦 **Data Management**
- **Export/Import**: JSON, Markdown, CSV formats
- **Cloud Sync**: rclone integration (Google Drive, Dropbox, etc.)
- **Automated Backups**: Scheduled local & cloud backups
- **Backup Management**: Download, delete, restore

### 📧 **Email Notifications**
- SMTP email service (Gmail, Outlook, custom)
- Pipeline execution reports
- Image notifications with attachments
- HTML email templates
- Test connection functionality

### 🎨 **Enhanced User Experience**
- **Dark/Light Mode**: Smooth theme toggle with persistence
- **Enhanced Lightbox**: Zoom (0.5x-5x), pan, fullscreen
- **Keyboard Navigation**: Arrow keys, zoom shortcuts
- **Touch Gestures**: Pinch to zoom, swipe to navigate
- **Responsive Design**: Works on desktop, tablet, mobile

### 📱 **Telegram Bot Integration**
- Auto-save photos from Telegram
- AI analysis on upload
- Bot commands for gallery management
- Group and private chat support

### 🔍 **Smart Organization**
- Full-text search across descriptions and tags
- Hierarchical boards with drag-drop sub-boards
- Multi-board image assignment
- Favorites filtering
- Duplicate detection
- Smart sorting options

---

## 🚀 Quick Start

### Prerequisites

```bash
# Required
Python 3.8+
SQLite 3

# Optional (for additional features)
LM Studio (for AI analysis)
rclone (for cloud sync)
APScheduler (for scheduled triggers)
```

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd AndjLov
```

2. **Install dependencies**

```bash
pip install -r requirements.txt

# Optional dependencies
pip install apscheduler  # For scheduled triggers
pip install opencv-python  # For face/plate detection
```

3. **Configure environment**

```bash
# Create .env file (optional)
export PHOTOS_DIR="/path/to/your/photos"
export LM_STUDIO_URL="http://localhost:1234"
```

4. **Start LM Studio** (optional, for AI features)

- Download and install [LM Studio](https://lmstudio.ai/)
- Load a vision-capable model (e.g., `llava-v1.6-vicuna-7b`)
- Start the local server (default port: 1234)

5. **Run the application**

```bash
python app.py
```

6. **Open in browser**

```
http://localhost:5000
```

---

## 📖 Complete Feature Guide

### 1. 🤖 AI Analysis

**Setup:**
1. Start LM Studio with a vision model
2. Click "🤖 Analyze" button (top header)
3. Choose batch or single image analysis

**Analysis Styles:**
- **Classic**: Balanced descriptions
- **Detailed**: Comprehensive analysis
- **Tags**: Focus on keywords

**Features:**
- Batch analysis with progress bar
- Auto-tagging from AI descriptions
- Analysis history tracking

---

### 2. 📊 Statistics Dashboard

**Access:** Tools → 📊 Statistics Dashboard

**Includes:**
- **6 Stat Cards**: Total images, favorites, analyzed, storage, boards, tags
- **Timeline Chart**: Images added over time
- **Top Tags Chart**: Most used tags (bar chart)
- **Storage Chart**: Distribution by media type (doughnut)
- **Activity Panel**: Recent images and pipeline executions

**Refresh:** Click 🔄 Refresh button

---

### 3. ⚡ Workflow Automation

**Access:** Tools → ⚡ Workflow Automation

#### Creating a Pipeline:

1. **Click "Create Pipeline"**
2. **Set Basic Info:**
   - Name: e.g., "Auto-Organize Photos"
   - Description: What this pipeline does
   - Trigger: Manual, On Scan, Scheduled, or Webhook

3. **Add Actions:**
   - Click "Add Action"
   - Choose from 13 action types:
     - `ai_analyze` - Analyze with AI
     - `add_tags` - Add tags
     - `remove_tags` - Remove tags
     - `add_to_board` - Add to board
     - `remove_from_board` - Remove from board
     - `create_board` - Create new board
     - `privacy_analyze` - Privacy scan
     - `mark_favorite` - Mark as favorite
     - `generate_citation` - Generate citation
     - `create_annotation` - Add annotation
     - `log_message` - Log message
     - `send_email` - Send email notification

4. **Add Conditions (optional):**
   - Click "Add Condition" on any action
   - Choose condition type:
     - `has_tag` - Image has specific tag
     - `is_favorite` - Image is favorited
     - `has_analysis` - Image analyzed
     - `has_faces` - Face detected
     - `has_plates` - Plate detected
     - `is_nsfw` - NSFW content
     - `file_size_gt/lt` - File size comparison
     - `filename_contains` - Filename pattern
     - `and/or/not` - Logical operators

5. **Save and Run:**
   - Click "Save Pipeline"
   - Select images (✓ Select mode)
   - Click "Run Pipeline"

#### Scheduled Triggers:

**Setup Schedule:**
1. Create or edit pipeline
2. Set trigger type to "Scheduled"
3. Configure schedule:
   - **Preset**: Daily, Weekly, Monthly, Hourly
   - **Cron**: Custom cron expression
   - **Interval**: Every X hours/days

**Example Schedules:**
```
Daily at midnight:     0 0 * * *
Every Monday:          0 0 * * 1
First of month:        0 0 1 * *
Every 6 hours:         0 */6 * * *
```

#### Webhook Integration:

**Setup:**
1. Settings → Enable Webhooks
2. Copy generated token
3. Use webhook URL: `POST /api/webhooks/trigger/<pipeline_id>`
4. Add header: `Authorization: Bearer <token>`

**Example (cURL):**
```bash
curl -X POST http://localhost:5000/api/webhooks/trigger/1 \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"image_ids": [1, 2, 3]}'
```

**Use Cases:**
- Zapier automation
- n8n workflows
- Make.com scenarios
- Custom scripts

---

### 4. 🔒 Privacy & Protection

**Access:** Tools → 🔒 Privacy & Protection

**Features:**

1. **Privacy Analysis:**
   - Face detection
   - License plate detection
   - NSFW content detection

2. **Privacy Zones:**
   - Draw rectangles on sensitive areas
   - Blur or pixelate zones
   - Adjustable blur strength

3. **Privacy Mode:**
   - Auto-blur detected faces/plates
   - Configurable blur strength
   - Toggle on/off per image

4. **Batch Operations:**
   - Analyze all images
   - Analyze selected images
   - Apply blur to all

---

### 5. 🎓 Research & Education

**Access:** Tools → 🎓 Research & Education

#### Annotations:

1. **Create Annotation:**
   - Click "Draw Annotation"
   - Select class (or create new)
   - Draw bounding box on image
   - Save annotation

2. **Dataset Classes:**
   - Create custom classes
   - Assign colors
   - Add descriptions

#### Citations:

**Generate Citation:**
1. Select image
2. Choose format: APA, MLA, BibTeX, Chicago
3. Copy generated citation

**Example (APA):**
```
Author, A. (2024). Image Title. Retrieved from /path/to/image.jpg
```

#### Dataset Export:

**Export Formats:**

1. **COCO JSON:**
   - Industry-standard format
   - Object detection tasks
   - Compatible with PyTorch, TensorFlow

2. **YOLO TXT:**
   - Class ID + normalized coordinates
   - One file per image
   - YOLO training ready

3. **CSV:**
   - Simple tabular format
   - Import into Excel/Pandas
   - Custom analysis

**Export Process:**
1. Click "Export Dataset"
2. Select format
3. Choose train/val/test split (70/15/15)
4. Download ZIP or view preview

---

### 6. 📦 Data Export/Import

**Access:** Tools → 📦 Export/Import Data

**Export Options:**

1. **JSON Export:**
   - Complete database dump
   - All metadata included
   - Machine-readable

2. **Markdown Export:**
   - Human-readable format
   - Image links included
   - Great for documentation

3. **CSV Export:**
   - Spreadsheet-compatible
   - Easy filtering/sorting
   - Import to Excel

**Import:**
1. Click "Import Data"
2. Select JSON file (exported previously)
3. Choose merge or replace
4. Import

**Cloud Sync:**

1. **Install rclone** (optional):
   ```bash
   # Linux/Mac
   curl https://rclone.org/install.sh | sudo bash

   # Windows
   # Download from https://rclone.org/downloads/
   ```

2. **Configure Remote:**
   ```bash
   rclone config
   # Follow prompts for Google Drive, Dropbox, etc.
   ```

3. **Enable Auto-Backup:**
   - Settings → Cloud Sync
   - Select rclone remote
   - Set backup interval (hours)
   - Enable auto-backup

4. **Manual Backup:**
   - Click "Create Backup"
   - Choose include images (yes/no)
   - Download or sync to cloud

---

### 7. 📧 Email Notifications

**Access:** Settings → 📧 Email Notifications

**Setup (Gmail Example):**

1. **Enable 2-Step Verification:**
   - Go to Google Account settings
   - Security → 2-Step Verification

2. **Generate App Password:**
   - Visit: https://myaccount.google.com/apppasswords
   - Create new app password
   - Copy 16-character password

3. **Configure in Settings:**
   ```
   SMTP Server:    smtp.gmail.com
   Port:           587
   Use TLS:        ✓ Yes
   Username:       your-email@gmail.com
   Password:       <16-char app password>
   From Email:     your-email@gmail.com
   From Name:      AI Gallery
   ```

4. **Test Connection:**
   - Click "🔍 Test Connection"
   - Verify success message

**Use in Pipelines:**
- Add "Send Email" action
- Configure recipient, subject, body
- Optionally attach image
- Run pipeline

---

### 8. 🎨 Dark Mode & Lightbox

#### Dark Mode:

**Toggle:** Click 🌙/☀️ button (top-right)

**Features:**
- Smooth transition (0.3s)
- Persists in localStorage
- All modals adapt
- Chart colors adjust

#### Enhanced Lightbox:

**Keyboard Shortcuts:**
```
Arrow Left/Right:  Navigate images
+/-:               Zoom in/out
0:                 Reset zoom
F:                 Fullscreen
ESC:               Close
```

**Mouse:**
- Scroll wheel to zoom
- Click and drag to pan (when zoomed)

**Touch (Mobile):**
- Pinch to zoom
- Swipe left/right to navigate
- Drag to pan when zoomed

**Zoom Controls:**
- Floating bar at bottom
- Zoom In/Out/Reset/Fullscreen buttons

---

### 9. 📁 Boards & Organization

**Create Board:**
1. Click "+" next to Boards
2. Enter name and description
3. Click Create

**Add Images to Board:**
1. Click image → "📋 Add to Board"
2. Select board(s)
3. Image appears in board

**Create Sub-Board:**
- Drag board onto another board
- Confirm to make it a sub-board

**Board Management:**
- Right-click board for options
- Rename, Delete, Merge boards
- Set cover image

---

### 10. 🔍 Search & Filtering

**Search Types:**
- **Full-text**: Searches descriptions, tags, filenames
- **Tag filter**: Click any tag to filter
- **Board filter**: Click board to view its images
- **Favorites**: Click "Favorites" in sidebar

**Advanced Search:**
- Combine multiple tags
- Search with wildcards
- Filter by analyzed status
- Filter by media type

---

## ⚙️ Configuration

### Environment Variables

```bash
# Required
PHOTOS_DIR=./photos                    # Your photos directory
DATABASE_PATH=data/gallery.db          # SQLite database

# Optional
LM_STUDIO_URL=http://localhost:1234    # AI service
SERVER_HOST=0.0.0.0                    # Bind address
SERVER_PORT=5000                       # Server port
```

### Supported Formats

**Images:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)
- BMP (.bmp)

**Videos:**
- MP4 (.mp4)
- WebM (.webm)
- OGG (.ogg)

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **AI Not Connected** 🔴

**Symptoms:** Red "AI Offline" indicator

**Solutions:**
```bash
# Check LM Studio is running
curl http://localhost:1234/v1/models

# Verify model is loaded
# In LM Studio → Check "Server" tab

# Test connection
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llava", "messages": [{"role": "user", "content": "Hi"}]}'
```

**Fix:**
1. Start LM Studio
2. Load vision model (llava-v1.6)
3. Start local server
4. Refresh AI Gallery page

---

#### 2. **ModuleNotFoundError**

**Error:** `ModuleNotFoundError: No module named 'flask'`

**Solution:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install specific module
pip install flask

# For optional features
pip install apscheduler opencv-python Pillow
```

---

#### 3. **Database Locked**

**Error:** `sqlite3.OperationalError: database is locked`

**Solutions:**
```bash
# Check for other instances
ps aux | grep "python app.py"
kill <PID>  # Kill if found

# Fix permissions
chmod 666 data/gallery.db

# Reset database (CAUTION: loses data)
rm data/gallery.db
python app.py  # Will recreate
```

---

#### 4. **Photos Not Found**

**Error:** Empty gallery after scan

**Solutions:**
```bash
# Verify directory exists
ls -la ./photos

# Check absolute path
export PHOTOS_DIR="/full/path/to/photos"
python app.py

# Verify permissions
chmod -R 755 ./photos
```

---

#### 5. **Email Not Sending**

**Symptoms:** Email test fails

**Solutions:**

**For Gmail:**
```
1. Enable 2-Step Verification
2. Create App Password (not regular password!)
3. Use smtp.gmail.com:587 with TLS
4. Check "Less secure app access" is OFF
```

**For Outlook:**
```
SMTP: smtp-mail.outlook.com
Port: 587
TLS: Yes
Password: Regular account password
```

**Test SMTP:**
```bash
python3 << EOF
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('your-email@gmail.com', 'app-password')
print("✅ SMTP connection successful!")
server.quit()
EOF
```

---

#### 6. **Scheduled Pipelines Not Running**

**Symptoms:** Schedules don't trigger

**Solutions:**
```bash
# Install APScheduler
pip install apscheduler

# Check logs for scheduler
tail -f logs/app.log | grep -i scheduler

# Verify pipeline is enabled
# In UI → Workflows → Check "Enabled" toggle

# Restart application
# Scheduler reloads on startup
```

---

#### 7. **Webhook 403 Forbidden**

**Error:** `403 Forbidden` when calling webhook

**Solutions:**
```bash
# Regenerate token
# Settings → Webhooks → Regenerate Token

# Verify Bearer token format
curl -X POST http://localhost:5000/api/webhooks/trigger/1 \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  --verbose

# Check webhook is enabled
# Settings → Webhooks → Enable checkbox
```

---

#### 8. **Dark Mode Not Persisting**

**Symptoms:** Theme resets on refresh

**Solutions:**
```javascript
// Check localStorage
console.log(localStorage.getItem('theme'));

// Clear and retry
localStorage.clear();
// Refresh page and toggle again

// Check browser privacy mode
// localStorage doesn't persist in incognito
```

---

#### 9. **Charts Not Displaying**

**Symptoms:** Empty chart containers

**Solutions:**
```bash
# Check Chart.js loaded
# In browser console:
console.log(typeof Chart);  # Should be "function"

# Verify CDN access
curl https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js

# Check for JavaScript errors
# Open Browser DevTools → Console
```

---

#### 10. **Port Already in Use**

**Error:** `Address already in use: 5000`

**Solutions:**
```bash
# Find process using port
lsof -i :5000   # Mac/Linux
netstat -ano | findstr :5000  # Windows

# Kill process
kill -9 <PID>   # Mac/Linux
taskkill /PID <PID> /F  # Windows

# Or use different port
export SERVER_PORT=5001
python app.py
```

---

## 🎯 Performance Optimization

### For Large Libraries (10,000+ images)

1. **Database Indexing:**
```sql
-- Already optimized, but verify:
CREATE INDEX IF NOT EXISTS idx_images_filepath ON images(filepath);
CREATE INDEX IF NOT EXISTS idx_images_analyzed ON images(analyzed_at);
```

2. **Pagination:**
```python
# Adjust limit in app.py
IMAGES_PER_PAGE = 100  # Default: 1000
```

3. **Thumbnail Caching:**
```bash
# Thumbnails generated on-the-fly
# Browser caches automatically
# For faster loading, pre-generate:

curl http://localhost:5000/api/images/{id}/thumbnail
# For each image
```

4. **Vacuum Database:**
```bash
sqlite3 data/gallery.db "VACUUM;"
```

5. **Batch Analysis:**
```python
# Process in smaller batches
# UI: Analyze 10-20 images at a time
# Prevents memory issues
```

---

## 🔐 Security Best Practices

### For Local Network Use:

1. **Firewall Rules:**
```bash
# Allow only local network
sudo ufw allow from 192.168.1.0/24 to any port 5000
```

2. **HTTPS (Optional):**
```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365

# Update app.py:
# app.run(ssl_context=('cert.pem', 'key.pem'))
```

3. **Authentication (Future):**
- Currently no built-in auth
- For public access, use reverse proxy with auth
- Example: nginx + basic auth

### Privacy Notes:

- ✅ All AI processing is local (no cloud)
- ✅ No telemetry or tracking
- ✅ Database is local SQLite
- ⚠️ Webhook tokens are sensitive (keep secret)
- ⚠️ Email passwords stored in DB (encrypt recommended)

---

## 📚 API Reference

### Full API Documentation

**Base URL:** `http://localhost:5000/api`

#### Images API

```bash
# List all images
GET /api/images?limit=100&offset=0&favorites_only=false

# Get image details
GET /api/images/:id

# Serve image file
GET /api/images/:id/file

# Serve thumbnail
GET /api/images/:id/thumbnail?size=300

# Toggle favorite
POST /api/images/:id/favorite

# Analyze with AI
POST /api/images/:id/analyze
Body: {"style": "classic"}

# Update image
PUT /api/images/:id
Body: {"title": "New Title", "description": "..."}

# Delete image
DELETE /api/images/:id

# Rename file
POST /api/images/:id/rename
Body: {"new_filename": "photo.jpg"}

# Search
GET /api/images/search?q=sunset+beach
```

#### Stats API

```bash
# Overview statistics
GET /api/stats/overview

# Timeline data
GET /api/stats/timeline

# Storage distribution
GET /api/stats/storage

# Recent activity
GET /api/stats/activity
```

#### Workflows API

```bash
# List pipelines
GET /api/pipelines

# Create pipeline
POST /api/pipelines
Body: {"name": "...", "trigger_type": "manual", "actions": [...]}

# Execute pipeline
POST /api/pipelines/:id/execute
Body: {"image_ids": [1, 2, 3]}

# Get templates
GET /api/pipelines/templates

# Schedule pipeline
POST /api/scheduler/pipelines/:id/schedule
Body: {"schedule_config": {"type": "cron", ...}}
```

#### Webhooks API

```bash
# Trigger pipeline via webhook
POST /api/webhooks/trigger/:pipeline_id
Headers: {"Authorization": "Bearer <token>"}
Body: {"image_ids": [1, 2, 3]}

# Get webhook config
GET /api/webhooks/config

# Save webhook config
POST /api/webhooks/config
Body: {"enabled": true}
```

---

## 🚀 Future Roadmap

### Planned Features (v2.0)

#### High Priority
- [ ] **User Authentication** - Multi-user support with login
- [ ] **EXIF Data Viewer** - Camera settings, GPS, timestamps
- [ ] **Map View** - Geographic visualization of photos
- [ ] **Advanced Filters** - Filter by date range, camera, location
- [ ] **Slideshow Mode** - Auto-advancing fullscreen slideshow
- [ ] **Mobile PWA** - Install as mobile app

#### Medium Priority
- [ ] **Vector Search** - CLIP embeddings for visual similarity
- [ ] **Face Recognition** - Group photos by people
- [ ] **Auto-Tagging** - ML-based tag suggestions
- [ ] **Batch Editing** - Apply edits to multiple images
- [ ] **Comments** - Add notes/comments to images
- [ ] **Sharing** - Generate shareable links

#### Advanced Features
- [ ] **Background Sync** - Watch folder for new images
- [ ] **Smart Albums** - Auto-updating collections
- [ ] **RAW Support** - CR2, NEF, ARW format support
- [ ] **Image Editing** - Crop, rotate, filters
- [ ] **Collaborative Boards** - Multi-user board sharing
- [ ] **API Webhooks Out** - Send events to external services

### Community Requests
- [ ] **Masonry Layout** - Pinterest-style grid
- [ ] **Color Palette** - Extract dominant colors
- [ ] **OCR** - Text extraction from images
- [ ] **QR Code Detection** - Auto-detect and extract QR codes
- [ ] **Panorama Stitching** - Combine photos into panoramas

---

## 🤝 Contributing

We welcome contributions! Here's how:

### Report Bugs
1. Check existing issues
2. Create new issue with:
   - Clear title
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots if applicable
   - Environment details

### Submit Features
1. Open feature request issue
2. Describe use case
3. Discuss implementation approach
4. Submit pull request

### Development Setup

```bash
# Fork repository
git clone https://github.com/YOUR_USERNAME/AndjLov.git
cd AndjLov

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest

# Format code
black .

# Lint
flake8 .

# Submit PR
git checkout -b feature/my-feature
git commit -m "Add my feature"
git push origin feature/my-feature
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

**Free to use for personal or commercial purposes.**

---

## 🙏 Acknowledgments

**Built with:**
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [SQLite](https://www.sqlite.org/) - Database
- [LM Studio](https://lmstudio.ai/) - Local AI
- [Chart.js](https://www.chartjs.org/) - Data visualization
- [APScheduler](https://apscheduler.readthedocs.io/) - Task scheduling

**Inspired by:**
- Google Photos
- Adobe Lightroom
- Plex Media Server

---

## 📞 Support

**Need help?**
- 📖 Read the documentation above
- 🐛 Check [Troubleshooting](#-troubleshooting) section
- 💬 Open an issue on GitHub
- 📧 Contact: [your-email@example.com]

**Useful Links:**
- [LM Studio Documentation](https://lmstudio.ai/docs)
- [rclone Documentation](https://rclone.org/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/en/3.0.x/)

---

## 💡 Tips & Best Practices

### 1. **Efficient Workflow**
```
Daily: Scan → Analyze new images → Auto-tag
Weekly: Review and organize into boards
Monthly: Backup to cloud, clean duplicates
```

### 2. **Keyboard Shortcuts**
```
Ctrl/Cmd + K     Focus search
ESC              Close modals
Arrow Keys       Navigate images (in lightbox)
+/-              Zoom in/out
F                Fullscreen
0                Reset zoom
```

### 3. **Pipeline Ideas**
```
Auto-Organizer:  on_scan → ai_analyze → privacy_analyze → add_tags
Research Flow:   ai_analyze → generate_citation → add_to_board:Research
Quality Check:   if file_size_lt 1MB → add_tag:low-quality
Email Digest:    scheduled daily → send_email with recent images
```

### 4. **Performance Tips**
- Keep database under 50MB (vacuum regularly)
- Analyze in batches of 10-20 images
- Use tags for better search performance
- Enable pagination for 1000+ images
- Pre-generate thumbnails for faster loading

---

**🎉 Enjoy organizing your photos with AI Gallery!**

Built with ❤️ by the community | Version 2.0 | © 2024
