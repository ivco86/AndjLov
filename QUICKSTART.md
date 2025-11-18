# 🚀 AI Gallery - Quick Start Guide

## 📥 Installation (5 minutes)

### Step 1: Prerequisites
1. **Install Python 3.8+**
   - Windows: Download from python.org
   - Mac: `brew install python3`
   - Linux: `sudo apt install python3 python3-pip`

2. **Install LM Studio**
   - Download from: https://lmstudio.ai/
   - Load a vision model (e.g., "llava-v1.6-vicuna-7b")
   - Start the local server (Settings → Local Server → Start)

### Step 2: Setup AI Gallery

**Windows:**
```bash
# Extract the archive
# Double-click start.bat
```

**Mac/Linux:**
```bash
# Extract the archive
tar -xzf ai-gallery.tar.gz
cd ai-gallery

# Run startup script
./start.sh
```

**Manual Setup (all platforms):**
```bash
# Extract archive
cd ai-gallery

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

### Step 3: Configure Photos Directory

**Option A: Use default**
- Place photos in `ai-gallery/photos/` folder

**Option B: Point to existing folder**
- Edit `app.py` line 15:
  ```python
  PHOTOS_DIR = '/path/to/your/photos'
  ```

### Step 4: Access Gallery

Open browser: **http://localhost:5000**

---

## 🎯 First Steps

### 1. Scan Your Photos
- Click "📂 Scan Directory"
- Wait for indexing to complete
- Your images appear in the grid

### 2. Analyze with AI
- Click "🤖 Analyze Batch" (analyzes 10 images)
- Or click individual images → 🤖 button
- **Note:** Each image takes ~10-60 seconds

### 3. Create Boards
- Click "+" next to "Boards" in sidebar
- Name your board (e.g., "Vacation 2024")
- Create sub-boards for nested organization

### 4. Add Images to Boards
- Click any image to open details
- Click "📋 Add to Board"
- Check boxes for desired boards
- Click "Save"

### 5. Search & Filter
- Use search box to find images
- Click "Favorites" to view starred images
- Click "Unanalyzed" to see pending images

---

## 🔧 Common Issues

### "AI Offline" / Not Connected

**Solutions:**
1. Make sure LM Studio is running
2. Check a vision model is loaded
3. Verify local server is started in LM Studio
4. Try restarting LM Studio

### No Images Showing

**Solutions:**
1. Check PHOTOS_DIR path is correct
2. Click "Scan Directory" button
3. Verify photos folder has images
4. Check supported formats: JPG, PNG, GIF, WEBP, BMP

### Analysis Taking Forever

**Solutions:**
1. Reduce batch size (analyze fewer images)
2. Use a smaller/faster model in LM Studio
3. Ensure LM Studio has GPU acceleration enabled
4. Close other heavy applications

### Database Locked Error

**Solutions:**
1. Close any other instances of the app
2. Restart the application
3. Check file permissions on `data/gallery.db`

---

## 💡 Pro Tips

### Efficient Workflow
```
1. Scan directory → Index all images
2. Analyze in batches → Get descriptions/tags
3. Create board structure → Organize before adding
4. Add images to boards → Drag or use modal
5. Search & enjoy → Find anything instantly
```

### Keyboard Shortcuts
- `Ctrl/Cmd + K` - Focus search
- `ESC` - Close any modal
- Click image - Open details

### Performance Tips
- **Batch Analysis:** Do 10-20 images at a time
- **Search:** Use specific keywords for better results
- **Boards:** Keep hierarchies 2-3 levels deep max
- **Database:** Restart app if it feels slow

### AI Prompt Customization
Edit `ai_service.py` line 55-63 to change how AI describes images:

```python
prompt = """Analyze this image and provide:
1. Your custom instructions here
2. More specific requirements
"""
```

---

## 📊 What Gets Stored

### Database (`data/gallery.db`)
- Image metadata (path, size, dimensions)
- AI descriptions and tags
- Board structure
- Favorite status

### Not Stored
- Actual image files (stay on disk)
- Changes to original files
- User settings (future feature)

### Backup
```bash
# Backup database
cp data/gallery.db data/gallery.backup.db

# Backup entire project
tar -czf ai-gallery-backup.tar.gz ai-gallery/
```

---

## 🎓 Advanced Usage

### Custom Photo Directory
```bash
# Set via environment variable
export PHOTOS_DIR="/mnt/photos"
python app.py

# Or edit app.py directly
PHOTOS_DIR = '/path/to/photos'
```

### Run on Different Port
```bash
# Edit app.py, last few lines
app.run(host='0.0.0.0', port=8080)
```

### Network Access
- Default: Accessible on local network
- URL: `http://YOUR-IP:5000`
- **Warning:** No authentication by default
- **Secure:** Use VPN or firewall rules

### Multiple Photo Libraries
Run separate instances:
```bash
# Instance 1
PHOTOS_DIR=/path/1 python app.py

# Instance 2 (different terminal)
PHOTOS_DIR=/path/2 SERVER_PORT=5001 python app.py
```

---

## 📈 Scaling Up

### For 10,000+ Images
- **Thumbnails:** Consider pre-generating
- **Database:** Enable WAL mode for better concurrency
- **Analysis:** Use faster model or GPU acceleration
- **Pagination:** Already built-in, handles large sets

### Performance Monitoring
```python
# Check stats endpoint
curl http://localhost:5000/api/health
```

---

## 🆘 Getting Help

### Debug Mode
Set in `app.py`:
```python
app.run(debug=True)  # Shows detailed errors
```

### Check Logs
- Console output shows all operations
- API errors appear in browser console (F12)

### Test AI Connection
```bash
# Check if LM Studio is responding
curl http://localhost:1234/v1/models
```

---

## ✅ Success Checklist

- [ ] Python 3.8+ installed
- [ ] LM Studio running with vision model
- [ ] Photos directory configured
- [ ] App starts without errors
- [ ] Browser opens to localhost:5000
- [ ] AI status shows "🟢 Connected"
- [ ] Scan finds your images
- [ ] Analysis generates descriptions
- [ ] Search works
- [ ] Boards can be created

If all checked ✅ - you're ready to go! 🎉

---

## 🔄 Updates & Maintenance

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Clear Cache
```bash
# Remove thumbnails (regenerated automatically)
rm -rf /tmp/thumb_*

# Reset database (WARNING: Loses all data)
rm data/gallery.db
python app.py  # Creates fresh database
```

### Backup Before Updates
```bash
cp -r ai-gallery ai-gallery-backup
```

---

**Need more help? Check README.md for full documentation!**
