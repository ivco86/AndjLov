#!/bin/bash

# AI Gallery Startup Script

echo "╔══════════════════════════════════════╗"
echo "║      AI Gallery - Starting...        ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install -q -r requirements.txt

# Check if photos directory exists
if [ ! -d "photos" ]; then
    echo "📁 Creating photos directory..."
    mkdir photos
fi

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "💾 Creating data directory..."
    mkdir data
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Starting AI Gallery..."
echo ""
echo "📝 Make sure LM Studio is running with a vision model!"
echo ""

# Start the application
python app.py
