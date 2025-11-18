@echo off
REM AI Gallery Startup Script for Windows

echo ========================================
echo      AI Gallery - Starting...
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -q -r requirements.txt

REM Check if photos directory exists
if not exist "photos\" (
    echo Creating photos directory...
    mkdir photos
)

REM Check if data directory exists
if not exist "data\" (
    echo Creating data directory...
    mkdir data
)

echo.
echo Setup complete!
echo.
echo Starting AI Gallery...
echo.
echo IMPORTANT: Make sure LM Studio is running with a vision model!
echo.

REM Start the application
python app.py

pause
