"""
Utility functions for AI Gallery
Helper functions for file handling, video processing, and image operations
"""

import os
from PIL import Image, ImageDraw
from pathlib import Path

# Try to import opencv for video frame extraction
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: opencv-python not installed. Video thumbnails will use placeholders.")

# Supported file formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
VIDEO_FORMATS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.m4v'}
ALL_MEDIA_FORMATS = SUPPORTED_FORMATS | VIDEO_FORMATS


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


def get_full_filepath(filepath, photos_dir):
    """
    Convert relative filepath to absolute path.
    Handles both old format (./photos/image.jpg) and new format (image.jpg)

    Args:
        filepath: Relative or absolute file path
        photos_dir: Base photos directory

    Returns:
        Full absolute file path
    """
    if not filepath:
        return filepath

    # If already absolute path, return as-is
    if os.path.isabs(filepath):
        return filepath

    # Normalize path separators
    normalized = filepath.replace('\\', '/')
    photos_dir_normalized = photos_dir.replace('\\', '/').lstrip('./')

    # Check if path already contains photos_dir (old format)
    # Examples: "./photos/image.jpg", "photos/image.jpg", "./photos/subfolder/image.jpg"
    if (normalized.startswith(photos_dir_normalized + '/') or
        normalized.startswith('./' + photos_dir_normalized + '/')):
        # Path already includes photos_dir, return as-is
        return filepath

    # New format - relative path without photos_dir
    return os.path.join(photos_dir, filepath)


def is_safe_path(filepath, base_dir):
    """
    Security check: Ensure filepath is within base_dir (prevent path traversal)

    Args:
        filepath: File path to check
        base_dir: Base directory that should contain the file

    Returns:
        True if path is safe, False otherwise
    """
    abs_filepath = os.path.abspath(filepath)
    abs_base_dir = os.path.abspath(base_dir)

    return abs_filepath.startswith(abs_base_dir)
