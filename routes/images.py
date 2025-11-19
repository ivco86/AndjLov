"""
Image routes - image CRUD, analysis, search, tags, upload, scan
"""

import os
import mimetypes
import platform
import subprocess
from pathlib import Path
from flask import Blueprint, jsonify, request, send_file
from werkzeug.utils import secure_filename
from PIL import Image

from shared import db, ai, PHOTOS_DIR, DATA_DIR
from utils import (
    ALL_MEDIA_FORMATS, VIDEO_FORMATS, SUPPORTED_FORMATS,
    get_full_filepath, extract_video_frame, create_video_placeholder,
    get_image_for_analysis
)
from reverse_image_search import ReverseImageSearch, get_copyright_tips, get_usage_detection_tips

images_bp = Blueprint('images', __name__)


# ============ IMAGE LIST & DETAILS ============

@images_bp.route('/api/images', methods=['GET'])
def get_images():
    """Get all images with optional filters"""
    limit = request.args.get('limit', 1000, type=int)
    offset = request.args.get('offset', 0, type=int)
    favorites_only = request.args.get('favorites', 'false').lower() == 'true'

    media_type = request.args.get('media_type')
    if media_type:
        media_type = media_type.strip().lower()
        if media_type in ('all', 'any'):
            media_type = None

    analyzed_param = request.args.get('analyzed')
    analyzed = None
    if analyzed_param is not None:
        analyzed_param = analyzed_param.strip().lower()
        if analyzed_param in ('true', '1'):
            analyzed = True
        elif analyzed_param in ('false', '0'):
            analyzed = False

    images = db.get_all_images(
        limit=limit,
        offset=offset,
        favorites_only=favorites_only,
        media_type=media_type,
        analyzed=analyzed
    )

    return jsonify({
        'images': images,
        'count': len(images),
        'offset': offset,
        'limit': limit
    })


@images_bp.route('/api/images/<int:image_id>', methods=['GET'])
def get_image(image_id):
    """Get single image details"""
    image = db.get_image(image_id)

    if not image:
        return jsonify({'error': 'Image not found'}), 404

    # Get boards containing this image
    boards = db.get_image_boards(image_id)
    image['boards'] = boards

    return jsonify(image)


@images_bp.route('/api/images/<int:image_id>', methods=['PATCH'])
def update_image(image_id):
    """Update image description and tags"""
    data = request.json

    # Get current image
    image = db.get_image(image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404

    description = data.get('description', '')
    tags = data.get('tags', [])

    try:
        # Use the database method to update image analysis
        # Clean up tags - remove empty strings
        clean_tags = [tag.strip() for tag in tags if tag and tag.strip()]

        # Update using the proper database method
        db.update_image_analysis(image_id, description, clean_tags)

        # Get updated image
        updated_image = db.get_image(image_id)

        return jsonify({
            'success': True,
            'image': updated_image
        })
    except Exception as e:
        return jsonify({'error': f'Failed to update: {str(e)}'}), 500


# ============ IMAGE FILE SERVING ============

@images_bp.route('/api/images/<int:image_id>/file', methods=['GET'])
def serve_image(image_id):
    """Serve actual image file"""
    image = db.get_image(image_id)

    if not image:
        return jsonify({'error': 'Image not found'}), 404

    filepath = get_full_filepath(image['filepath'], PHOTOS_DIR)

    # Security: Validate filepath is within PHOTOS_DIR
    abs_filepath = os.path.abspath(filepath)
    abs_photos_dir = os.path.abspath(PHOTOS_DIR)

    if not abs_filepath.startswith(abs_photos_dir):
        print(f"Security: Path traversal attempt blocked: {filepath}")
        return jsonify({'error': 'Invalid file path'}), 403

    if not os.path.exists(abs_filepath):
        return jsonify({'error': 'File not found on disk'}), 404

    # Additional check: ensure it's actually a file, not a directory
    if not os.path.isfile(abs_filepath):
        return jsonify({'error': 'Invalid file'}), 403

    return send_file(abs_filepath, mimetype=mimetypes.guess_type(abs_filepath)[0])


@images_bp.route('/api/images/<int:image_id>/thumbnail', methods=['GET'])
def serve_thumbnail(image_id):
    """Serve thumbnail (resized image for grid) with caching"""
    size = request.args.get('size', 300, type=int)
    size = min(size, 1000)  # Prevent abuse

    image = db.get_image(image_id)

    if not image:
        return jsonify({'error': 'Image not found'}), 404

    filepath = get_full_filepath(image['filepath'], PHOTOS_DIR)

    # Security: Validate filepath is within PHOTOS_DIR
    abs_filepath = os.path.abspath(filepath)
    abs_photos_dir = os.path.abspath(PHOTOS_DIR)

    if not abs_filepath.startswith(abs_photos_dir):
        print(f"Security: Path traversal attempt blocked: {filepath}")
        return jsonify({'error': 'Invalid file path'}), 403

    if not os.path.exists(abs_filepath):
        return jsonify({'error': 'File not found on disk'}), 404

    if not os.path.isfile(abs_filepath):
        return jsonify({'error': 'Invalid file'}), 403

    # Thumbnail caching
    thumbnail_cache_dir = os.path.join(DATA_DIR, 'thumbnails')
    os.makedirs(thumbnail_cache_dir, exist_ok=True)

    # Check if this is a video
    is_video = image.get('media_type') == 'video'

    # Generate cache key from image ID, size, and modification time
    try:
        mtime = int(os.path.getmtime(abs_filepath))
        cache_filename = f"{image_id}_{size}_{mtime}.jpg"
        cache_path = os.path.join(thumbnail_cache_dir, cache_filename)

        # Check if cached thumbnail exists
        if os.path.exists(cache_path):
            return send_file(cache_path, mimetype='image/jpeg')

        # Generate and cache thumbnail
        if is_video:
            # Try to extract frame from video
            img = extract_video_frame(abs_filepath, cache_path, time_sec=1.0)

            if not img:
                # Fallback to placeholder if opencv not available or extraction failed
                img = create_video_placeholder(size)
        else:
            # Regular image processing
            img = Image.open(abs_filepath)

        # Resize thumbnail
        img.thumbnail((size, size), Image.Resampling.LANCZOS)

        # Higher quality for better visual appearance (92 is a good balance)
        img.save(cache_path, 'JPEG', quality=92, optimize=True)

        # Clean up old thumbnails for this image
        for old_file in os.listdir(thumbnail_cache_dir):
            if old_file.startswith(f"{image_id}_") and old_file != cache_filename:
                try:
                    os.remove(os.path.join(thumbnail_cache_dir, old_file))
                except:
                    pass

        return send_file(cache_path, mimetype='image/jpeg')
    except Exception as e:
        print(f"Error generating thumbnail: {e}")

        # For videos, try to return placeholder
        if is_video:
            try:
                img = create_video_placeholder(size)
                img.save(cache_path, 'JPEG', quality=92)
                return send_file(cache_path, mimetype='image/jpeg')
            except:
                pass

        # Fallback to original file
        return send_file(abs_filepath, mimetype=mimetypes.guess_type(abs_filepath)[0])


# ============ IMAGE OPERATIONS ============

@images_bp.route('/api/images/<int:image_id>/favorite', methods=['POST'])
def toggle_favorite(image_id):
    """Toggle favorite status"""
    new_status = db.toggle_favorite(image_id)

    return jsonify({
        'success': True,
        'image_id': image_id,
        'is_favorite': new_status
    })


@images_bp.route('/api/images/<int:image_id>/rename', methods=['POST'])
def rename_image(image_id):
    """Rename image file"""
    data = request.json
    new_filename = data.get('new_filename')

    if not new_filename:
        return jsonify({'error': 'new_filename is required'}), 400

    # Get current image
    image = db.get_image(image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404

    old_path = get_full_filepath(image['filepath'], PHOTOS_DIR)

    if not os.path.exists(old_path):
        return jsonify({'error': 'File not found on disk'}), 404

    # Sanitize filename
    new_filename = secure_filename(new_filename)

    # Keep same directory
    directory = os.path.dirname(old_path)
    new_path = os.path.join(directory, new_filename)

    # Check if target exists
    if os.path.exists(new_path):
        return jsonify({'error': 'File with that name already exists'}), 409

    try:
        # Rename file on disk
        os.rename(old_path, new_path)

        # Store relative path in database
        abs_photos_dir = os.path.abspath(PHOTOS_DIR)
        abs_new_path = os.path.abspath(new_path)
        relative_new_path = os.path.relpath(abs_new_path, abs_photos_dir)

        # Update database
        db.rename_image(image_id, relative_new_path, new_filename)

        return jsonify({
            'success': True,
            'image_id': image_id,
            'old_filename': image['filename'],
            'new_filename': new_filename,
            'new_filepath': new_path
        })
    except Exception as e:
        return jsonify({'error': f'Failed to rename: {str(e)}'}), 500


@images_bp.route('/api/images/<int:image_id>/open-folder', methods=['POST'])
def open_image_folder(image_id):
    """Open the folder containing the image in file explorer"""
    # Get image info
    image = db.get_image(image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404

    # Get full file path
    db_filepath = image.get('filepath', '')
    filepath = get_full_filepath(db_filepath, PHOTOS_DIR)

    # Debug logging
    print(f"[OPEN FOLDER DEBUG] Image ID: {image_id}")
    print(f"[OPEN FOLDER DEBUG] DB filepath: '{db_filepath}'")
    print(f"[OPEN FOLDER DEBUG] Full filepath: '{filepath}'")
    print(f"[OPEN FOLDER DEBUG] File exists: {os.path.exists(filepath)}")
    print(f"[OPEN FOLDER DEBUG] PHOTOS_DIR: '{PHOTOS_DIR}'")

    if not os.path.exists(filepath):
        print(f"[OPEN FOLDER DEBUG] ERROR: File not found!")
        return jsonify({
            'error': 'File not found on disk',
            'db_filepath': db_filepath,
            'full_filepath': filepath
        }), 404

    # Get directory path and absolute file path
    abs_filepath = os.path.abspath(filepath)
    folder_path = os.path.dirname(abs_filepath)
    print(f"[OPEN FOLDER DEBUG] Absolute filepath: '{abs_filepath}'")
    print(f"[OPEN FOLDER DEBUG] Folder to open: '{folder_path}'")

    try:
        system = platform.system()

        if system == 'Windows':
            # Windows: open Explorer and select the file (needs absolute path)
            print(f"[OPEN FOLDER DEBUG] Running: explorer /select, {abs_filepath}")
            subprocess.run(['explorer', '/select,', abs_filepath])
        elif system == 'Darwin':  # macOS
            # Mac: open Finder and select the file (needs absolute path)
            subprocess.run(['open', '-R', abs_filepath])
        else:  # Linux
            # Linux: open file manager in the folder
            # Try different file managers
            try:
                subprocess.run(['xdg-open', folder_path])
            except FileNotFoundError:
                try:
                    subprocess.run(['nautilus', folder_path])
                except FileNotFoundError:
                    try:
                        subprocess.run(['dolphin', folder_path])
                    except FileNotFoundError:
                        return jsonify({
                            'error': 'Could not detect file manager. Please install xdg-utils.',
                            'folder_path': folder_path
                        }), 500

        print(f"[OPEN FOLDER DEBUG] ✓ Command executed successfully")
        return jsonify({
            'success': True,
            'folder_path': folder_path
        })

    except Exception as e:
        return jsonify({
            'error': f'Failed to open folder: {str(e)}',
            'folder_path': folder_path
        }), 500


# ============ AI ANALYSIS ============

@images_bp.route('/api/images/<int:image_id>/analyze', methods=['POST'])
def analyze_image(image_id):
    """Analyze single image/video with AI and optionally auto-rename"""
    temp_image_path = None
    try:
        image = db.get_image(image_id)

        if not image:
            return jsonify({'error': 'Image not found'}), 404

        filepath = get_full_filepath(image['filepath'], PHOTOS_DIR)
        media_type = image.get('media_type', 'image')

        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found on disk'}), 404

        # Check AI connection
        connected, message = ai.check_connection()
        if not connected:
            return jsonify({'error': f'AI not available: {message}'}), 503

        # Get style and custom prompt from request
        data = request.get_json() or {}
        style = data.get('style', 'classic')
        custom_prompt = data.get('custom_prompt', None)

        # For videos, extract frame first
        analysis_path = filepath
        if media_type == 'video':
            print(f"[ANALYZE] Extracting frame from video {image_id} for AI analysis...")

            # Get frame as PIL Image
            frame_img = get_image_for_analysis(filepath, media_type='video')

            if not frame_img:
                return jsonify({'error': 'Could not extract frame from video for analysis'}), 500

            # Save frame to temporary file
            temp_dir = os.path.join(DATA_DIR, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, f"video_frame_{image_id}.jpg")
            frame_img.save(temp_image_path, 'JPEG', quality=95)
            analysis_path = temp_image_path

            print(f"[ANALYZE] Video frame extracted to {temp_image_path}")

        # Analyze image with specified style
        print(f"[ANALYZE] Analyzing image {image_id} with style '{style}'...")
        result = ai.analyze_image(analysis_path, style=style, custom_prompt=custom_prompt)

        if result:
            print(f"[ANALYZE] AI analysis complete for image {image_id}")
            print(f"[ANALYZE] Description: {result['description'][:100]}...")
            print(f"[ANALYZE] Tags: {result['tags']}")

            # Update database with analysis
            print(f"[ANALYZE] Updating database for image {image_id}...")
            db.update_image_analysis(
                image_id,
                result['description'],
                result['tags']
            )
            print(f"[ANALYZE] ✅ Database updated successfully for image {image_id}")

            # Auto-rename if AI suggested a filename
            new_filename = None
            renamed = False

            if result.get('suggested_filename'):
                suggested = result['suggested_filename'].strip()

                if suggested and len(suggested) > 0:
                    # Sanitize filename
                    suggested = secure_filename(suggested)

                    # Get original extension
                    old_ext = Path(filepath).suffix

                    # Build new filename
                    new_filename = f"{suggested}{old_ext}"

                    # Get directory
                    directory = os.path.dirname(filepath)
                    new_filepath = os.path.join(directory, new_filename)

                    # Check if different from current name
                    if new_filepath != filepath:
                        # Check if target exists
                        if not os.path.exists(new_filepath):
                            try:
                                # Rename file on disk
                                os.rename(filepath, new_filepath)

                                # Update database
                                db.rename_image(image_id, new_filepath, new_filename)

                                renamed = True
                                print(f"Auto-renamed: {image['filename']} → {new_filename}")
                            except Exception as e:
                                print(f"Auto-rename failed: {e}")
                                renamed = False
                        else:
                            # File exists, add counter
                            counter = 1
                            base_name = suggested
                            while os.path.exists(new_filepath) and counter < 100:
                                new_filename = f"{base_name}_{counter}{old_ext}"
                                new_filepath = os.path.join(directory, new_filename)
                                counter += 1

                            if not os.path.exists(new_filepath):
                                try:
                                    os.rename(filepath, new_filepath)
                                    db.rename_image(image_id, new_filepath, new_filename)
                                    renamed = True
                                    print(f"Auto-renamed: {image['filename']} → {new_filename}")
                                except Exception as e:
                                    print(f"Auto-rename failed: {e}")

            return jsonify({
                'success': True,
                'image_id': image_id,
                'description': result['description'],
                'tags': result['tags'],
                'renamed': renamed,
                'new_filename': new_filename if renamed else image['filename'],
                'suggested_filename': result.get('suggested_filename', '')
            })
        else:
            return jsonify({'error': 'Analysis failed - AI returned no result'}), 500

    except Exception as e:
        print(f"Error analyzing image {image_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500
    finally:
        # Clean up temporary video frame if created
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                print(f"[ANALYZE] Cleaned up temporary frame file: {temp_image_path}")
            except Exception as e:
                print(f"[ANALYZE] Warning: Could not delete temp file {temp_image_path}: {e}")


@images_bp.route('/api/analyze-batch', methods=['POST'])
def batch_analyze():
    """Analyze all unanalyzed images with auto-rename"""
    limit = request.args.get('limit', 10, type=int)

    # Check AI connection
    connected, message = ai.check_connection()
    if not connected:
        return jsonify({'error': f'AI not available: {message}'}), 503

    # Get unanalyzed images
    images = db.get_unanalyzed_images(limit=limit)

    if not images:
        return jsonify({
            'success': True,
            'message': 'No unanalyzed images',
            'analyzed': 0
        })

    analyzed_count = 0
    failed_count = 0
    renamed_count = 0

    for image in images:
        filepath = get_full_filepath(image['filepath'], PHOTOS_DIR)
        image_id = image['id']

        if not os.path.exists(filepath):
            failed_count += 1
            continue

        result = ai.analyze_image(filepath)

        if result:
            # Update analysis
            db.update_image_analysis(
                image_id,
                result['description'],
                result['tags']
            )
            analyzed_count += 1

            # Auto-rename if AI suggested a filename
            if result.get('suggested_filename'):
                suggested = result['suggested_filename'].strip()

                if suggested and len(suggested) > 0:
                    # Sanitize filename
                    suggested = secure_filename(suggested)

                    # Get original extension
                    old_ext = Path(filepath).suffix

                    # Build new filename
                    new_filename = f"{suggested}{old_ext}"

                    # Get directory
                    directory = os.path.dirname(filepath)
                    new_filepath = os.path.join(directory, new_filename)

                    # Check if different from current name
                    if new_filepath != filepath:
                        # Check if target exists
                        if not os.path.exists(new_filepath):
                            try:
                                # Rename file on disk
                                os.rename(filepath, new_filepath)

                                # Update database
                                db.rename_image(image_id, new_filepath, new_filename)

                                renamed_count += 1
                                print(f"Batch auto-renamed: {image['filename']} → {new_filename}")
                            except Exception as e:
                                print(f"Batch auto-rename failed for {image['filename']}: {e}")
                        else:
                            # File exists, add counter
                            counter = 1
                            base_name = suggested
                            while os.path.exists(new_filepath) and counter < 100:
                                new_filename = f"{base_name}_{counter}{old_ext}"
                                new_filepath = os.path.join(directory, new_filename)
                                counter += 1

                            if not os.path.exists(new_filepath):
                                try:
                                    os.rename(filepath, new_filepath)
                                    db.rename_image(image_id, new_filepath, new_filename)
                                    renamed_count += 1
                                    print(f"Batch auto-renamed: {image['filename']} → {new_filename}")
                                except Exception as e:
                                    print(f"Batch auto-rename failed for {image['filename']}: {e}")
        else:
            failed_count += 1

    return jsonify({
        'success': True,
        'total': len(images),
        'analyzed': analyzed_count,
        'renamed': renamed_count,
        'failed': failed_count
    })


# ============ SEARCH & DISCOVERY ============

@images_bp.route('/api/images/<int:image_id>/similar', methods=['GET'])
def get_similar_images(image_id):
    """Get similar images based on shared tags"""
    try:
        limit = int(request.args.get('limit', 6))
        similar = db.get_similar_images(image_id, limit)

        return jsonify({
            'image_id': image_id,
            'similar': similar,
            'count': len(similar)
        })

    except Exception as e:
        print(f"Error finding similar images for {image_id}: {str(e)}")
        return jsonify({'error': f'Failed to find similar images: {str(e)}'}), 500


@images_bp.route('/api/images/search', methods=['GET'])
def search_images():
    """Search images by query"""
    query = request.args.get('q', '').strip()

    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400

    results = db.search_images(query)

    return jsonify({
        'query': query,
        'results': results,
        'count': len(results)
    })


@images_bp.route('/api/images/<int:image_id>/reverse-search', methods=['GET'])
def get_reverse_search_options(image_id):
    """Get reverse image search options for an image"""
    # Get image info
    image = db.get_image(image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404

    # Construct image URL (accessible from client)
    image_url = f'/api/images/{image_id}/file'

    # Get all search options
    search_options = ReverseImageSearch.get_all_search_options(image_id, image_url)

    # Get helpful tips
    copyright_tips = get_copyright_tips()
    usage_tips = get_usage_detection_tips()

    # Get search guide
    search_guide = ReverseImageSearch.create_search_guide()

    return jsonify({
        'success': True,
        'image_id': image_id,
        'image_filename': image['filename'],
        'search_options': search_options,
        'copyright_tips': copyright_tips,
        'usage_detection_tips': usage_tips,
        'search_guide': search_guide
    })


# ============ TAG MANAGEMENT ============

@images_bp.route('/api/tags', methods=['GET'])
def get_tags():
    """Get all tags with usage statistics"""
    try:
        tags = db.get_all_tags()
        return jsonify({
            'tags': tags,
            'count': len(tags)
        })
    except Exception as e:
        print(f"Error getting tags: {str(e)}")
        return jsonify({'error': f'Failed to get tags: {str(e)}'}), 500


@images_bp.route('/api/tags/suggestions', methods=['GET'])
def get_tag_suggestions():
    """Get tag suggestions for autocomplete"""
    try:
        prefix = request.args.get('prefix', '')
        limit = int(request.args.get('limit', 10))

        suggestions = db.get_tag_suggestions(prefix, limit)
        return jsonify({
            'suggestions': suggestions,
            'count': len(suggestions)
        })
    except Exception as e:
        print(f"Error getting tag suggestions: {str(e)}")
        return jsonify({'error': f'Failed to get tag suggestions: {str(e)}'}), 500


@images_bp.route('/api/tags/<tag>/related', methods=['GET'])
def get_related_tags(tag):
    """Get tags that frequently appear with the given tag"""
    try:
        limit = int(request.args.get('limit', 10))
        related = db.get_related_tags(tag, limit)
        return jsonify({
            'tag': tag,
            'related': related,
            'count': len(related)
        })
    except Exception as e:
        print(f"Error getting related tags: {str(e)}")
        return jsonify({'error': f'Failed to get related tags: {str(e)}'}), 500


# ============ UPLOAD & SCAN ============

@images_bp.route('/api/scan', methods=['POST'])
def scan_directory():
    """Scan photos directory for new images and videos"""
    if not os.path.exists(PHOTOS_DIR):
        return jsonify({'error': f'Photos directory not found: {PHOTOS_DIR}'}), 404

    found_media = []
    new_media = []
    skipped = 0

    # Walk through directory
    for root, dirs, files in os.walk(PHOTOS_DIR):
        for filename in files:
            ext = Path(filename).suffix.lower()

            if ext in ALL_MEDIA_FORMATS:
                full_filepath = os.path.join(root, filename)
                found_media.append(full_filepath)

                try:
                    # Store relative path from PHOTOS_DIR
                    abs_photos_dir = os.path.abspath(PHOTOS_DIR)
                    abs_filepath = os.path.abspath(full_filepath)
                    relative_path = os.path.relpath(abs_filepath, abs_photos_dir)

                    file_size = os.path.getsize(full_filepath)
                    width = None
                    height = None
                    media_type = 'video' if ext in VIDEO_FORMATS else 'image'

                    # Get dimensions for images only
                    if media_type == 'image':
                        img = Image.open(full_filepath)
                        width, height = img.size
                        img.close()

                    # Try to add to database with relative path
                    image_id = db.add_image(
                        filepath=relative_path,
                        filename=filename,
                        width=width,
                        height=height,
                        file_size=file_size,
                        media_type=media_type
                    )

                    if image_id:
                        new_media.append({
                            'id': image_id,
                            'filename': filename,
                            'filepath': full_filepath,
                            'media_type': media_type
                        })
                    else:
                        skipped += 1

                except Exception as e:
                    print(f"Error processing {full_filepath}: {e}")
                    skipped += 1

    return jsonify({
        'success': True,
        'found': len(found_media),
        'new': len(new_media),
        'skipped': skipped,
        'images': new_media
    })


@images_bp.route('/api/upload', methods=['POST'])
def upload_image():
    """Upload image or video file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALL_MEDIA_FORMATS:
        return jsonify({'error': f'Unsupported format: {ext}'}), 400

    try:
        # Sanitize filename
        filename = secure_filename(file.filename)

        # Ensure photos directory exists
        os.makedirs(PHOTOS_DIR, exist_ok=True)

        # Save file
        filepath = os.path.join(PHOTOS_DIR, filename)

        # Handle duplicates
        counter = 1
        base_name = Path(filename).stem
        while os.path.exists(filepath):
            filename = f"{base_name}_{counter}{ext}"
            filepath = os.path.join(PHOTOS_DIR, filename)
            counter += 1

        file.save(filepath)

        # Get file info
        file_size = os.path.getsize(filepath)
        width = None
        height = None
        media_type = 'video' if ext in VIDEO_FORMATS else 'image'

        # Get dimensions for images only
        if media_type == 'image':
            img = Image.open(filepath)
            width, height = img.size
            img.close()

        # Add to database with just filename (relative to PHOTOS_DIR)
        image_id = db.add_image(
            filepath=filename,
            filename=filename,
            width=width,
            height=height,
            file_size=file_size,
            media_type=media_type
        )

        return jsonify({
            'success': True,
            'image_id': image_id,
            'filename': filename,
            'filepath': filepath,
            'media_type': media_type
        })

    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500
