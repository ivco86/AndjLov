"""
AI routes - EXIF metadata, semantic search with CLIP embeddings
"""

import os
from flask import Blueprint, jsonify, request

from shared import db, PHOTOS_DIR
from utils import get_full_filepath
from exif_utils import extract_exif_data, get_camera_list_from_exif_data, format_exif_for_display
from embeddings_utils import (
    is_clip_available,
    generate_embedding_for_image,
    search_by_text_query,
    find_similar_images,
    embedding_to_blob,
    blob_to_embedding
)

ai_bp = Blueprint('ai', __name__)


# ============ EXIF METADATA ENDPOINTS ============

@ai_bp.route('/api/images/<int:image_id>/exif', methods=['GET'])
def get_image_exif(image_id):
    """Get EXIF metadata for an image"""
    # Check if image exists
    image = db.get_image(image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404

    # Get EXIF data from database
    exif_data = db.get_exif_data(image_id)

    if not exif_data:
        return jsonify({
            'image_id': image_id,
            'has_exif': False,
            'message': 'No EXIF data available. Try extracting it first.'
        })

    # Format for display
    formatted = format_exif_for_display(exif_data)

    return jsonify({
        'image_id': image_id,
        'has_exif': True,
        'exif': exif_data,
        'formatted': formatted
    })


@ai_bp.route('/api/images/<int:image_id>/exif/extract', methods=['POST'])
def extract_image_exif(image_id):
    """Extract EXIF metadata from an image file and save to database"""
    # Check if image exists
    image = db.get_image(image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404

    # Only extract from images, not videos
    if image.get('media_type') == 'video':
        return jsonify({'error': 'EXIF extraction not supported for videos'}), 400

    # Get file path
    filepath = get_full_filepath(image['filepath'], PHOTOS_DIR)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Image file not found on disk'}), 404

    # Extract EXIF
    exif_data = extract_exif_data(filepath)

    if not exif_data:
        return jsonify({
            'success': False,
            'message': 'No EXIF data found in image'
        })

    # Save to database
    success = db.save_exif_data(image_id, exif_data)

    if success:
        # Format for display
        formatted = format_exif_for_display(exif_data)

        return jsonify({
            'success': True,
            'image_id': image_id,
            'exif': exif_data,
            'formatted': formatted
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Failed to save EXIF data'
        }), 500


@ai_bp.route('/api/images/search/exif', methods=['POST'])
def search_images_by_exif():
    """Search images by EXIF criteria"""
    data = request.get_json() or {}

    # Extract search criteria
    camera_make = data.get('camera_make')
    camera_model = data.get('camera_model')
    min_iso = data.get('min_iso')
    max_iso = data.get('max_iso')
    min_aperture = data.get('min_aperture')
    max_aperture = data.get('max_aperture')
    min_focal_length = data.get('min_focal_length')
    max_focal_length = data.get('max_focal_length')
    has_gps = data.get('has_gps')
    limit = data.get('limit', 100)

    # Search
    results = db.search_by_exif(
        camera_make=camera_make,
        camera_model=camera_model,
        min_iso=min_iso,
        max_iso=max_iso,
        min_aperture=min_aperture,
        max_aperture=max_aperture,
        min_focal_length=min_focal_length,
        max_focal_length=max_focal_length,
        has_gps=has_gps,
        limit=limit
    )

    return jsonify({
        'results': results,
        'count': len(results),
        'criteria': data
    })


@ai_bp.route('/api/exif/cameras', methods=['GET'])
def get_cameras():
    """Get list of all cameras found in EXIF data"""
    cameras = db.get_all_cameras()

    return jsonify({
        'cameras': cameras,
        'count': len(cameras)
    })


# ============ SEMANTIC SEARCH ENDPOINTS ============

@ai_bp.route('/api/embeddings/status', methods=['GET'])
def embeddings_status():
    """Check if CLIP model is available and get embeddings statistics"""
    clip_available = is_clip_available()

    total_images = db.get_stats()['total_images']
    embeddings_count = db.count_embeddings()

    return jsonify({
        'clip_available': clip_available,
        'total_images': total_images,
        'embeddings_count': embeddings_count,
        'coverage_percent': round((embeddings_count / total_images * 100) if total_images > 0 else 0, 1),
        'message': 'CLIP model ready' if clip_available else 'Install transformers and torch to enable semantic search'
    })


@ai_bp.route('/api/embeddings/generate', methods=['POST'])
def generate_embeddings():
    """Generate CLIP embeddings for all images without embeddings"""
    if not is_clip_available():
        return jsonify({
            'error': 'CLIP not available. Install transformers and torch first.',
            'install_command': 'pip install transformers torch'
        }), 503

    # Get images without embeddings
    limit = request.args.get('limit', 100, type=int)
    images = db.get_images_without_embeddings(limit=limit)

    if not images:
        return jsonify({
            'success': True,
            'message': 'All images already have embeddings',
            'generated': 0
        })

    # Generate embeddings
    generated = 0
    failed = 0

    for image in images:
        filepath = get_full_filepath(image['filepath'], PHOTOS_DIR)

        if not os.path.exists(filepath):
            failed += 1
            continue

        # Generate embedding
        embedding = generate_embedding_for_image(filepath)

        if embedding is not None:
            # Convert to blob and save
            embedding_blob = embedding_to_blob(embedding)
            success = db.save_embedding(image['id'], embedding_blob)

            if success:
                generated += 1
                print(f"Generated embedding for image {image['id']}: {image['filename']}")
            else:
                failed += 1
        else:
            failed += 1

    return jsonify({
        'success': True,
        'total_processed': len(images),
        'generated': generated,
        'failed': failed,
        'message': f'Generated {generated} embeddings'
    })


@ai_bp.route('/api/search/semantic', methods=['POST'])
def semantic_search():
    """Search images by natural language query using CLIP"""
    if not is_clip_available():
        return jsonify({
            'error': 'CLIP not available. Install transformers and torch first.',
            'install_command': 'pip install transformers torch'
        }), 503

    data = request.get_json() or {}
    query = data.get('query', '').strip()
    top_k = data.get('top_k', 20)

    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    # Get all embeddings
    all_embeddings = db.get_all_embeddings()

    if not all_embeddings:
        return jsonify({
            'error': 'No embeddings found. Generate embeddings first.',
            'results': [],
            'count': 0
        }), 404

    # Convert blobs to numpy arrays
    for item in all_embeddings:
        item['embedding'] = blob_to_embedding(item['embedding'])

    # Search
    results = search_by_text_query(query, all_embeddings)

    if results is None:
        return jsonify({'error': 'Semantic search failed'}), 500

    # Get top K results
    top_results = results[:top_k]

    # Fetch image details
    images = []
    for result in top_results:
        image = db.get_image(result['image_id'])
        if image:
            images.append({
                **image,
                'similarity': round(result['similarity'], 3)
            })

    return jsonify({
        'query': query,
        'results': images,
        'count': len(images)
    })


@ai_bp.route('/api/images/<int:image_id>/similar', methods=['GET'])
def get_similar_images_by_embedding(image_id):
    """Find visually similar images using CLIP embeddings"""
    if not is_clip_available():
        return jsonify({
            'error': 'CLIP not available. Install transformers and torch first.',
            'install_command': 'pip install transformers torch'
        }), 503

    # Check if image exists
    image = db.get_image(image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404

    # Get embedding for this image
    query_embedding_blob = db.get_embedding(image_id)

    if not query_embedding_blob:
        return jsonify({
            'error': 'Image does not have an embedding. Generate it first.',
            'image_id': image_id
        }), 404

    # Convert to numpy array
    query_embedding = blob_to_embedding(query_embedding_blob)

    # Get all embeddings
    all_embeddings = db.get_all_embeddings()

    # Convert blobs to numpy arrays
    for item in all_embeddings:
        item['embedding'] = blob_to_embedding(item['embedding'])

    # Find similar
    top_k = request.args.get('top_k', 20, type=int)
    similar = find_similar_images(query_embedding, all_embeddings, top_k=top_k, exclude_id=image_id)

    # Fetch image details
    images = []
    for result in similar:
        sim_image = db.get_image(result['image_id'])
        if sim_image:
            images.append({
                **sim_image,
                'similarity': round(result['similarity'], 3)
            })

    return jsonify({
        'image_id': image_id,
        'similar': images,
        'count': len(images)
    })
