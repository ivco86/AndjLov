"""
Export routes - CSV, JSON, PDF exports for images and boards
"""

import os
import tempfile
from flask import Blueprint, jsonify, request, send_file, after_this_request
from reportlab.lib.pagesizes import A4, letter

from shared import db, PHOTOS_DIR, DATA_DIR
from export_utils import MetadataExporter
from pdf_catalog import PDFCatalogGenerator

export_bp = Blueprint('export', __name__)


# ============ IMAGE EXPORTS ============

@export_bp.route('/api/export/images/csv', methods=['POST'])
def export_images_csv():
    """Export images metadata to CSV"""
    data = request.get_json() or {}
    image_ids = data.get('image_ids', [])

    if not image_ids:
        return jsonify({'error': 'No image IDs provided'}), 400

    # Get images
    images = []
    for image_id in image_ids:
        image = db.get_image(image_id)
        if image:
            images.append(image)

    if not images:
        return jsonify({'error': 'No valid images found'}), 404

    # Generate CSV
    csv_content = MetadataExporter.to_csv(images)

    # Return as downloadable file
    return csv_content, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': f'attachment; filename="image_metadata.csv"'
    }


@export_bp.route('/api/export/images/json', methods=['POST'])
def export_images_json():
    """Export images metadata to JSON"""
    data = request.get_json() or {}
    image_ids = data.get('image_ids', [])
    include_summary = data.get('include_summary', True)

    if not image_ids:
        return jsonify({'error': 'No image IDs provided'}), 400

    # Get images
    images = []
    for image_id in image_ids:
        image = db.get_image(image_id)
        if image:
            images.append(image)

    if not images:
        return jsonify({'error': 'No valid images found'}), 404

    # Generate JSON
    if include_summary:
        json_content = MetadataExporter.to_json_with_summary(images)
    else:
        json_content = MetadataExporter.to_json(images)

    # Return as downloadable file
    return json_content, 200, {
        'Content-Type': 'application/json',
        'Content-Disposition': f'attachment; filename="image_metadata.json"'
    }


@export_bp.route('/api/export/images/pdf', methods=['POST'])
def export_images_pdf():
    """Generate PDF catalog from selected images"""
    data = request.get_json() or {}
    image_ids = data.get('image_ids', [])
    title = data.get('title', 'Image Catalog')
    subtitle = data.get('subtitle', None)
    page_size = data.get('page_size', 'A4')
    orientation = data.get('orientation', 'portrait')

    if not image_ids:
        return jsonify({'error': 'No image IDs provided'}), 400

    # Get images
    images = []
    for image_id in image_ids:
        image = db.get_image(image_id)
        if image:
            images.append(image)

    if not images:
        return jsonify({'error': 'No valid images found'}), 404

    # Map page size string to reportlab constant
    page_size_map = {
        'A4': A4,
        'letter': letter
    }
    page_size_obj = page_size_map.get(page_size, A4)

    # Create temporary PDF file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_path = temp_file.name
    temp_file.close()

    try:
        # Generate PDF
        generator = PDFCatalogGenerator(page_size=page_size_obj, orientation=orientation)

        # Get thumbnail directory
        thumbnail_dir = os.path.join(DATA_DIR, 'thumbnails')

        generator.generate_catalog(
            images=images,
            output_path=temp_path,
            title=title,
            subtitle=subtitle,
            data_dir=PHOTOS_DIR,
            thumbnail_dir=thumbnail_dir
        )

        # Send file
        filename = f"{title.replace(' ', '_')}.pdf"

        @after_this_request
        def remove_file(response):
            try:
                os.remove(temp_path)
            except Exception:
                pass
            return response

        return send_file(
            temp_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        # Clean up temp file on error
        try:
            os.remove(temp_path)
        except:
            pass
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500


# ============ BOARD EXPORTS ============

@export_bp.route('/api/export/boards/<int:board_id>/csv', methods=['GET'])
def export_board_csv(board_id):
    """Export board images metadata to CSV"""
    # Get board info
    board = db.get_board(board_id)
    if not board:
        return jsonify({'error': 'Board not found'}), 404

    # Get board images
    images = db.get_board_images(board_id)

    if not images:
        return jsonify({'error': 'No images in board'}), 404

    # Generate CSV
    csv_content = MetadataExporter.to_csv(images)

    # Return as downloadable file
    board_name = board['name'].replace(' ', '_')
    return csv_content, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': f'attachment; filename="{board_name}_metadata.csv"'
    }


@export_bp.route('/api/export/boards/<int:board_id>/json', methods=['GET'])
def export_board_json(board_id):
    """Export board images metadata to JSON"""
    # Get board info
    board = db.get_board(board_id)
    if not board:
        return jsonify({'error': 'Board not found'}), 404

    # Get board images
    images = db.get_board_images(board_id)

    if not images:
        return jsonify({'error': 'No images in board'}), 404

    # Generate JSON with summary
    json_content = MetadataExporter.to_json_with_summary(images, board_info=board)

    # Return as downloadable file
    board_name = board['name'].replace(' ', '_')
    return json_content, 200, {
        'Content-Type': 'application/json',
        'Content-Disposition': f'attachment; filename="{board_name}_metadata.json"'
    }


@export_bp.route('/api/export/boards/<int:board_id>/pdf', methods=['POST'])
def export_board_pdf(board_id):
    """Generate PDF catalog for a board"""
    # Get board info
    board = db.get_board(board_id)
    if not board:
        return jsonify({'error': 'Board not found'}), 404

    # Get board images
    images = db.get_board_images(board_id)

    if not images:
        return jsonify({'error': 'No images in board'}), 404

    # Get request options
    data = request.get_json() or {}
    page_size = data.get('page_size', 'A4')  # A4 or letter
    orientation = data.get('orientation', 'portrait')  # portrait or landscape

    # Map page size string to reportlab constant
    page_size_map = {
        'A4': A4,
        'letter': letter
    }
    page_size_obj = page_size_map.get(page_size, A4)

    # Create temporary PDF file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_path = temp_file.name
    temp_file.close()

    try:
        # Generate PDF
        generator = PDFCatalogGenerator(page_size=page_size_obj, orientation=orientation)

        # Get thumbnail directory
        thumbnail_dir = os.path.join(DATA_DIR, 'thumbnails')

        generator.generate_board_catalog(
            board_info=board,
            images=images,
            output_path=temp_path,
            data_dir=PHOTOS_DIR,
            thumbnail_dir=thumbnail_dir
        )

        # Send file
        board_name = board['name'].replace(' ', '_')
        filename = f"{board_name}_catalog.pdf"

        @after_this_request
        def remove_file(response):
            try:
                os.remove(temp_path)
            except Exception:
                pass
            return response

        return send_file(
            temp_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        # Clean up temp file on error
        try:
            os.remove(temp_path)
        except:
            pass
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500
