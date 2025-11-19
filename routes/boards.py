"""
Board routes - board CRUD, hierarchy, image assignments
"""

from flask import Blueprint, jsonify, request
from shared import db

boards_bp = Blueprint('boards', __name__)


@boards_bp.route('/api/boards', methods=['GET', 'POST'])
def boards():
    """Get all boards or create new board"""
    if request.method == 'GET':
        all_boards = db.get_all_boards()

        # Add image count for each board
        conn = db.get_connection()
        cursor = conn.cursor()
        for board in all_boards:
            cursor.execute(
                'SELECT COUNT(*) as count FROM board_images WHERE board_id = ?',
                (board['id'],)
            )
            result = cursor.fetchone()
            board['image_count'] = result['count'] if result else 0
        conn.close()

        # Organize into hierarchy
        top_level = []
        boards_map = {board['id']: board for board in all_boards}

        for board in all_boards:
            board['sub_boards'] = []

        for board in all_boards:
            if board['parent_id'] is None:
                top_level.append(board)
            else:
                parent = boards_map.get(board['parent_id'])
                if parent:
                    parent['sub_boards'].append(board)

        return jsonify({
            'boards': top_level,
            'total': len(all_boards)
        })

    elif request.method == 'POST':
        data = request.json
        name = data.get('name')
        description = data.get('description')
        parent_id = data.get('parent_id')

        if not name:
            return jsonify({'error': 'Board name is required'}), 400

        board_id = db.create_board(name, description, parent_id)

        return jsonify({
            'success': True,
            'board_id': board_id,
            'name': name
        }), 201


@boards_bp.route('/api/boards/<int:board_id>', methods=['GET', 'PUT', 'DELETE'])
def board_detail(board_id):
    """Get, update, or delete board"""
    if request.method == 'GET':
        board = db.get_board(board_id)

        if not board:
            return jsonify({'error': 'Board not found'}), 404

        # Get sub-boards
        board['sub_boards'] = db.get_sub_boards(board_id)

        # Get images in board
        board['images'] = db.get_board_images(board_id)

        return jsonify(board)

    elif request.method == 'PUT':
        data = request.json
        name = data.get('name')
        description = data.get('description')
        parent_id = data.get('parent_id')

        # Update basic info if provided
        if name is not None or description is not None:
            db.update_board(board_id, name, description)

        # Move board if parent_id is explicitly provided (including None for top-level)
        if 'parent_id' in data:
            try:
                db.move_board(board_id, parent_id)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400

        return jsonify({
            'success': True,
            'board_id': board_id
        })

    elif request.method == 'DELETE':
        # Read from query parameters
        delete_sub_boards = request.args.get('delete_sub_boards', 'false').lower() == 'true'

        db.delete_board(board_id, delete_sub_boards=delete_sub_boards)

        return jsonify({
            'success': True,
            'board_id': board_id,
            'deleted_sub_boards': delete_sub_boards
        })


@boards_bp.route('/api/boards/<int:board_id>/merge', methods=['POST'])
def merge_board(board_id):
    """Merge this board into another board"""
    data = request.json
    target_board_id = data.get('target_board_id')
    delete_source = data.get('delete_source', True)

    if not target_board_id:
        return jsonify({'error': 'target_board_id is required'}), 400

    if board_id == target_board_id:
        return jsonify({'error': 'Cannot merge board into itself'}), 400

    try:
        moved_count = db.merge_boards(board_id, target_board_id, delete_source)

        return jsonify({
            'success': True,
            'source_board_id': board_id,
            'target_board_id': target_board_id,
            'images_moved': moved_count,
            'source_deleted': delete_source
        })
    except Exception as e:
        print(f"Error merging boards: {e}")
        return jsonify({'error': f'Failed to merge boards: {str(e)}'}), 500


@boards_bp.route('/api/boards/<int:board_id>/images', methods=['POST', 'DELETE'])
def board_images(board_id):
    """Add or remove image from board"""
    data = request.json
    image_id = data.get('image_id')

    if not image_id:
        return jsonify({'error': 'image_id is required'}), 400

    if request.method == 'POST':
        db.add_image_to_board(board_id, image_id)

        return jsonify({
            'success': True,
            'board_id': board_id,
            'image_id': image_id,
            'action': 'added'
        })

    elif request.method == 'DELETE':
        db.remove_image_from_board(board_id, image_id)

        return jsonify({
            'success': True,
            'board_id': board_id,
            'image_id': image_id,
            'action': 'removed'
        })
