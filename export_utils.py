"""
Export utilities for AI Gallery
Handles CSV and JSON export of image metadata
"""

import csv
import json
from datetime import datetime
from io import StringIO
import os


class MetadataExporter:
    """Export image metadata to various formats"""

    @staticmethod
    def prepare_metadata(images, include_fields=None):
        """
        Prepare metadata for export

        Args:
            images: List of image dictionaries
            include_fields: Optional list of fields to include (None = all fields)

        Returns:
            list: List of prepared metadata dictionaries
        """
        if not include_fields:
            # Default fields to export
            include_fields = [
                'id', 'filename', 'filepath', 'description', 'tags',
                'width', 'height', 'file_size', 'media_type',
                'is_favorite', 'created_at', 'updated_at', 'analyzed_at'
            ]

        prepared = []
        for image in images:
            row = {}
            for field in include_fields:
                value = image.get(field, '')
                # Convert None to empty string
                if value is None:
                    value = ''
                # Convert boolean to string
                elif isinstance(value, bool):
                    value = 'Yes' if value else 'No'
                # Format file size
                elif field == 'file_size' and isinstance(value, (int, float)) and value > 0:
                    if value >= 1024 * 1024:
                        value = f"{value / (1024 * 1024):.2f} MB"
                    elif value >= 1024:
                        value = f"{value / 1024:.2f} KB"
                    else:
                        value = f"{value} bytes"
                # Format dates
                elif field in ['created_at', 'updated_at', 'analyzed_at'] and value:
                    # Keep ISO format but make it more readable
                    try:
                        if 'T' in str(value):
                            value = str(value).replace('T', ' ').split('.')[0]
                    except:
                        pass

                row[field] = value
            prepared.append(row)

        return prepared

    @staticmethod
    def to_csv(images, include_fields=None, include_header=True):
        """
        Export metadata to CSV format

        Args:
            images: List of image dictionaries
            include_fields: Optional list of fields to include
            include_header: Whether to include header row

        Returns:
            str: CSV formatted string
        """
        if not images:
            return ""

        prepared = MetadataExporter.prepare_metadata(images, include_fields)

        # Use StringIO to build CSV in memory
        output = StringIO()

        # Get fieldnames from first prepared row
        fieldnames = list(prepared[0].keys())

        writer = csv.DictWriter(output, fieldnames=fieldnames)

        if include_header:
            writer.writeheader()

        writer.writerows(prepared)

        return output.getvalue()

    @staticmethod
    def to_json(images, include_fields=None, pretty=True):
        """
        Export metadata to JSON format

        Args:
            images: List of image dictionaries
            include_fields: Optional list of fields to include
            pretty: Whether to format JSON with indentation

        Returns:
            str: JSON formatted string
        """
        if not images:
            return "[]"

        prepared = MetadataExporter.prepare_metadata(images, include_fields)

        if pretty:
            return json.dumps(prepared, indent=2, ensure_ascii=False)
        else:
            return json.dumps(prepared, ensure_ascii=False)

    @staticmethod
    def to_json_with_summary(images, board_info=None, include_fields=None):
        """
        Export metadata to JSON with summary information

        Args:
            images: List of image dictionaries
            board_info: Optional board information
            include_fields: Optional list of fields to include

        Returns:
            str: JSON formatted string with summary
        """
        if not images:
            return json.dumps({
                "summary": {"total_images": 0},
                "images": []
            }, indent=2)

        prepared = MetadataExporter.prepare_metadata(images, include_fields)

        # Calculate summary statistics
        total_size = sum(img.get('file_size', 0) for img in images if img.get('file_size'))
        analyzed_count = sum(1 for img in images if img.get('analyzed_at'))
        favorite_count = sum(1 for img in images if img.get('is_favorite'))

        # Collect all unique tags
        all_tags = set()
        for img in images:
            tags = img.get('tags', '')
            if tags:
                all_tags.update(tag.strip() for tag in tags.split(',') if tag.strip())

        summary = {
            "total_images": len(images),
            "analyzed_images": analyzed_count,
            "favorite_images": favorite_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2) if total_size else 0,
            "unique_tags": len(all_tags),
            "exported_at": datetime.now().isoformat()
        }

        if board_info:
            summary["board"] = {
                "name": board_info.get('name', ''),
                "description": board_info.get('description', '')
            }

        result = {
            "summary": summary,
            "images": prepared
        }

        return json.dumps(result, indent=2, ensure_ascii=False)

    @staticmethod
    def save_to_file(content, output_path, format='csv'):
        """
        Save exported content to file

        Args:
            content: String content to save
            output_path: Path where file should be saved
            format: File format ('csv' or 'json')

        Returns:
            str: Path to saved file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Write content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return output_path


class BoardExporter:
    """Export entire boards with hierarchy"""

    @staticmethod
    def export_board_structure(boards, images_by_board=None):
        """
        Export board structure with optional image counts

        Args:
            boards: List of board dictionaries
            images_by_board: Optional dict mapping board_id to list of images

        Returns:
            dict: Hierarchical board structure
        """
        def build_hierarchy(parent_id=None):
            result = []
            for board in boards:
                if board.get('parent_id') == parent_id:
                    board_data = {
                        'id': board['id'],
                        'name': board['name'],
                        'description': board.get('description', ''),
                        'created_at': board.get('created_at', '')
                    }

                    # Add image count if provided
                    if images_by_board:
                        board_images = images_by_board.get(board['id'], [])
                        board_data['image_count'] = len(board_images)

                    # Recursively add children
                    children = build_hierarchy(board['id'])
                    if children:
                        board_data['sub_boards'] = children

                    result.append(board_data)

            return result

        return {
            'boards': build_hierarchy(),
            'total_boards': len(boards),
            'exported_at': datetime.now().isoformat()
        }

    @staticmethod
    def export_full_catalog(boards, images_by_board):
        """
        Export complete catalog with all boards and their images

        Args:
            boards: List of board dictionaries
            images_by_board: Dict mapping board_id to list of images

        Returns:
            dict: Complete catalog structure
        """
        catalog = {
            'catalog_info': {
                'total_boards': len(boards),
                'total_images': sum(len(imgs) for imgs in images_by_board.values()),
                'exported_at': datetime.now().isoformat()
            },
            'boards': []
        }

        for board in boards:
            board_id = board['id']
            board_images = images_by_board.get(board_id, [])

            board_data = {
                'id': board_id,
                'name': board['name'],
                'description': board.get('description', ''),
                'parent_id': board.get('parent_id'),
                'image_count': len(board_images),
                'images': MetadataExporter.prepare_metadata(board_images)
            }

            catalog['boards'].append(board_data)

        return catalog
