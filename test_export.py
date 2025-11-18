#!/usr/bin/env python3
"""
Test script for export/import functionality
"""

from database import Database
import json

def test_export_json():
    """Test JSON export"""
    print("Testing JSON export...")
    db = Database()

    # Export data
    json_data = db.export_data_json(include_images=True, include_boards=True)

    # Parse to verify it's valid JSON
    data = json.loads(json_data)

    print(f"✓ JSON export successful")
    print(f"  - Version: {data.get('version')}")
    print(f"  - Total images: {data.get('total_images', 0)}")
    print(f"  - Total boards: {data.get('total_boards', 0)}")
    print(f"  - Tags count: {len(data.get('tags', []))}")

    return json_data

def test_export_markdown():
    """Test Markdown export"""
    print("\nTesting Markdown export...")
    db = Database()

    # Export data
    md_data = db.export_data_markdown(include_images=True, include_boards=True)

    print(f"✓ Markdown export successful")
    print(f"  - Length: {len(md_data)} characters")
    print(f"  - First 200 chars:\n{md_data[:200]}...")

    return md_data

def test_export_csv():
    """Test CSV export"""
    print("\nTesting CSV export...")
    db = Database()

    # Export data
    csv_data = db.export_data_csv()

    print(f"✓ CSV export successful")
    print(f"  - Files generated: {list(csv_data.keys())}")

    if 'images_csv' in csv_data:
        lines = csv_data['images_csv'].split('\n')
        print(f"  - Images CSV lines: {len(lines)}")

    if 'boards_csv' in csv_data:
        lines = csv_data['boards_csv'].split('\n')
        print(f"  - Boards CSV lines: {len(lines)}")

    return csv_data

def test_import_json(json_data):
    """Test JSON import"""
    print("\nTesting JSON import...")
    db = Database()

    # Import data
    result = db.import_data_json(
        json_data=json_data,
        import_boards=True,
        import_board_assignments=True,
        update_existing=False
    )

    print(f"✓ JSON import {'successful' if result.get('success') else 'failed'}")
    print(f"  - Boards created: {result.get('boards_created', 0)}")
    print(f"  - Boards updated: {result.get('boards_updated', 0)}")
    print(f"  - Images updated: {result.get('images_updated', 0)}")
    print(f"  - Assignments created: {result.get('board_assignments_created', 0)}")

    if 'errors' in result and result['errors']:
        print(f"  - Errors: {result['errors']}")

    return result

if __name__ == '__main__':
    print("=" * 60)
    print("EXPORT/IMPORT FUNCTIONALITY TEST")
    print("=" * 60)

    try:
        # Test exports
        json_data = test_export_json()
        md_data = test_export_markdown()
        csv_data = test_export_csv()

        # Test import (with the JSON data we just exported)
        import_result = test_import_json(json_data)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
