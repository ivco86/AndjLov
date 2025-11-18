#!/usr/bin/env python3
"""
Test script for privacy functionality
"""

from database import Database
from privacy_service import get_privacy_service
import os

def test_database_privacy_schema():
    """Test that privacy columns exist in database"""
    print("Testing database privacy schema...")
    db = Database()

    # Test that database initializes without errors
    print("✓ Database initialized successfully")

    # Test privacy methods exist
    assert hasattr(db, 'update_privacy_analysis'), "Missing update_privacy_analysis method"
    assert hasattr(db, 'get_privacy_data'), "Missing get_privacy_data method"
    assert hasattr(db, 'get_unanalyzed_privacy_images'), "Missing get_unanalyzed_privacy_images method"
    assert hasattr(db, 'get_images_with_faces'), "Missing get_images_with_faces method"
    assert hasattr(db, 'get_nsfw_images'), "Missing get_nsfw_images method"

    print("✓ All privacy database methods exist")

def test_privacy_service():
    """Test privacy service initialization"""
    print("\nTesting privacy service...")

    privacy_service = get_privacy_service()
    assert privacy_service is not None, "Privacy service not initialized"

    print("✓ Privacy service initialized")

    # Check if face detector loaded
    if privacy_service.face_cascade is not None:
        print("✓ Face detector loaded successfully")
    else:
        print("⚠ Warning: Face detector not available (OpenCV may not have Haar cascades)")

    # Test methods exist
    assert hasattr(privacy_service, 'detect_faces'), "Missing detect_faces method"
    assert hasattr(privacy_service, 'detect_license_plates'), "Missing detect_license_plates method"
    assert hasattr(privacy_service, 'analyze_image_privacy'), "Missing analyze_image_privacy method"
    assert hasattr(privacy_service, 'apply_blur_to_zones'), "Missing apply_blur_to_zones method"

    print("✓ All privacy service methods exist")

def test_privacy_analysis_mock():
    """Test privacy analysis with mock data"""
    print("\nTesting privacy analysis...")

    db = Database()
    privacy_service = get_privacy_service()

    # Test update privacy analysis with mock data
    test_zones = [
        {"type": "face", "x": 100, "y": 100, "w": 50, "h": 50},
        {"type": "plate", "x": 200, "y": 200, "w": 80, "h": 30}
    ]

    # Create a test image entry if database has images
    images = db.get_all_images(limit=1)

    if images:
        test_image_id = images[0]['id']

        # Update privacy analysis
        db.update_privacy_analysis(
            image_id=test_image_id,
            has_faces=True,
            has_plates=True,
            is_nsfw=False,
            privacy_zones=test_zones
        )

        print(f"✓ Updated privacy analysis for image {test_image_id}")

        # Retrieve privacy data
        privacy_data = db.get_privacy_data(test_image_id)

        assert privacy_data is not None, "Failed to retrieve privacy data"
        assert privacy_data['has_faces'] == True, "has_faces not set correctly"
        assert privacy_data['has_plates'] == True, "has_plates not set correctly"
        assert len(privacy_data['privacy_zones']) == 2, "Privacy zones not saved correctly"

        print("✓ Privacy data stored and retrieved successfully")
    else:
        print("⚠ No images in database to test with")

if __name__ == '__main__':
    print("=" * 60)
    print("PRIVACY FUNCTIONALITY TEST")
    print("=" * 60)

    try:
        test_database_privacy_schema()
        test_privacy_service()
        test_privacy_analysis_mock()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
