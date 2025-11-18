#!/usr/bin/env python3
"""
Reset the database by removing all data
"""
import os
import shutil

DATA_DIR = 'data'
DATABASE_PATH = os.path.join(DATA_DIR, 'gallery.db')
THUMBNAILS_DIR = os.path.join(DATA_DIR, 'thumbnails')

def reset_database():
    """Remove database and thumbnails"""

    # Remove database
    if os.path.exists(DATABASE_PATH):
        os.remove(DATABASE_PATH)
        print(f"✓ Removed database: {DATABASE_PATH}")
    else:
        print(f"ℹ Database not found: {DATABASE_PATH}")

    # Remove thumbnails
    if os.path.exists(THUMBNAILS_DIR):
        shutil.rmtree(THUMBNAILS_DIR)
        print(f"✓ Removed thumbnails: {THUMBNAILS_DIR}")
    else:
        print(f"ℹ Thumbnails directory not found: {THUMBNAILS_DIR}")

    print("\n✨ Database reset complete!")
    print("\nNext steps:")
    print("1. Add images to the 'photos/' directory")
    print("2. Start the application: python app.py")
    print("3. Click 'Scan Directory' to import images")

if __name__ == '__main__':
    reset_database()
