"""
Database layer for AI Gallery
Handles all SQLite operations for images, boards, and relationships
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class Database:
    def __init__(self, db_path: str = "data/gallery.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection with row factory and timeout"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
    
    
    
    
        """Initialize database schema"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                width INTEGER,
                height INTEGER,
                file_size INTEGER,
                is_favorite BOOLEAN DEFAULT 0,
                analyzed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Boards table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS boards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                parent_id INTEGER,
                cover_image_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES boards(id) ON DELETE CASCADE,
                FOREIGN KEY (cover_image_id) REFERENCES images(id) ON DELETE SET NULL
            )
        """)
        
        # Board-Image relationships (many-to-many)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS board_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                board_id INTEGER NOT NULL,
                image_id INTEGER NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (board_id) REFERENCES boards(id) ON DELETE CASCADE,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                UNIQUE(board_id, image_id)
            )
        """)

        # Image Embeddings for semantic search
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_embeddings (
                image_id INTEGER PRIMARY KEY,
                model_name TEXT NOT NULL,
                vector BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
            )
        """)

        # EXIF data for images
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exif_data (
                image_id INTEGER PRIMARY KEY,
                camera_make TEXT,
                camera_model TEXT,
                lens TEXT,
                iso INTEGER,
                aperture REAL,
                shutter_speed TEXT,
                focal_length REAL,
                capture_date TIMESTAMP,
                gps_lat REAL,
                gps_lon REAL,
                orientation INTEGER,
                flash TEXT,
                white_balance TEXT,
                metering_mode TEXT,
                exposure_mode TEXT,
                software TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
            )
        """)

        # Color analysis for images
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_colors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                hex_color TEXT NOT NULL,
                percentage REAL,
                is_dominant BOOLEAN DEFAULT 0,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
            )
        """)

        # Audit log for tracking changes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id INTEGER,
                action TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                triggered_by TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add media_type column if it doesn't exist (migration)
        cursor.execute("PRAGMA table_info(images)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'media_type' not in columns:
            cursor.execute("ALTER TABLE images ADD COLUMN media_type TEXT DEFAULT 'image'")
            conn.commit()

        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_filepath ON images(filepath)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_favorite ON images(is_favorite)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_analyzed ON images(analyzed_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_created ON images(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_media_type ON images(media_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_boards_parent ON boards(parent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_board_images_board ON board_images(board_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_board_images_image ON board_images(image_id)")

        # Indexes for new tables
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_model ON image_embeddings(model_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exif_camera ON exif_data(camera_make, camera_model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exif_capture_date ON exif_data(capture_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_colors_image ON image_colors(image_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_colors_dominant ON image_colors(is_dominant)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")

        # Full-text search index - check if needs migration
        cursor.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type = 'table' AND name = 'images_fts'
        """)
        fts_table = cursor.fetchone()
        recreate_fts = False
        
        if not fts_table:
            # Create new FTS table
            self._create_fulltext_table(cursor)
            recreate_fts = True
        elif fts_table['sql'] and 'content=' in fts_table['sql'].lower():
            # Old schema with content=images, needs migration
            cursor.execute("DROP TABLE IF EXISTS images_fts")
            self._create_fulltext_table(cursor)
            recreate_fts = True
        
        conn.commit()
        conn.close()
        
        # Rebuild FTS index if needed
        if recreate_fts:
            self.rebuild_fulltext_index()
    
    # ============ IMAGE OPERATIONS ============
    
    def add_image(self, filepath: str, filename: str = None, width: int = None,
                  height: int = None, file_size: int = None, media_type: str = 'image') -> int:
        """Add new image/video to database"""
        # Extract filename from filepath if not provided
        if filename is None:
            from pathlib import Path
            filename = Path(filepath).name

        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO images (filepath, filename, width, height, file_size, media_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (filepath, filename, width, height, file_size, media_type))

            image_id = cursor.lastrowid

            # Update FTS index
            cursor.execute("""
                INSERT INTO images_fts (rowid, filename, description, tags)
                VALUES (?, ?, '', '')
            """, (image_id, filename))

            conn.commit()
            return image_id
        except sqlite3.IntegrityError:
            # Image already exists
            cursor.execute("SELECT id FROM images WHERE filepath = ?", (filepath,))
            result = cursor.fetchone()
            return result['id'] if result else None
        finally:
            conn.close()
    
    def get_image(self, image_id: int) -> Optional[Dict]:
        """Get single image by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return self._row_to_dict(result)
        return None
    
    def get_all_images(self, limit: int = 1000, offset: int = 0, 
                       favorites_only: bool = False,
                       media_type: Optional[str] = None,
                       analyzed: Optional[bool] = None) -> List[Dict]:
        """Get all images with pagination and optional filters"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM images"
        clauses = []
        params = []
        
        if favorites_only:
            clauses.append("is_favorite = 1")
        
        if media_type:
            clauses.append("media_type = ?")
            params.append(media_type)
        
        if analyzed is True:
            clauses.append("analyzed_at IS NOT NULL")
        elif analyzed is False:
            clauses.append("analyzed_at IS NULL")
        
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in results]
    
    def update_image_analysis(self, image_id: int, description: str, tags: List[str]):
        """Update image with AI analysis results"""
        tags_json = json.dumps(tags)
        tags_text = ' '.join(tags)
        
        try:
            self._apply_image_analysis_update(image_id, description, tags_json, tags_text)
        except sqlite3.DatabaseError as error:
            if "malformed" in str(error).lower():
                print("Detected corrupted full-text index, attempting rebuild...")
                self.rebuild_fulltext_index()
                # Retry the update
                self._apply_image_analysis_update(image_id, description, tags_json, tags_text)
            else:
                raise
    
    def _apply_image_analysis_update(self, image_id: int, description: str,
                                     tags_json: str, tags_text: str):
        """Internal helper to perform image analysis update with FTS sync"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE images 
                SET description = ?, tags = ?, analyzed_at = ?, updated_at = ?
                WHERE id = ?
            """, (description, tags_json, datetime.now(), datetime.now(), image_id))
            
            # Update FTS index
            cursor.execute("""
                UPDATE images_fts 
                SET description = ?, tags = ?
                WHERE rowid = ?
            """, (description, tags_text, image_id))
            
            conn.commit()
        finally:
            conn.close()
    
    def toggle_favorite(self, image_id: int) -> bool:
        """Toggle favorite status, return new status"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT is_favorite FROM images WHERE id = ?", (image_id,))
        result = cursor.fetchone()
        
        if result:
            new_status = not result['is_favorite']
            cursor.execute("""
                UPDATE images SET is_favorite = ?, updated_at = ? WHERE id = ?
            """, (new_status, datetime.now(), image_id))
            conn.commit()
            conn.close()
            return new_status
        
        conn.close()
        return False
    
    def rename_image(self, image_id: int, new_filepath: str, new_filename: str):
        """Update image filepath after rename"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE images 
            SET filepath = ?, filename = ?, updated_at = ?
            WHERE id = ?
        """, (new_filepath, new_filename, datetime.now(), image_id))
        
        # Update FTS index
        cursor.execute("""
            UPDATE images_fts SET filename = ? WHERE rowid = ?
        """, (new_filename, image_id))
        
        conn.commit()
        conn.close()
    
    def search_images(self, query: str, limit: int = 100) -> List[Dict]:
        """Full-text search across filename, description, tags"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Use FTS5 for full-text search
        cursor.execute("""
            SELECT i.* FROM images i
            JOIN images_fts fts ON i.id = fts.rowid
            WHERE images_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))
        
        results = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in results]

    def get_similar_images(self, image_id: int, limit: int = 6) -> List[Dict]:
        """Get similar images based on shared tags"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Get the source image's tags
        cursor.execute("SELECT tags FROM images WHERE id = ?", (image_id,))
        result = cursor.fetchone()

        if not result or not result['tags']:
            conn.close()
            return []

        try:
            source_tags = json.loads(result['tags'])
        except:
            conn.close()
            return []

        if not source_tags:
            conn.close()
            return []

        # Find images with matching tags (simplified approach without json_each)
        # Get all analyzed images with tags
        cursor.execute("""
            SELECT * FROM images
            WHERE id != ?
              AND analyzed_at IS NOT NULL
              AND tags IS NOT NULL
              AND tags != '[]'
            ORDER BY analyzed_at DESC
            LIMIT 50
        """, (image_id,))

        results = cursor.fetchall()
        conn.close()

        # Calculate similarity based on shared tags
        similar_images = []
        for row in results:
            try:
                img_tags = json.loads(row['tags'])
                shared_tags = set(source_tags) & set(img_tags)
                if len(shared_tags) > 0:
                    img_dict = self._row_to_dict(row)
                    img_dict['similarity_score'] = len(shared_tags)
                    similar_images.append(img_dict)
            except:
                continue

        # Sort by similarity and return top N
        similar_images.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_images[:limit]

    def get_unanalyzed_images(self, limit: int = 100) -> List[Dict]:
        """Get images that haven't been analyzed yet"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM images 
            WHERE analyzed_at IS NULL 
            ORDER BY created_at ASC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in results]
    
    def delete_image(self, image_id: int):
        """Delete image from database (also removes from all boards)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
        cursor.execute("DELETE FROM images_fts WHERE rowid = ?", (image_id,))

        conn.commit()
        conn.close()

    # ============ TAG OPERATIONS ============

    def get_all_tags(self) -> List[Dict]:
        """
        Get all unique tags with usage count, sorted by popularity
        Returns: [{'tag': 'sunset', 'count': 5}, ...]
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT tags FROM images
            WHERE tags IS NOT NULL AND tags != '[]'
        """)

        results = cursor.fetchall()
        conn.close()

        # Count tag occurrences
        tag_counts = {}
        for row in results:
            try:
                tags = json.loads(row['tags'])
                for tag in tags:
                    tag = tag.lower().strip()
                    if tag:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
            except:
                continue

        # Convert to list of dicts and sort by count
        tag_list = [{'tag': tag, 'count': count} for tag, count in tag_counts.items()]
        tag_list.sort(key=lambda x: x['count'], reverse=True)

        return tag_list

    def get_tag_suggestions(self, prefix: str = '', limit: int = 10) -> List[str]:
        """
        Get tag suggestions for autocomplete

        Args:
            prefix: String prefix to filter tags (case-insensitive)
            limit: Maximum number of suggestions to return

        Returns: List of tag strings sorted by popularity
        """
        all_tags = self.get_all_tags()

        if prefix:
            prefix_lower = prefix.lower()
            filtered = [t for t in all_tags if t['tag'].startswith(prefix_lower)]
        else:
            filtered = all_tags

        return [t['tag'] for t in filtered[:limit]]

    def get_related_tags(self, tag: str, limit: int = 10) -> List[Dict]:
        """
        Get tags that frequently appear together with the given tag

        Args:
            tag: The tag to find related tags for
            limit: Maximum number of related tags to return

        Returns: [{'tag': 'related_tag', 'co_occurrence': 3}, ...]
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        tag_lower = tag.lower().strip()

        # Find all images with this tag
        cursor.execute("""
            SELECT tags FROM images
            WHERE tags IS NOT NULL AND tags != '[]'
        """)

        results = cursor.fetchall()
        conn.close()

        # Count co-occurrences
        co_occurrences = {}
        for row in results:
            try:
                tags = json.loads(row['tags'])
                tags_lower = [t.lower().strip() for t in tags]

                # If this image has the target tag
                if tag_lower in tags_lower:
                    # Count all other tags
                    for other_tag in tags_lower:
                        if other_tag != tag_lower:
                            co_occurrences[other_tag] = co_occurrences.get(other_tag, 0) + 1
            except:
                continue

        # Convert to list and sort
        related = [{'tag': t, 'co_occurrence': count} for t, count in co_occurrences.items()]
        related.sort(key=lambda x: x['co_occurrence'], reverse=True)

        return related[:limit]

    # ============ BOARD OPERATIONS ============
    
    def create_board(self, name: str, description: str = None, 
                     parent_id: int = None) -> int:
        """Create new board"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO boards (name, description, parent_id)
            VALUES (?, ?, ?)
        """, (name, description, parent_id))
        
        board_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return board_id
    
    def get_board(self, board_id: int) -> Optional[Dict]:
        """Get single board by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM boards WHERE id = ?", (board_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return self._row_to_dict(result)
        return None
    
    def get_all_boards(self) -> List[Dict]:
        """Get all boards"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM boards ORDER BY name")
        results = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in results]
    
    def get_sub_boards(self, parent_id: int = None) -> List[Dict]:
        """Get boards with specific parent (or top-level if parent_id is None)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if parent_id is None:
            cursor.execute("SELECT * FROM boards WHERE parent_id IS NULL ORDER BY name")
        else:
            cursor.execute("SELECT * FROM boards WHERE parent_id = ? ORDER BY name", (parent_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in results]
    
    def update_board(self, board_id: int, name: str = None, description: str = None):
        """Update board details"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        
        if updates:
            updates.append("updated_at = ?")
            params.append(datetime.now())
            params.append(board_id)
            
            query = f"UPDATE boards SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            conn.commit()
        
        conn.close()

    def move_board(self, board_id: int, new_parent_id: int = None):
        """
        Move board to a new parent (or to top level if new_parent_id is None)
        Prevents circular dependencies
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Check if new_parent_id is the same as current board
        if new_parent_id == board_id:
            conn.close()
            raise ValueError("Board cannot be its own parent")

        # Check if new_parent_id would create a circular dependency
        # (i.e., new parent is a descendant of the board being moved)
        if new_parent_id is not None:
            all_sub_boards = self._get_all_sub_boards(board_id, cursor)
            sub_board_ids = [b['id'] for b in all_sub_boards]

            if new_parent_id in sub_board_ids:
                conn.close()
                raise ValueError("Cannot move board under its own sub-board (circular dependency)")

        # Update the parent_id
        cursor.execute("""
            UPDATE boards
            SET parent_id = ?, updated_at = ?
            WHERE id = ?
        """, (new_parent_id, datetime.now(), board_id))

        conn.commit()
        conn.close()

    def delete_board(self, board_id: int, delete_sub_boards: bool = False):
        """
        Delete board and optionally its sub-boards

        Args:
            board_id: Board to delete
            delete_sub_boards: If True, also delete all sub-boards recursively
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        if delete_sub_boards:
            # Get all sub-boards recursively
            sub_boards = self._get_all_sub_boards(board_id, cursor)

            # Delete sub-boards first (bottom-up)
            for sub_board_id in reversed(sub_boards):
                cursor.execute("DELETE FROM boards WHERE id = ?", (sub_board_id,))
                print(f"Deleted sub-board {sub_board_id}")

        # Delete the board itself
        cursor.execute("DELETE FROM boards WHERE id = ?", (board_id,))
        print(f"Deleted board {board_id}")

        conn.commit()
        conn.close()

    def _get_all_sub_boards(self, board_id: int, cursor) -> List[int]:
        """Recursively get all sub-board IDs"""
        sub_board_ids = []

        cursor.execute("SELECT id FROM boards WHERE parent_id = ?", (board_id,))
        direct_subs = cursor.fetchall()

        for sub in direct_subs:
            sub_id = sub['id']
            sub_board_ids.append(sub_id)
            # Recursively get sub-boards of this sub-board
            sub_board_ids.extend(self._get_all_sub_boards(sub_id, cursor))

        return sub_board_ids

    def merge_boards(self, source_board_id: int, target_board_id: int, delete_source: bool = True):
        """
        Merge source board into target board

        Args:
            source_board_id: Board to merge from
            target_board_id: Board to merge into
            delete_source: If True, delete source board after merge

        Returns:
            Number of images moved
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Get all images from source board
        cursor.execute("""
            SELECT image_id FROM board_images
            WHERE board_id = ?
        """, (source_board_id,))

        images = cursor.fetchall()
        moved_count = 0

        # Move images to target board
        for row in images:
            image_id = row['image_id']
            try:
                cursor.execute("""
                    INSERT INTO board_images (board_id, image_id)
                    VALUES (?, ?)
                """, (target_board_id, image_id))
                moved_count += 1
            except sqlite3.IntegrityError:
                # Already in target board, skip
                pass

        # Move sub-boards to target board
        cursor.execute("""
            UPDATE boards
            SET parent_id = ?
            WHERE parent_id = ?
        """, (target_board_id, source_board_id))

        sub_boards_moved = cursor.rowcount

        conn.commit()

        # Delete source board if requested
        if delete_source:
            cursor.execute("DELETE FROM boards WHERE id = ?", (source_board_id,))
            conn.commit()
            print(f"Merged board {source_board_id} into {target_board_id}: {moved_count} images, {sub_boards_moved} sub-boards moved")
        else:
            print(f"Copied {moved_count} images from board {source_board_id} to {target_board_id}")

        conn.close()
        return moved_count
    
    # ============ BOARD-IMAGE RELATIONSHIPS ============
    
    def get_parent_boards(self, board_id: int) -> List[int]:
        """Get all parent boards recursively (from direct parent up to root)"""
        parent_ids = []
        current_id = board_id

        conn = self.get_connection()
        cursor = conn.cursor()

        print(f"get_parent_boards: Looking for parents of board {board_id}")

        # Walk up the tree to find all parents
        max_depth = 10  # Prevent infinite loops
        depth = 0
        while depth < max_depth:
            cursor.execute("SELECT id, name, parent_id FROM boards WHERE id = ?", (current_id,))
            result = cursor.fetchone()

            if not result:
                print(f"  Board {current_id} not found in database!")
                break

            print(f"  Depth {depth}: Board {result['id']} ('{result['name']}') has parent_id={result['parent_id']}")

            if result['parent_id'] is None:
                print(f"  Reached root board (no parent)")
                break

            parent_id = result['parent_id']
            parent_ids.append(parent_id)
            current_id = parent_id
            depth += 1

        conn.close()
        print(f"get_parent_boards: Found {len(parent_ids)} parents: {parent_ids}")
        return parent_ids

    def add_image_to_board(self, board_id: int, image_id: int, auto_add_to_parents: bool = True):
        """
        Add image to board and optionally to all parent boards

        Args:
            board_id: The board to add the image to
            image_id: The image to add
            auto_add_to_parents: If True, also adds image to all parent boards
        """
        print(f"add_image_to_board: board_id={board_id}, image_id={image_id}, auto_add_to_parents={auto_add_to_parents}")

        conn = self.get_connection()
        cursor = conn.cursor()

        # Add to the specified board
        try:
            cursor.execute("""
                INSERT INTO board_images (board_id, image_id)
                VALUES (?, ?)
            """, (board_id, image_id))
            conn.commit()
            print(f"✓ Added image {image_id} to board {board_id}")
        except sqlite3.IntegrityError:
            # Already exists, ignore
            print(f"Image {image_id} already in board {board_id}")
            pass
        finally:
            conn.close()

        # Auto-add to parent boards if enabled
        if auto_add_to_parents:
            parent_ids = self.get_parent_boards(board_id)
            print(f"Found {len(parent_ids)} parent boards: {parent_ids}")

            if len(parent_ids) == 0:
                print(f"Board {board_id} has no parent boards")

            for parent_id in parent_ids:
                conn = self.get_connection()
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        INSERT INTO board_images (board_id, image_id)
                        VALUES (?, ?)
                    """, (parent_id, image_id))
                    conn.commit()
                    print(f"✓ Auto-added image {image_id} to parent board {parent_id}")
                except sqlite3.IntegrityError:
                    # Already exists, ignore
                    print(f"Image {image_id} already in parent board {parent_id}")
                    pass
                finally:
                    conn.close()
        else:
            print(f"Auto-add to parents is disabled")
    
    def remove_image_from_board(self, board_id: int, image_id: int):
        """Remove image from board"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM board_images 
            WHERE board_id = ? AND image_id = ?
        """, (board_id, image_id))
        
        conn.commit()
        conn.close()
    
    def get_board_images(self, board_id: int) -> List[Dict]:
        """Get all images in a board"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT i.* FROM images i
            JOIN board_images bi ON i.id = bi.image_id
            WHERE bi.board_id = ?
            ORDER BY bi.added_at DESC
        """, (board_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in results]
    
    def get_image_boards(self, image_id: int) -> List[Dict]:
        """Get all boards containing an image"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT b.* FROM boards b
            JOIN board_images bi ON b.id = bi.board_id
            WHERE bi.image_id = ?
            ORDER BY b.name
        """, (image_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in results]
    
    # ============ STATISTICS ============
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total images
        cursor.execute("SELECT COUNT(*) as count FROM images")
        total_images = cursor.fetchone()['count']
        
        # Analyzed images
        cursor.execute("SELECT COUNT(*) as count FROM images WHERE analyzed_at IS NOT NULL")
        analyzed_images = cursor.fetchone()['count']
        
        # Favorite images
        cursor.execute("SELECT COUNT(*) as count FROM images WHERE is_favorite = 1")
        favorite_images = cursor.fetchone()['count']
        
        # Total boards
        cursor.execute("SELECT COUNT(*) as count FROM boards")
        total_boards = cursor.fetchone()['count']
        
        conn.close()
        
        return {
            'total_images': total_images,
            'analyzed_images': analyzed_images,
            'unanalyzed_images': total_images - analyzed_images,
            'favorite_images': favorite_images,
            'total_boards': total_boards
        }
    
    # ============ HELPER METHODS ============
    
    def rebuild_fulltext_index(self):
        """Rebuild the FTS index from the images table"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Drop and recreate FTS table with retries
            for attempt in range(3):
                try:
                    cursor.execute("DROP TABLE IF EXISTS images_fts")
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" in str(exc).lower() and attempt < 2:
                        import time
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    raise
            
            self._create_fulltext_table(cursor)
            
            # Rebuild from images table
            cursor.execute("SELECT id, filename, description, tags FROM images")
            rows = cursor.fetchall()
            
            for row in rows:
                tags_list = []
                
                if row['tags']:
                    try:
                        tags_list = json.loads(row['tags'])
                    except json.JSONDecodeError:
                        # Corrupted tag data, skip gracefully
                        tags_list = []
                
                cursor.execute("""
                    INSERT INTO images_fts (rowid, filename, description, tags)
                    VALUES (?, ?, ?, ?)
                """, (
                    row['id'],
                    row['filename'] or '',
                    row['description'] or '',
                    ' '.join(tags_list)
                ))
            
            conn.commit()
        finally:
            conn.close()
    
    def _create_fulltext_table(self, cursor):
        """Create the FTS5 table with the expected schema (no content= clause)"""
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(
                filename, description, tags
            )
        """)
    
    def _row_to_dict(self, row) -> Dict:
        """Convert SQLite Row to dictionary"""
        if row is None:
            return None
        
        data = dict(row)
        
        # Parse JSON fields
        if 'tags' in data and data['tags']:
            try:
                data['tags'] = json.loads(data['tags'])
            except:
                data['tags'] = []
        else:
            data['tags'] = []
        
        # Convert boolean
        if 'is_favorite' in data:
            data['is_favorite'] = bool(data['is_favorite'])

        return data

    # ============ EXIF DATA OPERATIONS ============

    def save_exif_data(self, image_id: int, exif_dict: Dict) -> bool:
        """Save EXIF data for an image"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO exif_data (
                    image_id, camera_make, camera_model, lens, iso, aperture,
                    shutter_speed, focal_length, capture_date, gps_lat, gps_lon,
                    orientation, flash, white_balance, metering_mode, exposure_mode, software
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_id,
                exif_dict.get('camera_make'),
                exif_dict.get('camera_model'),
                exif_dict.get('lens'),
                exif_dict.get('iso'),
                exif_dict.get('aperture'),
                exif_dict.get('shutter_speed'),
                exif_dict.get('focal_length'),
                exif_dict.get('capture_date'),
                exif_dict.get('gps_lat'),
                exif_dict.get('gps_lon'),
                exif_dict.get('orientation'),
                exif_dict.get('flash'),
                exif_dict.get('white_balance'),
                exif_dict.get('metering_mode'),
                exif_dict.get('exposure_mode'),
                exif_dict.get('software')
            ))

            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving EXIF data: {e}")
            return False
        finally:
            conn.close()

    def get_exif_data(self, image_id: int) -> Optional[Dict]:
        """Get EXIF data for an image"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM exif_data WHERE image_id = ?", (image_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def search_images_by_exif(self, filters: Dict) -> List[Dict]:
        """Search images by EXIF criteria

        Example filters:
        {
            'camera_make': 'Canon',
            'min_iso': 400,
            'max_iso': 1600,
            'min_aperture': 1.4,
            'focal_length': 50
        }
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            query = """
                SELECT i.* FROM images i
                JOIN exif_data e ON i.id = e.image_id
                WHERE 1=1
            """
            params = []

            if 'camera_make' in filters:
                query += " AND e.camera_make = ?"
                params.append(filters['camera_make'])

            if 'camera_model' in filters:
                query += " AND e.camera_model = ?"
                params.append(filters['camera_model'])

            if 'min_iso' in filters:
                query += " AND e.iso >= ?"
                params.append(filters['min_iso'])

            if 'max_iso' in filters:
                query += " AND e.iso <= ?"
                params.append(filters['max_iso'])

            if 'min_aperture' in filters:
                query += " AND e.aperture >= ?"
                params.append(filters['min_aperture'])

            if 'max_aperture' in filters:
                query += " AND e.aperture <= ?"
                params.append(filters['max_aperture'])

            if 'focal_length' in filters:
                query += " AND e.focal_length = ?"
                params.append(filters['focal_length'])

            query += " ORDER BY i.created_at DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [self._row_to_dict(row) for row in rows]
        finally:
            conn.close()

    # ============ EMBEDDING OPERATIONS ============

    def save_embedding(self, image_id: int, model_name: str, vector: bytes) -> bool:
        """Save image embedding vector"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO image_embeddings (image_id, model_name, vector)
                VALUES (?, ?, ?)
            """, (image_id, model_name, vector))

            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving embedding: {e}")
            return False
        finally:
            conn.close()

    def get_embedding(self, image_id: int) -> Optional[Dict]:
        """Get embedding for an image"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT image_id, model_name, vector, created_at
                FROM image_embeddings
                WHERE image_id = ?
            """, (image_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_all_embeddings(self, model_name: Optional[str] = None) -> List[Dict]:
        """Get all embeddings, optionally filtered by model"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            if model_name:
                cursor.execute("""
                    SELECT image_id, model_name, vector, created_at
                    FROM image_embeddings
                    WHERE model_name = ?
                """, (model_name,))
            else:
                cursor.execute("""
                    SELECT image_id, model_name, vector, created_at
                    FROM image_embeddings
                """)

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def delete_embedding(self, image_id: int) -> bool:
        """Delete embedding for an image"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM image_embeddings WHERE image_id = ?", (image_id,))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting embedding: {e}")
            return False
        finally:
            conn.close()

    # ============ COLOR OPERATIONS ============

    def save_image_colors(self, image_id: int, colors: List[Dict]) -> bool:
        """Save color palette for an image

        Args:
            image_id: Image ID
            colors: List of dicts with keys: hex_color, percentage, is_dominant
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Delete existing colors
            cursor.execute("DELETE FROM image_colors WHERE image_id = ?", (image_id,))

            # Insert new colors
            for color in colors:
                cursor.execute("""
                    INSERT INTO image_colors (image_id, hex_color, percentage, is_dominant)
                    VALUES (?, ?, ?, ?)
                """, (
                    image_id,
                    color['hex_color'],
                    color.get('percentage'),
                    color.get('is_dominant', False)
                ))

            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving colors: {e}")
            return False
        finally:
            conn.close()

    def get_image_colors(self, image_id: int) -> List[Dict]:
        """Get color palette for an image"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT hex_color, percentage, is_dominant
                FROM image_colors
                WHERE image_id = ?
                ORDER BY percentage DESC
            """, (image_id,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def search_images_by_color(self, hex_color: str, tolerance: int = 30) -> List[Dict]:
        """Search images by dominant color (simple hex matching)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Simple implementation - exact match on dominant colors
            # A more sophisticated version would calculate color distance
            cursor.execute("""
                SELECT DISTINCT i.* FROM images i
                JOIN image_colors c ON i.id = c.image_id
                WHERE c.is_dominant = 1 AND c.hex_color = ?
                ORDER BY i.created_at DESC
            """, (hex_color,))

            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
        finally:
            conn.close()

    # ============ AUDIT LOG OPERATIONS ============

    def log_action(self, entity_type: str, entity_id: int, action: str,
                   old_value: str = None, new_value: str = None,
                   triggered_by: str = 'user') -> bool:
        """Log an action to the audit log"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO audit_log (entity_type, entity_id, action, old_value, new_value, triggered_by)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (entity_type, entity_id, action, old_value, new_value, triggered_by))

            conn.commit()
            return True
        except Exception as e:
            print(f"Error logging action: {e}")
            return False
        finally:
            conn.close()

    def get_audit_log(self, entity_type: Optional[str] = None,
                      entity_id: Optional[int] = None,
                      limit: int = 100) -> List[Dict]:
        """Get audit log entries"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            query = "SELECT * FROM audit_log WHERE 1=1"
            params = []

            if entity_type:
                query += " AND entity_type = ?"
                params.append(entity_type)

            if entity_id:
                query += " AND entity_id = ?"
                params.append(entity_id)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]
        finally:
            conn.close()