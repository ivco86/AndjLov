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
        
        # Add media_type column if it doesn't exist (migration)
        cursor.execute("PRAGMA table_info(images)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'media_type' not in columns:
            cursor.execute("ALTER TABLE images ADD COLUMN media_type TEXT DEFAULT 'image'")
            conn.commit()

        # Add privacy columns if they don't exist (migration)
        cursor.execute("PRAGMA table_info(images)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'has_faces' not in columns:
            cursor.execute("ALTER TABLE images ADD COLUMN has_faces BOOLEAN DEFAULT 0")
            conn.commit()

        if 'has_plates' not in columns:
            cursor.execute("ALTER TABLE images ADD COLUMN has_plates BOOLEAN DEFAULT 0")
            conn.commit()

        if 'is_nsfw' not in columns:
            cursor.execute("ALTER TABLE images ADD COLUMN is_nsfw BOOLEAN DEFAULT 0")
            conn.commit()

        if 'privacy_zones' not in columns:
            cursor.execute("ALTER TABLE images ADD COLUMN privacy_zones TEXT")
            conn.commit()

        if 'privacy_analyzed_at' not in columns:
            cursor.execute("ALTER TABLE images ADD COLUMN privacy_analyzed_at TIMESTAMP")
            conn.commit()

        # Annotations table for research/ML training
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                class_id INTEGER,
                x REAL NOT NULL,
                y REAL NOT NULL,
                width REAL NOT NULL,
                height REAL NOT NULL,
                confidence REAL DEFAULT 1.0,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
            )
        """)

        # Dataset classes table for ML training
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                color TEXT DEFAULT '#FF5722',
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Workflow pipelines table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipelines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                trigger_type TEXT NOT NULL,
                trigger_config TEXT,
                actions TEXT NOT NULL,
                enabled BOOLEAN DEFAULT 1,
                last_run TIMESTAMP,
                run_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Pipeline execution logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_id INTEGER NOT NULL,
                trigger_source TEXT,
                status TEXT NOT NULL,
                total_actions INTEGER DEFAULT 0,
                completed_actions INTEGER DEFAULT 0,
                failed_actions INTEGER DEFAULT 0,
                execution_log TEXT,
                error_message TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (pipeline_id) REFERENCES pipelines(id) ON DELETE CASCADE
            )
        """)

        # Settings table for app configuration
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_filepath ON images(filepath)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_favorite ON images(is_favorite)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_analyzed ON images(analyzed_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_created ON images(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_media_type ON images(media_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_privacy_analyzed ON images(privacy_analyzed_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_has_faces ON images(has_faces)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_nsfw ON images(is_nsfw)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_boards_parent ON boards(parent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_board_images_board ON board_images(board_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_board_images_image ON board_images(image_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_annotations_image ON annotations(image_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_annotations_class ON annotations(class_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pipelines_enabled ON pipelines(enabled)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pipelines_trigger_type ON pipelines(trigger_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_executions_pipeline ON pipeline_executions(pipeline_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_executions_status ON pipeline_executions(status)")
        
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

    # ============ PRIVACY OPERATIONS ============

    def update_privacy_analysis(self, image_id: int, has_faces: bool = False,
                                has_plates: bool = False, is_nsfw: bool = False,
                                privacy_zones: List[Dict] = None):
        """
        Update image with privacy analysis results

        Args:
            image_id: Image ID
            has_faces: Whether faces were detected
            has_plates: Whether license plates were detected
            is_nsfw: Whether NSFW content was detected
            privacy_zones: List of zones to blur, format: [{"type": "face", "x": 100, "y": 50, "w": 80, "h": 80}, ...]
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        privacy_zones_json = json.dumps(privacy_zones) if privacy_zones else None

        cursor.execute("""
            UPDATE images
            SET has_faces = ?, has_plates = ?, is_nsfw = ?,
                privacy_zones = ?, privacy_analyzed_at = ?, updated_at = ?
            WHERE id = ?
        """, (has_faces, has_plates, is_nsfw, privacy_zones_json,
              datetime.now(), datetime.now(), image_id))

        conn.commit()
        conn.close()

    def get_privacy_data(self, image_id: int) -> Optional[Dict]:
        """
        Get privacy analysis data for an image

        Returns:
            Dictionary with privacy data or None if not analyzed
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT has_faces, has_plates, is_nsfw, privacy_zones, privacy_analyzed_at
            FROM images
            WHERE id = ?
        """, (image_id,))

        result = cursor.fetchone()
        conn.close()

        if result:
            data = dict(result)
            # Parse privacy_zones JSON
            if data['privacy_zones']:
                try:
                    data['privacy_zones'] = json.loads(data['privacy_zones'])
                except:
                    data['privacy_zones'] = []
            else:
                data['privacy_zones'] = []

            return data

        return None

    def get_unanalyzed_privacy_images(self, limit: int = 100) -> List[Dict]:
        """Get images that haven't been privacy analyzed yet"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM images
            WHERE privacy_analyzed_at IS NULL
            ORDER BY created_at ASC
            LIMIT ?
        """, (limit,))

        results = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in results]

    def get_images_with_faces(self, limit: int = 100) -> List[Dict]:
        """Get images that contain detected faces"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM images
            WHERE has_faces = 1
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        results = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in results]

    def get_nsfw_images(self, limit: int = 100) -> List[Dict]:
        """Get images flagged as NSFW"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM images
            WHERE is_nsfw = 1
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        results = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in results]

    # ============ EXPORT/IMPORT OPERATIONS ============

    def export_data_json(self, include_images: bool = True, include_boards: bool = True) -> str:
        """
        Export all data as JSON string

        Args:
            include_images: Include image data in export
            include_boards: Include board data in export

        Returns:
            JSON string with all data
        """
        export_data = {
            'export_date': datetime.now().isoformat(),
            'version': '1.0'
        }

        if include_images:
            images = self.get_all_images(limit=10000)
            export_data['images'] = images
            export_data['total_images'] = len(images)

        if include_boards:
            boards = self.get_all_boards()
            # Enrich boards with image IDs
            for board in boards:
                board_images = self.get_board_images(board['id'])
                board['image_ids'] = [img['id'] for img in board_images]
                board['image_count'] = len(board_images)

            export_data['boards'] = boards
            export_data['total_boards'] = len(boards)

        # Get tags
        export_data['tags'] = self.get_all_tags()

        return json.dumps(export_data, indent=2, ensure_ascii=False)

    def export_data_markdown(self, include_images: bool = True, include_boards: bool = True) -> str:
        """
        Export all data as Markdown format (human-readable, AI-friendly)

        Args:
            include_images: Include image data in export
            include_boards: Include board data in export

        Returns:
            Markdown formatted string
        """
        md_lines = []
        md_lines.append("# AI Gallery Export")
        md_lines.append(f"\n**Export Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        stats = self.get_stats()
        md_lines.append("## Statistics")
        md_lines.append(f"- Total Images: {stats['total_images']}")
        md_lines.append(f"- Analyzed Images: {stats['analyzed_images']}")
        md_lines.append(f"- Favorite Images: {stats['favorite_images']}")
        md_lines.append(f"- Total Boards: {stats['total_boards']}\n")

        if include_boards:
            md_lines.append("## Boards\n")
            boards = self.get_all_boards()

            # Organize boards hierarchically
            board_dict = {b['id']: b for b in boards}
            root_boards = [b for b in boards if b['parent_id'] is None]

            def render_board(board, level=0):
                indent = "  " * level
                lines = []
                lines.append(f"{indent}### {'📁 ' * (level + 1)}{board['name']}")

                if board['description']:
                    lines.append(f"{indent}**Description:** {board['description']}")

                # Get images in this board
                board_images = self.get_board_images(board['id'])
                lines.append(f"{indent}**Images:** {len(board_images)}")

                if board_images:
                    lines.append(f"{indent}**Image List:**")
                    for img in board_images[:5]:  # Show first 5
                        tags_str = ', '.join(img['tags'][:3]) if img['tags'] else 'no tags'
                        lines.append(f"{indent}- `{img['filename']}` - {img['description'][:50] if img['description'] else 'No description'}... (Tags: {tags_str})")

                    if len(board_images) > 5:
                        lines.append(f"{indent}  _(and {len(board_images) - 5} more images)_")

                lines.append("")

                # Recursively render sub-boards
                sub_boards = [b for b in boards if b['parent_id'] == board['id']]
                for sub_board in sub_boards:
                    lines.extend(render_board(sub_board, level + 1))

                return lines

            for root_board in root_boards:
                md_lines.extend(render_board(root_board))

        if include_images:
            md_lines.append("\n## All Images\n")
            images = self.get_all_images(limit=10000)

            for img in images:
                md_lines.append(f"### 🖼️ {img['filename']}")
                md_lines.append(f"- **ID:** {img['id']}")
                md_lines.append(f"- **Path:** `{img['filepath']}`")

                if img['description']:
                    md_lines.append(f"- **Description:** {img['description']}")

                if img['tags']:
                    md_lines.append(f"- **Tags:** {', '.join(img['tags'])}")

                md_lines.append(f"- **Favorite:** {'⭐ Yes' if img['is_favorite'] else 'No'}")
                md_lines.append(f"- **Analyzed:** {'✅ Yes' if img['analyzed_at'] else '❌ No'}")

                # Show which boards contain this image
                img_boards = self.get_image_boards(img['id'])
                if img_boards:
                    board_names = [b['name'] for b in img_boards]
                    md_lines.append(f"- **Boards:** {', '.join(board_names)}")

                md_lines.append("")

        md_lines.append("\n## Tags\n")
        tags = self.get_all_tags()
        md_lines.append("| Tag | Usage Count |")
        md_lines.append("|-----|-------------|")
        for tag_info in tags[:50]:  # Top 50 tags
            md_lines.append(f"| {tag_info['tag']} | {tag_info['count']} |")

        return "\n".join(md_lines)

    def export_data_csv(self) -> Dict[str, str]:
        """
        Export data as CSV format (separate CSV for images and boards)

        Returns:
            Dictionary with keys 'images_csv', 'boards_csv', 'board_images_csv'
        """
        import csv
        from io import StringIO

        result = {}

        # Export images
        images = self.get_all_images(limit=10000)
        if images:
            output = StringIO()
            fieldnames = ['id', 'filename', 'filepath', 'description', 'tags', 'width', 'height',
                         'file_size', 'is_favorite', 'analyzed_at', 'created_at']
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for img in images:
                row = {k: img.get(k, '') for k in fieldnames}
                # Convert tags list to string
                row['tags'] = ', '.join(img['tags']) if img['tags'] else ''
                writer.writerow(row)

            result['images_csv'] = output.getvalue()

        # Export boards
        boards = self.get_all_boards()
        if boards:
            output = StringIO()
            fieldnames = ['id', 'name', 'description', 'parent_id', 'image_count', 'created_at']
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for board in boards:
                row = {k: board.get(k, '') for k in fieldnames}
                # Add image count
                board_images = self.get_board_images(board['id'])
                row['image_count'] = len(board_images)
                writer.writerow(row)

            result['boards_csv'] = output.getvalue()

        # Export board-image relationships
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT board_id, image_id, added_at FROM board_images")
        relationships = cursor.fetchall()
        conn.close()

        if relationships:
            output = StringIO()
            fieldnames = ['board_id', 'image_id', 'added_at']
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for rel in relationships:
                writer.writerow(dict(rel))

            result['board_images_csv'] = output.getvalue()

        return result

    def import_data_json(self, json_data: str,
                        import_boards: bool = True,
                        import_board_assignments: bool = True,
                        update_existing: bool = False) -> Dict:
        """
        Import data from JSON export

        Args:
            json_data: JSON string from export_data_json()
            import_boards: Import board structure
            import_board_assignments: Import which images belong to which boards
            update_existing: Update existing images/boards or skip them

        Returns:
            Dictionary with import statistics
        """
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            return {'error': f'Invalid JSON: {str(e)}', 'success': False}

        stats = {
            'boards_created': 0,
            'boards_updated': 0,
            'images_updated': 0,
            'board_assignments_created': 0,
            'errors': []
        }

        conn = self.get_connection()
        cursor = conn.cursor()

        # Import boards first (to establish hierarchy)
        if import_boards and 'boards' in data:
            # Map old board IDs to new board IDs
            board_id_map = {}

            # First pass: create all boards without parent relationships
            boards_to_import = data['boards']

            # Sort by parent_id (None first, then by ID) to create parents before children
            boards_sorted = sorted(boards_to_import,
                                  key=lambda b: (b.get('parent_id') is not None, b.get('parent_id') or 0))

            for board_data in boards_sorted:
                board_name = board_data.get('name')
                board_desc = board_data.get('description')
                old_parent_id = board_data.get('parent_id')

                # Map parent ID if it exists
                new_parent_id = board_id_map.get(old_parent_id) if old_parent_id else None

                # Check if board already exists
                cursor.execute("SELECT id FROM boards WHERE name = ?", (board_name,))
                existing = cursor.fetchone()

                if existing:
                    if update_existing:
                        board_id = existing['id']
                        cursor.execute("""
                            UPDATE boards SET description = ?, parent_id = ?, updated_at = ?
                            WHERE id = ?
                        """, (board_desc, new_parent_id, datetime.now(), board_id))
                        stats['boards_updated'] += 1
                        board_id_map[board_data['id']] = board_id
                    else:
                        # Skip, but still map the ID
                        board_id_map[board_data['id']] = existing['id']
                else:
                    # Create new board
                    cursor.execute("""
                        INSERT INTO boards (name, description, parent_id)
                        VALUES (?, ?, ?)
                    """, (board_name, board_desc, new_parent_id))
                    new_board_id = cursor.lastrowid
                    board_id_map[board_data['id']] = new_board_id
                    stats['boards_created'] += 1

            conn.commit()

            # Import board-image assignments
            if import_board_assignments:
                for board_data in boards_to_import:
                    if 'image_ids' in board_data:
                        old_board_id = board_data['id']
                        new_board_id = board_id_map.get(old_board_id)

                        if new_board_id:
                            for image_id in board_data['image_ids']:
                                try:
                                    cursor.execute("""
                                        INSERT INTO board_images (board_id, image_id)
                                        VALUES (?, ?)
                                    """, (new_board_id, image_id))
                                    stats['board_assignments_created'] += 1
                                except sqlite3.IntegrityError:
                                    # Already exists, skip
                                    pass

                conn.commit()

        # Note: We don't import image files themselves, only update metadata
        # Images must already exist in the database
        if 'images' in data and update_existing:
            for img_data in data['images']:
                image_id = img_data.get('id')
                description = img_data.get('description')
                tags = img_data.get('tags', [])
                is_favorite = img_data.get('is_favorite', False)

                # Check if image exists
                cursor.execute("SELECT id FROM images WHERE id = ?", (image_id,))
                if cursor.fetchone():
                    # Update metadata
                    cursor.execute("""
                        UPDATE images
                        SET description = ?, tags = ?, is_favorite = ?, updated_at = ?
                        WHERE id = ?
                    """, (description, json.dumps(tags), is_favorite, datetime.now(), image_id))

                    # Update FTS
                    cursor.execute("""
                        UPDATE images_fts
                        SET description = ?, tags = ?
                        WHERE rowid = ?
                    """, (description, ' '.join(tags), image_id))

                    stats['images_updated'] += 1

            conn.commit()

        conn.close()

        stats['success'] = True
        return stats

    # ============ ANNOTATION OPERATIONS (Research & Education) ============

    def add_annotation(self, image_id: int, class_name: str, x: float, y: float,
                      width: float, height: float, class_id: int = None,
                      confidence: float = 1.0, notes: str = None) -> int:
        """Add bounding box annotation to image"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO annotations (image_id, class_name, class_id, x, y, width, height, confidence, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (image_id, class_name, class_id, x, y, width, height, confidence, notes))

        annotation_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return annotation_id

    def get_annotations(self, image_id: int) -> List[Dict]:
        """Get all annotations for an image"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, image_id, class_name, class_id, x, y, width, height, confidence, notes, created_at
            FROM annotations
            WHERE image_id = ?
            ORDER BY created_at DESC
        """, (image_id,))

        annotations = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return annotations

    def update_annotation(self, annotation_id: int, class_name: str = None, x: float = None,
                         y: float = None, width: float = None, height: float = None,
                         class_id: int = None, confidence: float = None, notes: str = None):
        """Update an annotation"""
        conn = self.get_connection()
        cursor = conn.cursor()

        updates = []
        params = []

        if class_name is not None:
            updates.append("class_name = ?")
            params.append(class_name)
        if class_id is not None:
            updates.append("class_id = ?")
            params.append(class_id)
        if x is not None:
            updates.append("x = ?")
            params.append(x)
        if y is not None:
            updates.append("y = ?")
            params.append(y)
        if width is not None:
            updates.append("width = ?")
            params.append(width)
        if height is not None:
            updates.append("height = ?")
            params.append(height)
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)

        if updates:
            updates.append("updated_at = ?")
            params.append(datetime.now())
            params.append(annotation_id)

            query = f"UPDATE annotations SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            conn.commit()

        conn.close()

    def delete_annotation(self, annotation_id: int):
        """Delete an annotation"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
        conn.commit()
        conn.close()

    def delete_annotations_for_image(self, image_id: int):
        """Delete all annotations for an image"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM annotations WHERE image_id = ?", (image_id,))
        conn.commit()
        conn.close()

    def get_all_annotations(self) -> List[Dict]:
        """Get all annotations across all images"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT a.*, i.filepath, i.filename, i.width as img_width, i.height as img_height
            FROM annotations a
            JOIN images i ON a.image_id = i.id
            ORDER BY a.created_at DESC
        """)

        annotations = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return annotations

    def get_dataset_statistics(self) -> Dict:
        """Get statistics about annotated dataset"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Total images with annotations
        cursor.execute("""
            SELECT COUNT(DISTINCT image_id) as annotated_images
            FROM annotations
        """)
        annotated_images = cursor.fetchone()['annotated_images']

        # Total annotations
        cursor.execute("SELECT COUNT(*) as total_annotations FROM annotations")
        total_annotations = cursor.fetchone()['total_annotations']

        # Annotations per class
        cursor.execute("""
            SELECT class_name, COUNT(*) as count
            FROM annotations
            GROUP BY class_name
            ORDER BY count DESC
        """)
        class_distribution = [dict(row) for row in cursor.fetchall()]

        # Total images
        cursor.execute("SELECT COUNT(*) as total_images FROM images")
        total_images = cursor.fetchone()['total_images']

        conn.close()

        return {
            'total_images': total_images,
            'annotated_images': annotated_images,
            'unannotated_images': total_images - annotated_images,
            'total_annotations': total_annotations,
            'class_distribution': class_distribution
        }

    # ============ DATASET CLASS OPERATIONS ============

    def add_dataset_class(self, name: str, color: str = '#FF5722', description: str = None) -> int:
        """Add a dataset class/category"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO dataset_classes (name, color, description)
                VALUES (?, ?, ?)
            """, (name, color, description))

            class_id = cursor.lastrowid
            conn.commit()
            conn.close()

            return class_id
        except sqlite3.IntegrityError:
            # Class already exists
            cursor.execute("SELECT id FROM dataset_classes WHERE name = ?", (name,))
            class_id = cursor.fetchone()['id']
            conn.close()
            return class_id

    def get_dataset_classes(self) -> List[Dict]:
        """Get all dataset classes"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, color, description, created_at
            FROM dataset_classes
            ORDER BY name
        """)

        classes = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return classes

    def update_dataset_class(self, class_id: int, name: str = None, color: str = None, description: str = None):
        """Update a dataset class"""
        conn = self.get_connection()
        cursor = conn.cursor()

        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if color is not None:
            updates.append("color = ?")
            params.append(color)
        if description is not None:
            updates.append("description = ?")
            params.append(description)

        if updates:
            params.append(class_id)
            query = f"UPDATE dataset_classes SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            conn.commit()

        conn.close()

    def delete_dataset_class(self, class_id: int):
        """Delete a dataset class"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM dataset_classes WHERE id = ?", (class_id,))
        conn.commit()
        conn.close()

    # ============ WORKFLOW PIPELINE OPERATIONS ============

    def create_pipeline(self, name: str, description: str, trigger_type: str,
                       trigger_config: Dict, actions: List[Dict], enabled: bool = True) -> int:
        """Create a new workflow pipeline"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO pipelines (name, description, trigger_type, trigger_config, actions, enabled)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, description, trigger_type, json.dumps(trigger_config), json.dumps(actions), enabled))

        pipeline_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return pipeline_id

    def get_pipeline(self, pipeline_id: int) -> Optional[Dict]:
        """Get pipeline by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, description, trigger_type, trigger_config, actions, enabled,
                   last_run, run_count, created_at, updated_at
            FROM pipelines
            WHERE id = ?
        """, (pipeline_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            pipeline = dict(row)
            pipeline['trigger_config'] = json.loads(pipeline['trigger_config']) if pipeline['trigger_config'] else {}
            pipeline['actions'] = json.loads(pipeline['actions']) if pipeline['actions'] else []
            return pipeline

        return None

    def get_all_pipelines(self, enabled_only: bool = False) -> List[Dict]:
        """Get all pipelines"""
        conn = self.get_connection()
        cursor = conn.cursor()

        if enabled_only:
            cursor.execute("""
                SELECT id, name, description, trigger_type, trigger_config, actions, enabled,
                       last_run, run_count, created_at, updated_at
                FROM pipelines
                WHERE enabled = 1
                ORDER BY created_at DESC
            """)
        else:
            cursor.execute("""
                SELECT id, name, description, trigger_type, trigger_config, actions, enabled,
                       last_run, run_count, created_at, updated_at
                FROM pipelines
                ORDER BY created_at DESC
            """)

        pipelines = []
        for row in cursor.fetchall():
            pipeline = dict(row)
            pipeline['trigger_config'] = json.loads(pipeline['trigger_config']) if pipeline['trigger_config'] else {}
            pipeline['actions'] = json.loads(pipeline['actions']) if pipeline['actions'] else []
            pipelines.append(pipeline)

        conn.close()
        return pipelines

    def update_pipeline(self, pipeline_id: int, name: str = None, description: str = None,
                       trigger_type: str = None, trigger_config: Dict = None,
                       actions: List[Dict] = None, enabled: bool = None):
        """Update a pipeline"""
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
        if trigger_type is not None:
            updates.append("trigger_type = ?")
            params.append(trigger_type)
        if trigger_config is not None:
            updates.append("trigger_config = ?")
            params.append(json.dumps(trigger_config))
        if actions is not None:
            updates.append("actions = ?")
            params.append(json.dumps(actions))
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(enabled)

        if updates:
            updates.append("updated_at = ?")
            params.append(datetime.now())
            params.append(pipeline_id)

            query = f"UPDATE pipelines SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            conn.commit()

        conn.close()

    def delete_pipeline(self, pipeline_id: int):
        """Delete a pipeline"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM pipelines WHERE id = ?", (pipeline_id,))
        conn.commit()
        conn.close()

    def update_pipeline_stats(self, pipeline_id: int):
        """Update pipeline last_run and run_count"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE pipelines
            SET last_run = ?, run_count = run_count + 1
            WHERE id = ?
        """, (datetime.now(), pipeline_id))

        conn.commit()
        conn.close()

    def get_pipelines_by_trigger(self, trigger_type: str) -> List[Dict]:
        """Get all enabled pipelines for a specific trigger type"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, description, trigger_type, trigger_config, actions, enabled,
                   last_run, run_count, created_at, updated_at
            FROM pipelines
            WHERE trigger_type = ? AND enabled = 1
            ORDER BY created_at DESC
        """, (trigger_type,))

        pipelines = []
        for row in cursor.fetchall():
            pipeline = dict(row)
            pipeline['trigger_config'] = json.loads(pipeline['trigger_config']) if pipeline['trigger_config'] else {}
            pipeline['actions'] = json.loads(pipeline['actions']) if pipeline['actions'] else []
            pipelines.append(pipeline)

        conn.close()
        return pipelines

    # ============ PIPELINE EXECUTION LOGS ============

    def create_execution_log(self, pipeline_id: int, trigger_source: str = None) -> int:
        """Create a new execution log entry"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO pipeline_executions (pipeline_id, trigger_source, status, execution_log)
            VALUES (?, ?, 'running', '[]')
        """, (pipeline_id, trigger_source))

        execution_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return execution_id

    def update_execution_log(self, execution_id: int, status: str = None,
                            total_actions: int = None, completed_actions: int = None,
                            failed_actions: int = None, execution_log: List[Dict] = None,
                            error_message: str = None):
        """Update execution log"""
        conn = self.get_connection()
        cursor = conn.cursor()

        updates = []
        params = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)

            # Set completed_at if status is completed or failed
            if status in ['completed', 'failed']:
                updates.append("completed_at = ?")
                params.append(datetime.now())

        if total_actions is not None:
            updates.append("total_actions = ?")
            params.append(total_actions)
        if completed_actions is not None:
            updates.append("completed_actions = ?")
            params.append(completed_actions)
        if failed_actions is not None:
            updates.append("failed_actions = ?")
            params.append(failed_actions)
        if execution_log is not None:
            updates.append("execution_log = ?")
            params.append(json.dumps(execution_log))
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        if updates:
            params.append(execution_id)
            query = f"UPDATE pipeline_executions SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            conn.commit()

        conn.close()

    def get_execution_log(self, execution_id: int) -> Optional[Dict]:
        """Get execution log by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM pipeline_executions WHERE id = ?
        """, (execution_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            log = dict(row)
            log['execution_log'] = json.loads(log['execution_log']) if log['execution_log'] else []
            return log

        return None

    def get_pipeline_execution_history(self, pipeline_id: int, limit: int = 50) -> List[Dict]:
        """Get execution history for a pipeline"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM pipeline_executions
            WHERE pipeline_id = ?
            ORDER BY started_at DESC
            LIMIT ?
        """, (pipeline_id, limit))

        logs = []
        for row in cursor.fetchall():
            log = dict(row)
            log['execution_log'] = json.loads(log['execution_log']) if log['execution_log'] else []
            logs.append(log)

        conn.close()
        return logs

    def get_recent_executions(self, limit: int = 100) -> List[Dict]:
        """Get recent pipeline executions across all pipelines"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT pe.*, p.name as pipeline_name
            FROM pipeline_executions pe
            JOIN pipelines p ON pe.pipeline_id = p.id
            ORDER BY pe.started_at DESC
            LIMIT ?
        """, (limit,))

        logs = []
        for row in cursor.fetchall():
            log = dict(row)
            log['execution_log'] = json.loads(log['execution_log']) if log['execution_log'] else []
            logs.append(log)

        conn.close()
        return logs

    # ============ SETTINGS OPERATIONS ============

    def get_setting(self, key: str) -> Optional[str]:
        """Get a setting value by key"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        result = cursor.fetchone()

        conn.close()
        return result['value'] if result else None

    def set_setting(self, key: str, value: str):
        """Set a setting value"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO settings (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (key, value))

        conn.commit()
        conn.close()

    def get_all_settings(self) -> Dict[str, str]:
        """Get all settings as a dictionary"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT key, value FROM settings")
        settings = {row['key']: row['value'] for row in cursor.fetchall()}

        conn.close()
        return settings