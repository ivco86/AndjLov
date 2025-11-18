"""
Cloud Sync Service - Backup and sync to cloud storage
Supports local backup and rclone integration for cloud providers
"""

import os
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import zipfile


class CloudSyncService:
    """Service for backing up and syncing to cloud storage"""

    def __init__(self, db, data_dir='data', backup_dir='data/backups'):
        self.db = db
        self.data_dir = Path(data_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Check if rclone is available
        self.has_rclone = self._check_rclone()

    def _check_rclone(self) -> bool:
        """Check if rclone is installed"""
        try:
            result = subprocess.run(['rclone', 'version'],
                                  capture_output=True,
                                  timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def create_backup(self, include_images: bool = True) -> Dict:
        """Create a local backup of database and optionally images"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f'backup_{timestamp}'
            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(parents=True, exist_ok=True)

            # Backup database
            db_path = Path(self.db.db_path)
            if db_path.exists():
                shutil.copy2(db_path, backup_path / 'gallery.db')

            # Backup config files
            for config_file in ['.env', 'config.json']:
                config_path = Path(config_file)
                if config_path.exists():
                    shutil.copy2(config_path, backup_path / config_file)

            # Backup images if requested
            if include_images:
                images_backup = backup_path / 'images'
                images_backup.mkdir(exist_ok=True)

                # Get all image paths
                images = self.db.get_all_images(limit=100000)
                for img in images:
                    img_path = Path(img['filepath'])
                    if img_path.exists():
                        # Create subdirectory structure
                        rel_path = img_path.relative_to(Path.cwd()) if img_path.is_absolute() else img_path
                        dest_path = images_backup / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)

                        try:
                            shutil.copy2(img_path, dest_path)
                        except Exception as e:
                            print(f"Warning: Could not backup {img_path}: {e}")

            # Create ZIP archive
            zip_path = self.backup_dir / f'{backup_name}.zip'
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(backup_path):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(backup_path)
                        zipf.write(file_path, arcname)

            # Remove uncompressed backup folder
            shutil.rmtree(backup_path)

            # Save backup metadata
            backup_info = {
                'timestamp': timestamp,
                'name': backup_name,
                'file': str(zip_path),
                'size': zip_path.stat().st_size,
                'include_images': include_images,
                'image_count': len(images) if include_images else 0
            }

            self._save_backup_info(backup_info)

            return {
                'success': True,
                'backup_file': str(zip_path),
                'size': zip_path.stat().st_size,
                'name': backup_name
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _save_backup_info(self, backup_info: Dict):
        """Save backup metadata to database"""
        backups = self.get_backups_list()
        backups.append(backup_info)

        # Keep only last 50 backups in metadata
        backups = backups[-50:]

        self.db.set_setting('backups_list', json.dumps(backups))

    def get_backups_list(self) -> List[Dict]:
        """Get list of available backups"""
        backups_str = self.db.get_setting('backups_list')
        if backups_str:
            return json.loads(backups_str)
        return []

    def delete_backup(self, backup_name: str) -> Dict:
        """Delete a backup file"""
        try:
            zip_path = self.backup_dir / f'{backup_name}.zip'
            if zip_path.exists():
                zip_path.unlink()

            # Remove from metadata
            backups = self.get_backups_list()
            backups = [b for b in backups if b['name'] != backup_name]
            self.db.set_setting('backups_list', json.dumps(backups))

            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def sync_to_cloud(self, remote: str, path: str = None) -> Dict:
        """Sync backup to cloud using rclone"""
        if not self.has_rclone:
            return {
                'success': False,
                'error': 'rclone not installed. Install from https://rclone.org'
            }

        try:
            # Use backup directory if no path specified
            source = path or str(self.backup_dir)
            destination = f'{remote}:AI-Gallery-Backups'

            # Run rclone sync
            result = subprocess.run(
                ['rclone', 'sync', source, destination, '--progress'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode == 0:
                return {
                    'success': True,
                    'message': 'Sync completed successfully',
                    'output': result.stdout
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr or 'Sync failed'
                }

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Sync timeout (>5 minutes)'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def list_rclone_remotes(self) -> Dict:
        """List configured rclone remotes"""
        if not self.has_rclone:
            return {
                'success': False,
                'error': 'rclone not installed'
            }

        try:
            result = subprocess.run(
                ['rclone', 'listremotes'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                remotes = [r.strip().rstrip(':') for r in result.stdout.split('\n') if r.strip()]
                return {
                    'success': True,
                    'remotes': remotes,
                    'count': len(remotes)
                }
            else:
                return {'success': False, 'error': result.stderr}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_sync_config(self) -> Dict:
        """Get cloud sync configuration"""
        config_str = self.db.get_setting('cloud_sync_config')
        if config_str:
            return json.loads(config_str)
        return {
            'enabled': False,
            'auto_backup': False,
            'backup_interval_hours': 24,
            'include_images': True,
            'rclone_remote': None,
            'last_backup': None,
            'last_sync': None
        }

    def save_sync_config(self, config: Dict) -> Dict:
        """Save cloud sync configuration"""
        try:
            self.db.set_setting('cloud_sync_config', json.dumps(config))
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def auto_backup_if_needed(self):
        """Check if auto backup is needed and create one"""
        config = self.get_sync_config()

        if not config.get('auto_backup'):
            return

        last_backup = config.get('last_backup')
        interval_hours = config.get('backup_interval_hours', 24)

        should_backup = False
        if not last_backup:
            should_backup = True
        else:
            from datetime import datetime, timedelta
            last_backup_time = datetime.fromisoformat(last_backup)
            if datetime.now() - last_backup_time > timedelta(hours=interval_hours):
                should_backup = True

        if should_backup:
            print("🔄 Creating automatic backup...")
            result = self.create_backup(include_images=config.get('include_images', True))

            if result.get('success'):
                config['last_backup'] = datetime.now().isoformat()
                self.save_sync_config(config)

                # Auto-sync if remote configured
                if config.get('rclone_remote'):
                    print(f"☁️ Syncing to {config['rclone_remote']}...")
                    sync_result = self.sync_to_cloud(config['rclone_remote'])

                    if sync_result.get('success'):
                        config['last_sync'] = datetime.now().isoformat()
                        self.save_sync_config(config)


# Global instance
_cloud_sync_service = None


def get_cloud_sync_service(db=None, data_dir='data'):
    """Get or create cloud sync service singleton"""
    global _cloud_sync_service

    if _cloud_sync_service is None and db:
        _cloud_sync_service = CloudSyncService(db, data_dir)

    return _cloud_sync_service
