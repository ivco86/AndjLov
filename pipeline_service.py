"""
Pipeline Service - Workflow Automation
Handles pipeline execution, action registry, and triggers
"""

import json
import threading
import time
from datetime import datetime
from typing import List, Dict, Optional, Callable
from pathlib import Path


class ActionRegistry:
    """Registry for all available pipeline actions"""

    def __init__(self, db, ai_service=None, privacy_service=None, research_service=None):
        self.db = db
        self.ai_service = ai_service
        self.privacy_service = privacy_service
        self.research_service = research_service
        self.actions = {}
        self._register_built_in_actions()

    def _register_built_in_actions(self):
        """Register all built-in actions"""

        # AI Analysis Actions
        self.register_action(
            'ai_analyze',
            'Analyze image with AI',
            self._action_ai_analyze,
            {'style': 'classic'}
        )

        # Tagging Actions
        self.register_action(
            'add_tags',
            'Add tags to image',
            self._action_add_tags,
            {'tags': []}
        )

        self.register_action(
            'remove_tags',
            'Remove tags from image',
            self._action_remove_tags,
            {'tags': []}
        )

        # Board Actions
        self.register_action(
            'add_to_board',
            'Add image to board',
            self._action_add_to_board,
            {'board_id': None, 'board_name': None}
        )

        self.register_action(
            'remove_from_board',
            'Remove image from board',
            self._action_remove_from_board,
            {'board_id': None}
        )

        self.register_action(
            'create_board',
            'Create new board',
            self._action_create_board,
            {'name': '', 'description': ''}
        )

        # Privacy Actions
        self.register_action(
            'privacy_analyze',
            'Analyze image for privacy concerns',
            self._action_privacy_analyze,
            {}
        )

        # Favorite Action
        self.register_action(
            'mark_favorite',
            'Mark image as favorite',
            self._action_mark_favorite,
            {}
        )

        # Research Actions
        self.register_action(
            'generate_citation',
            'Generate citation',
            self._action_generate_citation,
            {'format': 'apa'}
        )

        self.register_action(
            'create_annotation',
            'Create annotation',
            self._action_create_annotation,
            {'class_name': '', 'x': 0, 'y': 0, 'width': 0, 'height': 0}
        )

        # Notification Action
        self.register_action(
            'log_message',
            'Log a message',
            self._action_log_message,
            {'message': ''}
        )

    def register_action(self, action_type: str, description: str,
                       handler: Callable, default_params: Dict):
        """Register a new action type"""
        self.actions[action_type] = {
            'description': description,
            'handler': handler,
            'default_params': default_params
        }

    def execute_action(self, action_type: str, image_id: int, params: Dict, context: Dict) -> Dict:
        """Execute an action"""
        if action_type not in self.actions:
            return {
                'success': False,
                'error': f'Unknown action type: {action_type}'
            }

        action_info = self.actions[action_type]
        handler = action_info['handler']

        # Merge default params with provided params
        merged_params = {**action_info['default_params'], **params}

        try:
            result = handler(image_id, merged_params, context)
            return {
                'success': True,
                'result': result
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_available_actions(self) -> List[Dict]:
        """Get list of all available actions"""
        return [
            {
                'type': action_type,
                'description': info['description'],
                'default_params': info['default_params']
            }
            for action_type, info in self.actions.items()
        ]

    # ============ BUILT-IN ACTION HANDLERS ============

    def _action_ai_analyze(self, image_id: int, params: Dict, context: Dict) -> str:
        """Analyze image with AI"""
        if not self.ai_service:
            raise Exception("AI service not available")

        image = self.db.get_image(image_id)
        if not image:
            raise Exception(f"Image {image_id} not found")

        result = self.ai_service.analyze_image(image['filepath'], style=params.get('style', 'classic'))

        if result:
            self.db.update_image_analysis(image_id, result['description'], result['tags'])
            return f"Analyzed: {result['description'][:50]}..."

        raise Exception("AI analysis failed")

    def _action_add_tags(self, image_id: int, params: Dict, context: Dict) -> str:
        """Add tags to image"""
        tags_to_add = params.get('tags', [])
        if not tags_to_add:
            return "No tags to add"

        image = self.db.get_image(image_id)
        if not image:
            raise Exception(f"Image {image_id} not found")

        existing_tags = json.loads(image.get('tags', '[]')) if image.get('tags') else []
        new_tags = list(set(existing_tags + tags_to_add))

        self.db.update_image(image_id, tags=new_tags)
        return f"Added tags: {', '.join(tags_to_add)}"

    def _action_remove_tags(self, image_id: int, params: Dict, context: Dict) -> str:
        """Remove tags from image"""
        tags_to_remove = params.get('tags', [])
        if not tags_to_remove:
            return "No tags to remove"

        image = self.db.get_image(image_id)
        if not image:
            raise Exception(f"Image {image_id} not found")

        existing_tags = json.loads(image.get('tags', '[]')) if image.get('tags') else []
        new_tags = [tag for tag in existing_tags if tag not in tags_to_remove]

        self.db.update_image(image_id, tags=new_tags)
        return f"Removed tags: {', '.join(tags_to_remove)}"

    def _action_add_to_board(self, image_id: int, params: Dict, context: Dict) -> str:
        """Add image to board"""
        board_id = params.get('board_id')
        board_name = params.get('board_name')

        # Find or create board
        if not board_id and board_name:
            # Try to find board by name
            boards = self.db.get_all_boards()
            board = next((b for b in boards if b['name'] == board_name), None)

            if board:
                board_id = board['id']
            else:
                # Create new board
                board_id = self.db.create_board(board_name, '')

        if not board_id:
            raise Exception("Board ID or name required")

        self.db.add_image_to_board(board_id, image_id)
        return f"Added to board {board_id}"

    def _action_remove_from_board(self, image_id: int, params: Dict, context: Dict) -> str:
        """Remove image from board"""
        board_id = params.get('board_id')

        if not board_id:
            raise Exception("Board ID required")

        self.db.remove_image_from_board(board_id, image_id)
        return f"Removed from board {board_id}"

    def _action_create_board(self, image_id: int, params: Dict, context: Dict) -> str:
        """Create a new board"""
        name = params.get('name')
        description = params.get('description', '')

        if not name:
            raise Exception("Board name required")

        board_id = self.db.create_board(name, description)
        context['created_board_id'] = board_id

        return f"Created board: {name} (ID: {board_id})"

    def _action_privacy_analyze(self, image_id: int, params: Dict, context: Dict) -> str:
        """Analyze image for privacy"""
        if not self.privacy_service:
            raise Exception("Privacy service not available")

        image = self.db.get_image(image_id)
        if not image:
            raise Exception(f"Image {image_id} not found")

        result = self.privacy_service.analyze_image_privacy(image['filepath'])

        self.db.update_privacy_analysis(
            image_id=image_id,
            has_faces=result['has_faces'],
            has_plates=result['has_plates'],
            is_nsfw=result['is_nsfw'],
            privacy_zones=result['privacy_zones']
        )

        detected = []
        if result['has_faces']:
            detected.append('faces')
        if result['has_plates']:
            detected.append('plates')

        return f"Privacy scan: {', '.join(detected) if detected else 'nothing detected'}"

    def _action_mark_favorite(self, image_id: int, params: Dict, context: Dict) -> str:
        """Mark image as favorite"""
        self.db.update_image(image_id, is_favorite=True)
        return "Marked as favorite"

    def _action_generate_citation(self, image_id: int, params: Dict, context: Dict) -> str:
        """Generate citation"""
        if not self.research_service:
            raise Exception("Research service not available")

        image = self.db.get_image(image_id)
        if not image:
            raise Exception(f"Image {image_id} not found")

        format_type = params.get('format', 'apa')
        citation = self.research_service.generate_citation(image, format_type)

        return f"Generated {format_type.upper()} citation"

    def _action_create_annotation(self, image_id: int, params: Dict, context: Dict) -> str:
        """Create annotation"""
        class_name = params.get('class_name')
        if not class_name:
            raise Exception("Class name required")

        annotation_id = self.db.add_annotation(
            image_id=image_id,
            class_name=class_name,
            x=params.get('x', 0),
            y=params.get('y', 0),
            width=params.get('width', 100),
            height=params.get('height', 100)
        )

        return f"Created annotation: {class_name}"

    def _action_log_message(self, image_id: int, params: Dict, context: Dict) -> str:
        """Log a message"""
        message = params.get('message', 'No message')
        print(f"[Pipeline Log] Image {image_id}: {message}")
        return message


class PipelineExecutor:
    """Executes pipelines and manages execution state"""

    def __init__(self, db, action_registry: ActionRegistry):
        self.db = db
        self.action_registry = action_registry

    def execute_pipeline(self, pipeline_id: int, image_ids: List[int],
                        trigger_source: str = 'manual') -> Dict:
        """Execute a pipeline on a list of images"""
        pipeline = self.db.get_pipeline(pipeline_id)
        if not pipeline:
            return {'success': False, 'error': 'Pipeline not found'}

        # Create execution log
        execution_id = self.db.create_execution_log(pipeline_id, trigger_source)

        execution_logs = []
        total_images = len(image_ids)
        successful_images = 0
        failed_images = 0

        actions = pipeline['actions']
        total_actions = len(actions) * total_images

        self.db.update_execution_log(
            execution_id,
            total_actions=total_actions
        )

        # Execute pipeline for each image
        for image_id in image_ids:
            image_log = {
                'image_id': image_id,
                'actions': []
            }

            context = {}  # Shared context between actions
            image_failed = False

            # Execute each action
            for action in actions:
                action_type = action.get('type')
                action_params = action.get('params', {})

                action_result = self.action_registry.execute_action(
                    action_type,
                    image_id,
                    action_params,
                    context
                )

                image_log['actions'].append({
                    'type': action_type,
                    'success': action_result.get('success'),
                    'result': action_result.get('result'),
                    'error': action_result.get('error')
                })

                if not action_result.get('success'):
                    image_failed = True
                    failed_images += 1
                    break  # Stop processing this image on error

            if not image_failed:
                successful_images += 1

            execution_logs.append(image_log)

            # Update progress
            completed = successful_images * len(actions) + failed_images * len(actions)
            self.db.update_execution_log(
                execution_id,
                completed_actions=completed,
                failed_actions=failed_images,
                execution_log=execution_logs
            )

        # Mark execution as completed
        status = 'completed' if failed_images == 0 else 'completed_with_errors'

        self.db.update_execution_log(
            execution_id,
            status=status,
            completed_actions=total_actions,
            failed_actions=failed_images
        )

        # Update pipeline stats
        self.db.update_pipeline_stats(pipeline_id)

        return {
            'success': True,
            'execution_id': execution_id,
            'total_images': total_images,
            'successful': successful_images,
            'failed': failed_images,
            'status': status
        }


class PipelineTemplates:
    """Pre-built pipeline templates"""

    @staticmethod
    def get_all_templates() -> List[Dict]:
        """Get all available templates"""
        return [
            {
                'id': 'auto_organize',
                'name': 'Auto-Organize New Images',
                'description': 'Automatically analyze and tag new images',
                'trigger_type': 'on_scan',
                'trigger_config': {},
                'actions': [
                    {'type': 'ai_analyze', 'params': {'style': 'classic'}},
                    {'type': 'privacy_analyze', 'params': {}}
                ]
            },
            {
                'id': 'research_workflow',
                'name': 'Research Assistant',
                'description': 'Analyze, generate citations, and prepare for research',
                'trigger_type': 'manual',
                'trigger_config': {},
                'actions': [
                    {'type': 'ai_analyze', 'params': {'style': 'detailed'}},
                    {'type': 'generate_citation', 'params': {'format': 'apa'}},
                    {'type': 'add_tags', 'params': {'tags': ['research']}}
                ]
            },
            {
                'id': 'privacy_first',
                'name': 'Privacy-First Scanning',
                'description': 'Detect and protect sensitive content',
                'trigger_type': 'on_scan',
                'trigger_config': {},
                'actions': [
                    {'type': 'privacy_analyze', 'params': {}},
                    {'type': 'add_tags', 'params': {'tags': ['scanned']}}
                ]
            },
            {
                'id': 'favorites_backup',
                'name': 'Favorites to Board',
                'description': 'Add favorite images to a special board',
                'trigger_type': 'manual',
                'trigger_config': {},
                'actions': [
                    {'type': 'mark_favorite', 'params': {}},
                    {'type': 'add_to_board', 'params': {'board_name': 'Favorites'}}
                ]
            }
        ]

    @staticmethod
    def create_from_template(template_id: str, db) -> Optional[int]:
        """Create a pipeline from a template"""
        templates = PipelineTemplates.get_all_templates()
        template = next((t for t in templates if t['id'] == template_id), None)

        if not template:
            return None

        pipeline_id = db.create_pipeline(
            name=template['name'],
            description=template['description'],
            trigger_type=template['trigger_type'],
            trigger_config=template['trigger_config'],
            actions=template['actions'],
            enabled=True
        )

        return pipeline_id


# Global instance
_pipeline_service = None


def get_pipeline_service(db=None, ai_service=None, privacy_service=None, research_service=None):
    """Get or create pipeline service singleton"""
    global _pipeline_service

    if _pipeline_service is None and db:
        action_registry = ActionRegistry(db, ai_service, privacy_service, research_service)
        executor = PipelineExecutor(db, action_registry)

        _pipeline_service = {
            'actions': action_registry,
            'executor': executor,
            'templates': PipelineTemplates()
        }

    return _pipeline_service
