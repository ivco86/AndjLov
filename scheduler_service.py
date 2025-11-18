"""
Scheduler Service - Cron-like scheduling for pipelines
Uses APScheduler for background job scheduling
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
import threading
import time

# Try to import APScheduler
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False
    print("Warning: APScheduler not installed. Scheduled triggers will not work.")
    print("Install with: pip install apscheduler")


class SchedulerService:
    """Service for scheduling pipeline executions"""

    def __init__(self, db, pipeline_executor):
        self.db = db
        self.pipeline_executor = pipeline_executor
        self.scheduler = None
        self.scheduled_jobs = {}  # job_id -> job info

        if HAS_APSCHEDULER:
            self.scheduler = BackgroundScheduler()
            self.scheduler.start()
            print("✅ Scheduler service started")
        else:
            print("⚠️ Scheduler service disabled (APScheduler not installed)")

    def schedule_pipeline(self, pipeline_id: int, schedule_config: Dict) -> Dict:
        """Schedule a pipeline for automatic execution"""
        if not HAS_APSCHEDULER:
            return {'success': False, 'error': 'APScheduler not installed'}

        pipeline = self.db.get_pipeline(pipeline_id)
        if not pipeline:
            return {'success': False, 'error': 'Pipeline not found'}

        schedule_type = schedule_config.get('type', 'cron')
        job_id = f"pipeline_{pipeline_id}"

        # Remove existing job if any
        if job_id in self.scheduled_jobs:
            self.unschedule_pipeline(pipeline_id)

        try:
            if schedule_type == 'cron':
                # Cron-based scheduling
                trigger = CronTrigger(
                    minute=schedule_config.get('minute', '*'),
                    hour=schedule_config.get('hour', '*'),
                    day=schedule_config.get('day', '*'),
                    month=schedule_config.get('month', '*'),
                    day_of_week=schedule_config.get('day_of_week', '*')
                )
            elif schedule_type == 'interval':
                # Interval-based scheduling
                trigger = IntervalTrigger(
                    weeks=schedule_config.get('weeks', 0),
                    days=schedule_config.get('days', 0),
                    hours=schedule_config.get('hours', 0),
                    minutes=schedule_config.get('minutes', 0)
                )
            elif schedule_type == 'preset':
                # Preset schedules (daily, weekly, etc.)
                preset = schedule_config.get('preset', 'daily')
                trigger = self._get_preset_trigger(preset)
            else:
                return {'success': False, 'error': f'Unknown schedule type: {schedule_type}'}

            # Add job to scheduler
            job = self.scheduler.add_job(
                func=self._execute_scheduled_pipeline,
                trigger=trigger,
                args=[pipeline_id],
                id=job_id,
                name=f"Pipeline: {pipeline['name']}",
                replace_existing=True
            )

            self.scheduled_jobs[job_id] = {
                'pipeline_id': pipeline_id,
                'pipeline_name': pipeline['name'],
                'schedule_config': schedule_config,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None
            }

            # Update pipeline trigger config
            pipeline['trigger_config'] = json.dumps(schedule_config)
            self.db.update_pipeline(
                pipeline_id=pipeline_id,
                trigger_config=schedule_config
            )

            return {
                'success': True,
                'job_id': job_id,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def unschedule_pipeline(self, pipeline_id: int) -> Dict:
        """Remove scheduled execution for a pipeline"""
        if not HAS_APSCHEDULER:
            return {'success': False, 'error': 'APScheduler not installed'}

        job_id = f"pipeline_{pipeline_id}"

        try:
            if job_id in self.scheduled_jobs:
                self.scheduler.remove_job(job_id)
                del self.scheduled_jobs[job_id]

            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_scheduled_pipelines(self) -> List[Dict]:
        """Get all scheduled pipelines"""
        if not HAS_APSCHEDULER:
            return []

        jobs = []
        for job_id, job_info in self.scheduled_jobs.items():
            # Get job from scheduler to update next_run
            job = self.scheduler.get_job(job_id)
            if job:
                job_info['next_run'] = job.next_run_time.isoformat() if job.next_run_time else None

            jobs.append(job_info)

        return jobs

    def _execute_scheduled_pipeline(self, pipeline_id: int):
        """Execute a pipeline (called by scheduler)"""
        print(f"⏰ Executing scheduled pipeline {pipeline_id}")

        try:
            pipeline = self.db.get_pipeline(pipeline_id)
            if not pipeline:
                print(f"❌ Pipeline {pipeline_id} not found")
                return

            # Get images to process based on trigger config
            trigger_config = json.loads(pipeline.get('trigger_config', '{}'))
            image_filter = trigger_config.get('image_filter', {})

            # Get images based on filter
            if image_filter.get('type') == 'all':
                images = self.db.get_all_images(limit=10000)
            elif image_filter.get('type') == 'recent':
                days = image_filter.get('days', 7)
                images = self.db.get_recent_images(days=days)
            elif image_filter.get('type') == 'unanalyzed':
                images = self.db.get_all_images(analyzed=False, limit=10000)
            elif image_filter.get('type') == 'board':
                board_id = image_filter.get('board_id')
                images = self.db.get_board_images(board_id) if board_id else []
            else:
                # Default: all images
                images = self.db.get_all_images(limit=10000)

            image_ids = [img['id'] for img in images]

            if not image_ids:
                print(f"ℹ️ No images to process for pipeline {pipeline_id}")
                return

            # Execute pipeline
            result = self.pipeline_executor.execute_pipeline(
                pipeline_id=pipeline_id,
                image_ids=image_ids,
                trigger_source='scheduled'
            )

            print(f"✅ Scheduled pipeline {pipeline_id} completed: {result}")

        except Exception as e:
            print(f"❌ Error executing scheduled pipeline {pipeline_id}: {e}")

    def _get_preset_trigger(self, preset: str):
        """Get a cron trigger for preset schedules"""
        presets = {
            'daily': CronTrigger(hour=0, minute=0),  # Every day at midnight
            'weekly': CronTrigger(day_of_week=0, hour=0, minute=0),  # Every Monday
            'monthly': CronTrigger(day=1, hour=0, minute=0),  # 1st of month
            'hourly': CronTrigger(minute=0),  # Every hour
            'every_6_hours': CronTrigger(hour='*/6'),  # Every 6 hours
            'every_12_hours': CronTrigger(hour='*/12'),  # Every 12 hours
        }

        return presets.get(preset, CronTrigger(hour=0, minute=0))

    def reload_scheduled_pipelines(self):
        """Reload all scheduled pipelines from database"""
        if not HAS_APSCHEDULER:
            return

        # Get all pipelines with scheduled trigger
        pipelines = self.db.get_all_pipelines()

        for pipeline in pipelines:
            if pipeline['trigger_type'] == 'scheduled' and pipeline['enabled']:
                trigger_config = json.loads(pipeline.get('trigger_config', '{}'))
                if trigger_config:
                    self.schedule_pipeline(pipeline['id'], trigger_config)

        print(f"🔄 Reloaded {len(self.scheduled_jobs)} scheduled pipelines")

    def shutdown(self):
        """Shutdown the scheduler"""
        if HAS_APSCHEDULER and self.scheduler:
            self.scheduler.shutdown()
            print("🛑 Scheduler service stopped")


# Global instance
_scheduler_service = None


def get_scheduler_service(db=None, pipeline_executor=None):
    """Get or create scheduler service singleton"""
    global _scheduler_service

    if _scheduler_service is None and db and pipeline_executor:
        _scheduler_service = SchedulerService(db, pipeline_executor)

    return _scheduler_service
