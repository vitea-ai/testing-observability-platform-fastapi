"""
Queue service for managing Celery tasks.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from celery.result import AsyncResult
from celery import states
from app.workers.celery_app import celery_app
from app.core.logging import logger


class QueueService:
    """Service for managing and monitoring queued tasks."""
    
    def __init__(self):
        self.celery_app = celery_app
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a Celery task.
        
        Args:
            task_id: The Celery task ID
            
        Returns:
            Dictionary with task status information
        """
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            
            # Get basic status
            status = result.status
            info = result.info
            
            # Build response based on status
            response = {
                "task_id": task_id,
                "status": status.lower(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if status == states.PENDING:
                # Task not found or not started yet
                response.update({
                    "message": "Task is queued or does not exist",
                    "progress": 0
                })
            elif status == states.STARTED:
                # Task is currently running
                response.update({
                    "message": "Task is processing",
                    "progress": info.get("progress", 0) if isinstance(info, dict) else 0
                })
            elif status == states.SUCCESS:
                # Task completed successfully
                response.update({
                    "message": "Task completed successfully",
                    "progress": 100,
                    "result": info
                })
            elif status == states.FAILURE:
                # Task failed
                error_info = str(info) if info else "Unknown error"
                response.update({
                    "message": "Task failed",
                    "progress": 0,
                    "error": error_info
                })
            elif status == states.RETRY:
                # Task is being retried
                response.update({
                    "message": "Task is being retried",
                    "progress": info.get("progress", 0) if isinstance(info, dict) else 0
                })
            elif status == states.REVOKED:
                # Task was cancelled
                response.update({
                    "message": "Task was cancelled",
                    "progress": 0
                })
            else:
                # Unknown status
                response.update({
                    "message": f"Task status: {status}",
                    "progress": 0,
                    "info": info
                })
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting task status for {task_id}: {e}")
            return {
                "task_id": task_id,
                "status": "error",
                "message": "Failed to get task status",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cancel_task(self, task_id: str, terminate: bool = False) -> bool:
        """
        Cancel a running or queued task.
        
        Args:
            task_id: The Celery task ID to cancel
            terminate: If True, forcefully terminate the task
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            
            # Check if task is already completed
            if result.status in [states.SUCCESS, states.FAILURE]:
                logger.warning(f"Cannot cancel completed task {task_id}")
                return False
            
            # Revoke the task
            if terminate:
                result.revoke(terminate=True, signal="SIGTERM")
                logger.info(f"Forcefully terminated task {task_id}")
            else:
                result.revoke()
                logger.info(f"Cancelled task {task_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the task queue.
        
        Returns:
            Dictionary with queue statistics
        """
        try:
            # Get queue information from Celery
            inspect = self.celery_app.control.inspect()
            
            # Get active tasks
            active = inspect.active()
            active_count = sum(len(tasks) for tasks in (active or {}).values())
            
            # Get scheduled tasks
            scheduled = inspect.scheduled()
            scheduled_count = sum(len(tasks) for tasks in (scheduled or {}).values())
            
            # Get reserved tasks
            reserved = inspect.reserved()
            reserved_count = sum(len(tasks) for tasks in (reserved or {}).values())
            
            # Get registered tasks
            registered = inspect.registered()
            registered_tasks = []
            if registered:
                for worker, tasks in registered.items():
                    registered_tasks.extend(tasks)
            registered_tasks = list(set(registered_tasks))
            
            # Get worker stats
            stats = inspect.stats()
            worker_count = len(stats) if stats else 0
            
            return {
                "active_tasks": active_count,
                "scheduled_tasks": scheduled_count,
                "reserved_tasks": reserved_count,
                "total_pending": scheduled_count + reserved_count,
                "worker_count": worker_count,
                "registered_tasks": registered_tasks,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {
                "error": "Failed to get queue statistics",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Get list of currently active tasks.
        
        Returns:
            List of active task information
        """
        try:
            inspect = self.celery_app.control.inspect()
            active = inspect.active()
            
            if not active:
                return []
            
            tasks = []
            for worker, worker_tasks in active.items():
                for task in worker_tasks:
                    tasks.append({
                        "task_id": task.get("id"),
                        "name": task.get("name"),
                        "worker": worker,
                        "args": task.get("args"),
                        "kwargs": task.get("kwargs"),
                        "time_start": task.get("time_start"),
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error getting active tasks: {e}")
            return []
    
    async def purge_queue(self, queue_name: str = "evaluations") -> int:
        """
        Purge all pending tasks from a specific queue.
        WARNING: This will delete all pending tasks!
        
        Args:
            queue_name: Name of the queue to purge
            
        Returns:
            Number of tasks purged
        """
        try:
            result = self.celery_app.control.purge()
            logger.warning(f"Purged queue {queue_name}: {result}")
            return result if isinstance(result, int) else 0
            
        except Exception as e:
            logger.error(f"Error purging queue {queue_name}: {e}")
            return 0
    
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Get the result of a completed task.
        
        Args:
            task_id: The Celery task ID
            
        Returns:
            Task result if available, None otherwise
        """
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            
            if result.status == states.SUCCESS:
                return result.result
            elif result.status == states.FAILURE:
                return {"error": str(result.info)}
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting task result for {task_id}: {e}")
            return None
    
    async def retry_task(self, task_id: str) -> Optional[str]:
        """
        Retry a failed task.
        
        Args:
            task_id: The original task ID
            
        Returns:
            New task ID if retry successful, None otherwise
        """
        try:
            # Get original task info
            result = AsyncResult(task_id, app=self.celery_app)
            
            if result.status != states.FAILURE:
                logger.warning(f"Task {task_id} is not in failed state, cannot retry")
                return None
            
            # Get task metadata
            task_name = result.name
            task_args = result.args
            task_kwargs = result.kwargs
            
            if not task_name:
                logger.error(f"Cannot determine task name for {task_id}")
                return None
            
            # Resubmit the task
            new_task = self.celery_app.send_task(
                task_name,
                args=task_args,
                kwargs=task_kwargs,
                queue="evaluations"
            )
            
            logger.info(f"Retried task {task_id} as new task {new_task.id}")
            return new_task.id
            
        except Exception as e:
            logger.error(f"Error retrying task {task_id}: {e}")
            return None