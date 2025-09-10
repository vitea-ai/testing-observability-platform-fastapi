"""
Celery application configuration for async task processing.
"""

import os
from celery import Celery
from app.core.config import settings

# Create Celery instance
celery_app = Celery(
    "vitea_evaluator",
    broker=settings.get_celery_broker_url(),
    backend=settings.get_celery_result_backend(),
    include=[
        "app.workers.tasks.evaluation_tasks",
        "app.workers.tasks.experiment_runner_tasks"
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,  # Results expire after 1 hour
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    task_soft_time_limit=540,  # 9 minutes soft limit
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # Process one task at a time per worker
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks to prevent memory leaks
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    result_backend_always_retry=True,
    result_backend_max_retries=10,
)

# Task routing configuration
celery_app.conf.task_routes = {
    "app.workers.tasks.evaluation_tasks.*": {"queue": "evaluations"},
}

# Default queue configurations
celery_app.conf.task_default_queue = "default"
celery_app.conf.task_default_exchange = "default"
celery_app.conf.task_default_exchange_type = "direct"
celery_app.conf.task_default_routing_key = "default"

# Define task queues
celery_app.conf.task_queues = {
    "evaluations": {
        "exchange": "evaluations",
        "exchange_type": "direct",
        "routing_key": "evaluation.task",
    },
    "default": {
        "exchange": "default",
        "exchange_type": "direct",
        "routing_key": "default",
    },
}

# Retry configuration
celery_app.conf.task_annotations = {
    "*": {
        "rate_limit": "100/m",  # Max 100 tasks per minute
        "max_retries": 3,
        "default_retry_delay": 60,  # 1 minute between retries
    },
}

# Logging configuration
celery_app.conf.worker_log_format = "[%(asctime)s: %(levelname)s/%(processName)s] %(message)s"
celery_app.conf.worker_task_log_format = "[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s"

# Redis specific settings
celery_app.conf.redis_max_connections = settings.redis_max_connections
celery_app.conf.redis_retry_on_timeout = True
celery_app.conf.redis_socket_keepalive = True
celery_app.conf.redis_socket_keepalive_options = {
    1: 3,  # TCP_KEEPIDLE
    2: 3,  # TCP_KEEPINTVL
    3: 3,  # TCP_KEEPCNT
}