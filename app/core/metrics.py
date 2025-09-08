"""
Prometheus metrics configuration for observability.

This module sets up Prometheus metrics collection using prometheus-fastapi-instrumentator
which provides automatic HTTP metrics and exposes /metrics endpoint.
"""

from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.logging import logger

# HTTP metrics are automatically provided by prometheus-fastapi-instrumentator

# ====================
# Business Metrics - Datasets
# ====================
DATASET_UPLOADS = Counter(
    "dataset_uploads_total",
    "Total dataset uploads",
    ["status", "format"]
)

DATASET_SIZE = Histogram(
    "dataset_size_bytes",
    "Size of uploaded datasets in bytes",
    buckets=(1024, 10240, 102400, 1048576, 10485760, 104857600, 1073741824)  # 1KB to 1GB
)

DATASET_ROW_COUNT = Histogram(
    "dataset_row_count",
    "Number of rows in uploaded datasets",
    buckets=(10, 100, 500, 1000, 5000, 10000, 50000, 100000)
)

DATASET_PROCESSING_TIME = Histogram(
    "dataset_processing_seconds",
    "Time to process dataset uploads in seconds",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60)
)

# ====================
# Business Metrics - Experiments
# ====================
EXPERIMENT_RUNS = Counter(
    "experiment_runs_total",
    "Total experiment runs",
    ["status", "model"]
)

EXPERIMENT_TEST_COUNT = Histogram(
    "experiment_test_count",
    "Number of tests per experiment",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000)
)

EXPERIMENT_DURATION = Histogram(
    "experiment_duration_seconds",
    "Experiment execution time in seconds",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800)
)

EXPERIMENT_SUCCESS_RATE = Gauge(
    "experiment_success_rate",
    "Success rate of the last experiment",
    ["experiment_name"]
)

# ====================
# Business Metrics - Evaluations
# ====================
EVALUATION_COUNT = Counter(
    "evaluations_total",
    "Total evaluations performed",
    ["evaluator", "passed"]
)

EVALUATION_SCORE = Histogram(
    "evaluation_score",
    "Distribution of evaluation scores",
    ["evaluator"],
    buckets=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

EVALUATION_LATENCY = Histogram(
    "evaluation_latency_seconds",
    "Evaluation processing time in seconds",
    ["evaluator"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)
)

# ====================
# Business Metrics - Tokens/Cost
# ====================
TOKEN_USAGE = Counter(
    "tokens_used_total",
    "Total tokens consumed",
    ["model", "operation"]
)

TOKEN_COST = Counter(
    "token_cost_dollars",
    "Estimated token cost in dollars",
    ["model", "operation"]
)

# ====================
# System Metrics
# ====================
ACTIVE_DATASETS = Gauge(
    "active_datasets",
    "Number of active datasets in the system"
)

ACTIVE_EXPERIMENTS = Gauge(
    "active_experiments",
    "Number of active experiments running"
)

DATABASE_CONNECTIONS = Gauge(
    "database_connections",
    "Number of active database connections"
)


def setup_metrics(app) -> Instrumentator:
    """
    Setup Prometheus metrics using prometheus-fastapi-instrumentator.
    
    This automatically provides:
    - HTTP request metrics (count, duration, in-progress)
    - /metrics endpoint for Prometheus scraping
    - Custom business metrics
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Configured Instrumentator instance
    """
    
    # Create and configure the instrumentator
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health", "/ready"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="http_requests_in_progress",
        inprogress_labels=True,
    )
    
    # Instrument the app and expose metrics endpoint
    instrumentator.instrument(app).expose(app, endpoint="/metrics", tags=["Monitoring"])
    
    logger.info(
        "Prometheus metrics configured",
        service_name="testing-observability-platform",
        endpoint="/metrics"
    )
    
    return instrumentator


# ====================
# Metric Helper Functions
# ====================

def record_dataset_upload(
    status: str,
    format_type: str,
    size_bytes: int,
    row_count: int,
    processing_time_seconds: float
) -> None:
    """Record metrics for a dataset upload."""
    DATASET_UPLOADS.labels(status=status, format=format_type).inc()
    if status == "success":
        DATASET_SIZE.observe(size_bytes)
        DATASET_ROW_COUNT.observe(row_count)
    DATASET_PROCESSING_TIME.observe(processing_time_seconds)


def record_experiment_run(
    status: str,
    model: str,
    test_count: int,
    success_count: int,
    duration_seconds: float,
    experiment_name: str
) -> None:
    """Record metrics for an experiment run."""
    EXPERIMENT_RUNS.labels(status=status, model=model).inc()
    EXPERIMENT_TEST_COUNT.observe(test_count)
    EXPERIMENT_DURATION.observe(duration_seconds)
    
    if test_count > 0:
        success_rate = success_count / test_count
        EXPERIMENT_SUCCESS_RATE.labels(experiment_name=experiment_name).set(success_rate)


def record_evaluation(
    evaluator: str,
    passed: bool,
    score: float,
    latency_seconds: float
) -> None:
    """Record metrics for an evaluation."""
    EVALUATION_COUNT.labels(
        evaluator=evaluator, 
        passed=str(passed).lower()
    ).inc()
    EVALUATION_SCORE.labels(evaluator=evaluator).observe(score)
    EVALUATION_LATENCY.labels(evaluator=evaluator).observe(latency_seconds)


def record_token_usage(
    model: str,
    operation: str,
    token_count: int,
    estimated_cost: float = 0.0
) -> None:
    """Record token usage metrics."""
    TOKEN_USAGE.labels(model=model, operation=operation).inc(token_count)
    if estimated_cost > 0:
        TOKEN_COST.labels(model=model, operation=operation).inc(estimated_cost)