"""
Celery tasks for automated experiment execution via HTTP endpoints.
"""

import asyncio
import httpx
import json
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from celery import Task
from celery.utils.log import get_task_logger

from app.workers.celery_app import celery_app
from app.core.config import settings
from app.core.websocket_manager import WebSocketManager
from app.core.database import CelerySessionLocal
from app.services.experiment_service import ExperimentService
from app.models.experiment import ExperimentStatus

logger = get_task_logger(__name__)

# WebSocket manager instance (will be initialized in FastAPI app)
ws_manager: Optional[WebSocketManager] = None


class ExperimentRunnerTask(Task):
    """Base task class with database session management."""
    
    async def get_db_session(self):
        """Get a database session for the task."""
        async with CelerySessionLocal() as session:
            yield session


@celery_app.task(
    bind=True,
    base=ExperimentRunnerTask,
    name="run_automated_experiment",
    max_retries=3,
    default_retry_delay=60,
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
)
def run_automated_experiment_task(
    self,
    experiment_id: str,
    endpoint_config: Dict[str, Any],
    test_cases: List[Dict[str, Any]],
    batch_size: int = 10,
    experiment_name: str = "Unknown",
) -> Dict[str, Any]:
    """
    Process experiment test cases through HTTP endpoint.
    
    Args:
        experiment_id: Experiment ID
        endpoint_config: HTTP endpoint configuration including URL, headers, timeout
        test_cases: List of test cases with inputs from CSV
        batch_size: Number of concurrent requests to make
        experiment_name: Name of the experiment for logging
        
    Returns:
        Dictionary with execution results
    """
    try:
        logger.info(f"Starting automated experiment execution for {experiment_id}")
        
        # Run async execution logic
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            _run_experiment_async(
                self,
                experiment_id,
                endpoint_config,
                test_cases,
                batch_size,
                experiment_name
            )
        )
        loop.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Experiment execution task failed for {experiment_id}: {str(e)}")
        
        # Retry the task with exponential backoff
        if self.request.retries < self.max_retries:
            retry_in = min(60 * (2 ** self.request.retries), 600)  # Max 10 minutes
            logger.info(f"Retrying experiment task in {retry_in} seconds (attempt {self.request.retries + 1}/{self.max_retries})")
            raise self.retry(exc=e, countdown=retry_in)
        
        # Max retries exceeded, mark experiment as failed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_mark_experiment_failed(experiment_id, str(e)))
        loop.close()
        
        raise


async def _run_experiment_async(
    task: Task,
    experiment_id: str,
    endpoint_config: Dict[str, Any],
    test_cases: List[Dict[str, Any]],
    batch_size: int,
    experiment_name: str,
) -> Dict[str, Any]:
    """
    Async implementation of experiment execution.
    """
    results = []
    failed_count = 0
    success_count = 0
    total_latency = 0
    
    # Extract endpoint configuration
    endpoint_url = endpoint_config.get("url")
    headers = endpoint_config.get("headers", {})
    timeout = endpoint_config.get("timeout", 30)
    retry_attempts = endpoint_config.get("retry_attempts", 3)
    retry_delay = endpoint_config.get("retry_delay", 1)
    
    if not endpoint_url:
        raise ValueError("Endpoint URL is required for automated execution")
    
    logger.info(f"Using endpoint URL: {endpoint_url} with timeout: {timeout}s")
    
    # Send initial WebSocket update
    await _send_websocket_update(
        task.request.id,
        "processing",
        {"message": f"Starting automated execution with {len(test_cases)} test cases", "progress": 0}
    )
    
    # Create httpx client with explicit configuration for Docker networking
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        verify=False  # Disable SSL verification for internal services
    ) as client:
        # Process test cases in batches
        for batch_start in range(0, len(test_cases), batch_size):
            batch_end = min(batch_start + batch_size, len(test_cases))
            batch = test_cases[batch_start:batch_end]
            
            # Send progress update
            progress = int((batch_start / len(test_cases)) * 100)
            await _send_websocket_update(
                task.request.id,
                "processing",
                {"message": f"Processing batch {batch_start//batch_size + 1}", "progress": progress}
            )
            
            # Process batch concurrently
            batch_tasks = []
            for test_case in batch:
                batch_tasks.append(
                    _execute_single_test(
                        client,
                        endpoint_url,
                        headers,
                        test_case,
                        retry_attempts,
                        retry_delay
                    )
                )
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for test_case, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    # Handle exception case
                    failed_count += 1
                    results.append({
                        "test_id": test_case.get("test_id", "unknown"),
                        "input": test_case.get("input", ""),
                        "expected_output": test_case.get("expected_output", ""),
                        "actual_output": "",
                        "status": "error",
                        "error": str(result),
                        "execution_time": 0,
                        "metadata": test_case.get("metadata", {})
                    })
                else:
                    # Handle successful execution
                    if result["status"] == "success":
                        success_count += 1
                    else:
                        failed_count += 1
                    
                    total_latency += result.get("execution_time", 0)
                    results.append(result)
    
    # Update experiment in database
    async with CelerySessionLocal() as db:
        service = ExperimentService(db)
        
        # Store test results
        await service.store_test_results(
            experiment_id=UUID(experiment_id),
            results=results
        )
        
        # Update experiment status
        await service.update_experiment_status(
            experiment_id=UUID(experiment_id),
            status=ExperimentStatus.COMPLETED,
            progress=100.0
        )
    
    # Calculate summary
    total = len(results)
    avg_latency = total_latency / max(total, 1)
    success_rate = (success_count / max(total, 1)) * 100
    
    summary = {
        "total_tests": total,
        "successful": success_count,
        "failed": failed_count,
        "success_rate": success_rate,
        "average_latency_ms": avg_latency
    }
    
    # Send final WebSocket update
    await _send_websocket_update(
        task.request.id,
        "completed",
        {
            "message": "Experiment execution completed",
            "progress": 100,
            "summary": summary,
            "results_count": len(results)
        }
    )
    
    logger.info(f"Completed automated experiment {experiment_id} with {total} test cases")
    
    return {
        "task_id": task.request.id,
        "experiment_id": experiment_id,
        "results": results,
        "summary": summary,
        "completed_at": datetime.utcnow().isoformat()
    }


def _extract_guardrail_output(response_data: Dict[str, Any]) -> str:
    """
    Extract detected PHI types from guardrail response.
    """
    action = response_data.get("action", "no_action")
    
    if action == "modify" and "modifications" in response_data:
        # Extract unique PHI types detected
        detected_types = set()
        for mod in response_data["modifications"]:
            # Use the 'type' field which contains the PHI type
            if "type" in mod:
                detected_types.add(mod["type"])
                
        # Return comma-separated list of detected types
        if detected_types:
            return ", ".join(sorted(detected_types))
    
    return "None"


def _determine_test_status(actual_output: str, expected_output: str) -> str:
    """
    Determine if test passed by comparing actual and expected outputs.
    Handles both exact matches and subset matching for PHI types.
    """
    # Handle None/empty cases
    if expected_output in ["None", "", None]:
        return "passed" if actual_output in ["None", "", None, "no_action"] else "failed"
    
    # For PHI detection, check if detected types match expected
    if "," in expected_output or "," in actual_output:
        # Parse as comma-separated lists
        expected_types = set(t.strip().lower() for t in expected_output.split(",") if t.strip())
        actual_types = set(t.strip().lower() for t in actual_output.split(",") if t.strip())
        
        # Check if actual contains all expected types (may have additional ones)
        return "passed" if expected_types.issubset(actual_types) else "failed"
    
    # Exact match
    return "passed" if actual_output == expected_output else "failed"


async def _execute_single_test(
    client: httpx.AsyncClient,
    endpoint_url: str,
    headers: Dict[str, str],
    test_case: Dict[str, Any],
    retry_attempts: int,
    retry_delay: float
) -> Dict[str, Any]:
    """
    Execute a single test case against the HTTP endpoint with retry logic.
    """
    test_id = test_case.get("test_id", "unknown")
    input_data = test_case.get("input", "")
    expected_output = test_case.get("expected_output", "")
    
    # Prepare request payload
    # Check if endpoint is a guardrail service (ends with /check)
    if endpoint_url.endswith("/check"):
        # Guardrail format - use 'content' field
        # Ensure context is a dict, not an array
        context = test_case.get("context", {})
        if isinstance(context, list):
            context = {}  # Convert empty list to empty dict for guardrails
        
        payload = {
            "content": input_data,
            "content_type": "application/json" if input_data.startswith("{") else "text/plain",
            "context": context,
            "metadata": test_case.get("metadata", {})
        }
    else:
        # Standard format
        payload = {
            "input": input_data,
            "context": test_case.get("context", []),
            "metadata": test_case.get("metadata", {})
        }
    
    last_error = None
    for attempt in range(retry_attempts):
        try:
            start_time = datetime.utcnow()
            
            # Make HTTP request
            response = await client.post(
                endpoint_url,
                json=payload,
                headers=headers
            )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                response_data = None
                try:
                    # Check if response is HTML (likely hitting a frontend page)
                    content_type = response.headers.get("content-type", "")
                    if "text/html" in content_type:
                        logger.error(f"Received HTML response from {endpoint_url}. This appears to be a web page, not an API endpoint.")
                        last_error = f"Invalid endpoint: {endpoint_url} returned HTML instead of JSON. Please use a valid API endpoint."
                        continue
                    
                    response_data = response.json()
                except Exception as json_error:
                    logger.error(f"Failed to parse JSON response from {endpoint_url}: {json_error}")
                    logger.error(f"Response text: {response.text[:500]}")
                    last_error = f"Invalid JSON response: {str(json_error)}"
                    continue
                
                if response_data is None:
                    last_error = "Response data is None after parsing"
                    continue
                
                # Ensure response_data is a dictionary
                if not isinstance(response_data, dict):
                    logger.error(f"Response is not a dictionary from {endpoint_url}. Type: {type(response_data)}")
                    last_error = f"Invalid response format: expected dictionary, got {type(response_data).__name__}"
                    continue
                
                # Extract actual output based on response format
                if endpoint_url.endswith("/check"):
                    # Guardrail response - extract detected PHI types
                    actual_output = _extract_guardrail_output(response_data)
                else:
                    # Standard response
                    actual_output = response_data.get("output", response_data.get("result", ""))
                
                # Determine test status
                status = _determine_test_status(actual_output, expected_output)
                
                # Safely handle metadata
                test_metadata = test_case.get("metadata") or {}
                response_metadata = {}
                if isinstance(response_data, dict):
                    response_metadata = response_data.get("metadata", {})
                
                return {
                    "test_id": test_id,
                    "input": input_data,
                    "expected_output": expected_output,
                    "actual_output": actual_output,
                    "status": status,
                    "execution_time": execution_time,
                    "metadata": {
                        **test_metadata,
                        "attempt": attempt + 1,
                        "response_metadata": response_metadata
                    }
                }
            else:
                # Non-200 response
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                
        except httpx.TimeoutException:
            last_error = f"Request timeout after {client.timeout.total} seconds"
        except httpx.RequestError as e:
            last_error = f"Request error: {str(e)}"
            logger.warning(f"Request error for {endpoint_url}: {str(e)}, attempt {attempt + 1}")
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error for {endpoint_url}: {str(e)}, attempt {attempt + 1}")
        
        # Wait before retry (except for last attempt)
        if attempt < retry_attempts - 1:
            await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
    
    # All retries failed
    return {
        "test_id": test_id,
        "input": input_data,
        "expected_output": expected_output,
        "actual_output": "",
        "status": "error",
        "error": last_error,
        "execution_time": 0,
        "metadata": {
            **test_case.get("metadata", {}),
            "attempts": retry_attempts,
            "final_error": last_error
        }
    }


async def _mark_experiment_failed(experiment_id: str, error_message: str):
    """Mark experiment as failed."""
    async with CelerySessionLocal() as db:
        service = ExperimentService(db)
        try:
            await service.update_experiment_status(
                experiment_id=UUID(experiment_id),
                status=ExperimentStatus.FAILED,
                error_message=error_message
            )
        except Exception as e:
            logger.error(f"Failed to mark experiment {experiment_id} as failed: {e}")


async def _send_websocket_update(task_id: str, status: str, data: Dict[str, Any]):
    """Send WebSocket update to connected clients."""
    try:
        if ws_manager:
            message = {
                "task_id": task_id,
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                **data
            }
            await ws_manager.broadcast(task_id, message)
            logger.debug(f"Sent WebSocket update for task {task_id}: {status}")
    except Exception as e:
        logger.warning(f"Failed to send WebSocket update: {e}")


@celery_app.task(name="cancel_experiment_execution")
def cancel_experiment_execution_task(task_id: str) -> bool:
    """
    Cancel a running experiment execution task.
    
    Args:
        task_id: The Celery task ID to cancel
        
    Returns:
        True if cancelled successfully, False otherwise
    """
    try:
        celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
        logger.info(f"Cancelled experiment execution task: {task_id}")
        
        # Send WebSocket update
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            _send_websocket_update(
                task_id,
                "cancelled",
                {"message": "Experiment execution cancelled by user", "progress": 0}
            )
        )
        loop.close()
        
        return True
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        return False