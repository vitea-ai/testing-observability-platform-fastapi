"""
Celery tasks for evaluation processing.
"""

import asyncio
import httpx
import json
from typing import List, Dict, Any
from uuid import UUID
from datetime import datetime
from celery import Task
from celery.utils.log import get_task_logger

from app.workers.celery_app import celery_app
from app.core.config import settings
from app.core.database import CelerySessionLocal
from app.services.evaluation_service import EvaluationService
from app.models.evaluation import Evaluation

logger = get_task_logger(__name__)


class EvaluationTask(Task):
    """Base task class with database session management."""
    
    async def get_db_session(self):
        """Get a database session for the task."""
        async with CelerySessionLocal() as session:
            yield session


@celery_app.task(
    bind=True,
    base=EvaluationTask,
    name="evaluate_experiment",
    max_retries=3,
    default_retry_delay=60,
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
)
def evaluate_experiment_task(
    self,
    evaluation_ids: List[str],
    experiment_id: str,
    evaluator_ids: List[str],
    test_results: List[Dict[str, Any]],
    experiment_name: str = "Unknown",
) -> Dict[str, Any]:
    """
    Process evaluation task asynchronously.
    
    Args:
        evaluation_ids: List of evaluation record IDs (one per evaluator)
        experiment_id: Experiment ID
        evaluator_ids: List of evaluator IDs to run
        test_results: Test results from the experiment
        experiment_name: Name of the experiment for logging
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        logger.info(f"Starting evaluation task for experiment {experiment_id} with evaluators: {evaluator_ids}")
        
        # Run async evaluation logic
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            _run_evaluation_async(
                self,
                evaluation_ids,
                experiment_id,
                evaluator_ids,
                test_results,
                experiment_name
            )
        )
        loop.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Evaluation task failed for experiment {experiment_id}: {str(e)}")
        
        # Retry the task with exponential backoff
        if self.request.retries < self.max_retries:
            retry_in = min(60 * (2 ** self.request.retries), 600)  # Max 10 minutes
            logger.info(f"Retrying evaluation task in {retry_in} seconds (attempt {self.request.retries + 1}/{self.max_retries})")
            raise self.retry(exc=e, countdown=retry_in)
        
        # Max retries exceeded, mark evaluations as failed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_mark_evaluations_failed(evaluation_ids, str(e)))
        loop.close()
        
        raise


async def _run_evaluation_async(
    task: Task,
    evaluation_ids: List[str],
    experiment_id: str,
    evaluator_ids: List[str],
    test_results: List[Dict[str, Any]],
    experiment_name: str,
) -> Dict[str, Any]:
    """
    Async implementation of evaluation processing.
    """
    evaluation_results = []
    missing_evaluators = []
    
    # Update evaluation status to running in database immediately
    if evaluation_ids:
        async with CelerySessionLocal() as db:
            service = EvaluationService(db)
            await service.update_evaluation_status(
                evaluation_ids=evaluation_ids,
                status="running"
            )
            logger.info(f"Updated {len(evaluation_ids)} evaluations to 'running' status")
    
    # Transform test results to format expected by evaluator service
    test_cases = []
    for result in test_results:
        test_case = {
            "input": result.get("input", ""),
            "actual_output": result.get("actual_output", ""),
            "expected_output": result.get("expected_output", ""),
            "context": result.get("context", []),
            "test_case_type": result.get("test_case_type", "single_turn"),
            "metadata": result.get("metadata", {})
        }
        # Ensure context is a list
        if isinstance(test_case["context"], str):
            test_case["context"] = [test_case["context"]] if test_case["context"] else []
        elif not isinstance(test_case["context"], list):
            test_case["context"] = []
        test_cases.append(test_case)
    
    # Get evaluator service URL
    evaluator_service_url = settings.evaluator_service_url or "http://localhost:9002"
    
    # Log evaluation start
    logger.info(f"Starting evaluation with {len(evaluator_ids)} evaluators for experiment {experiment_id}")
    
    # Brief delay to allow status propagation
    await asyncio.sleep(0.5)
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # First, fetch available evaluators
        try:
            response = await client.get(f"{evaluator_service_url}/evaluators")
            available_evaluators = response.json().get("evaluators", [])
            available_ids = [e["id"] for e in available_evaluators]
            
            # Check for missing evaluators
            missing_evaluators = [e_id for e_id in evaluator_ids if e_id not in available_ids]
            if missing_evaluators:
                logger.warning(f"Evaluators not found: {missing_evaluators}. Available: {available_ids}")
                for missing_id in missing_evaluators:
                    evaluation_results.append({
                        "evaluator_id": missing_id,
                        "metric_name": missing_id,
                        "status": "failed",
                        "score": 0,
                        "details": {
                            "error": f"Evaluator '{missing_id}' not found in evaluator service"
                        }
                    })
        except Exception as e:
            logger.error(f"Failed to fetch evaluators: {e}")
            missing_evaluators = []
        
        # Run each evaluator
        total_evaluators = len(evaluator_ids)
        for idx, evaluator_id in enumerate(evaluator_ids):
            if evaluator_id in missing_evaluators:
                continue
            
            try:
                logger.info(f"Running evaluator {evaluator_id} ({idx + 1}/{total_evaluators})")
                
                # Log progress update
                progress = int(((idx + 0.5) / total_evaluators) * 100)
                logger.info(f"Running {evaluator_id} - {progress}% complete")
                
                # Call evaluator service
                response = await client.post(
                    f"{evaluator_service_url}/evaluate/{evaluator_id}",
                    json={"test_cases": test_cases, "config": {}}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    evaluation_results.append({
                        "evaluator_id": evaluator_id,
                        "metric_name": evaluator_id.replace("_", " ").title(),
                        "score": round(result.get("overall_score", 0) * 100),
                        "status": "completed",
                        "details": result.get("results", [])
                    })
                else:
                    evaluation_results.append({
                        "evaluator_id": evaluator_id,
                        "metric_name": evaluator_id,
                        "score": 0,
                        "status": "failed",
                        "details": {"error": f"HTTP {response.status_code}: {response.text}"}
                    })
                    
            except httpx.TimeoutException:
                logger.error(f"Evaluator {evaluator_id} timed out")
                evaluation_results.append({
                    "evaluator_id": evaluator_id,
                    "metric_name": evaluator_id,
                    "score": 0,
                    "status": "failed",
                    "details": {"error": "Evaluation timed out after 300 seconds"}
                })
            except Exception as e:
                logger.error(f"Evaluator {evaluator_id} failed: {e}")
                evaluation_results.append({
                    "evaluator_id": evaluator_id,
                    "metric_name": evaluator_id,
                    "score": 0,
                    "status": "failed",
                    "details": {"error": str(e)}
                })
    
    # Update evaluation records in database
    async with CelerySessionLocal() as db:
        eval_service = EvaluationService(db)
        
        for i, evaluator_id in enumerate(evaluator_ids):
            if i < len(evaluation_ids):
                eval_id = UUID(evaluation_ids[i])
                # Find the corresponding result
                result = next((r for r in evaluation_results if r["evaluator_id"] == evaluator_id), None)
                
                if result:
                    # Update the evaluation in database
                    if result["status"] == "completed":
                        await eval_service.complete_evaluation(
                            evaluation_id=eval_id,
                            results={
                                "details": result.get("details", []),
                                "summary": result
                            },
                            score=result.get("score", 0) / 100.0,  # Convert to 0-1 scale
                            total_tests=len(test_results),
                            passed_tests=sum(1 for d in result.get("details", []) if d.get("passed", False)),
                            failed_tests=sum(1 for d in result.get("details", []) if not d.get("passed", True))
                        )
                    else:
                        await eval_service.fail_evaluation(
                            evaluation_id=eval_id,
                            error_message=result.get("details", {}).get("error", "Evaluation failed")
                        )
                else:
                    # Mark as failed if no result
                    await eval_service.fail_evaluation(
                        evaluation_id=eval_id,
                        error_message="Evaluation failed to produce results"
                    )
    
    # Calculate summary
    total = len(evaluation_results)
    passed = sum(1 for r in evaluation_results if r["status"] == "completed")
    failed = sum(1 for r in evaluation_results if r["status"] == "failed")
    avg_score = sum(r["score"] for r in evaluation_results if r["status"] == "completed") / max(passed, 1)
    
    summary = {
        "total_evaluated": total,
        "passed": passed,
        "failed": failed,
        "average_score": avg_score
    }
    
    # Update evaluation status to completed in database
    if evaluation_ids:
        async with CelerySessionLocal() as db:
            service = EvaluationService(db)
            await service.update_evaluation_status(
                evaluation_ids=evaluation_ids,
                status="completed",
                result={"summary": summary, "results": evaluation_results}
            )
    
    # Log completion
    logger.info(f"Completed evaluation for experiment {experiment_id} with {len(evaluation_results)} results, summary: {summary}")
    
    return {
        "task_id": task.request.id,
        "experiment_id": experiment_id,
        "evaluation_ids": evaluation_ids,
        "results": evaluation_results,
        "summary": summary,
        "completed_at": datetime.utcnow().isoformat()
    }


async def _mark_evaluations_failed(evaluation_ids: List[str], error_message: str):
    """Mark all evaluations as failed."""
    async with CelerySessionLocal() as db:
        eval_service = EvaluationService(db)
        for eval_id_str in evaluation_ids:
            try:
                eval_id = UUID(eval_id_str)
                await eval_service.fail_evaluation(eval_id, error_message)
            except Exception as e:
                logger.error(f"Failed to mark evaluation {eval_id_str} as failed: {e}")


@celery_app.task(name="cancel_evaluation")
def cancel_evaluation_task(task_id: str) -> bool:
    """
    Cancel a running evaluation task.
    
    Args:
        task_id: The Celery task ID to cancel
        
    Returns:
        True if cancelled successfully, False otherwise
    """
    try:
        celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
        logger.info(f"Cancelled evaluation task: {task_id}")
        
        # Log cancellation
        logger.info(f"Evaluation cancelled by user for task {task_id}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        return False