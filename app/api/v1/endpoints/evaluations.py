"""
Evaluations endpoint for managing evaluation operations.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import logger
from app.core.dependencies import (
    get_current_user_optional,
    get_db
)
from app.services.evaluation_service import EvaluationService

router = APIRouter()

# In-memory storage for Tier 1 (development)
evaluation_storage: Dict[str, Dict[str, Any]] = {}


@router.get(
    "/",
    summary="Get all evaluations",
    description="Get all evaluations"
)
async def get_all_evaluations(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Get all evaluations with pagination (Node.js API compatibility).
    Returns evaluations that were run standalone (not part of experiments).
    """
    logger.info(f"Getting all evaluations - page: {page}, limit: {limit}")
    
    if settings.deployment_tier == "development":
        # Tier 1: Return from in-memory storage
        evaluations = list(evaluation_storage.values())
        
        # Sort by created_at descending
        evaluations.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        # Paginate
        start = (page - 1) * limit
        end = start + limit
        paginated = evaluations[start:end]
        
        return {
            "evaluations": paginated,
            "total": len(evaluations),
            "page": page,
            "limit": limit
        }
    else:
        # Tier 2+: Fetch from database
        service = EvaluationService(db)
        
        # Get paginated evaluations
        result = await service.get_all_evaluations(
            page=page,
            limit=limit
        )
        
        # Format for API response
        formatted_evaluations = []
        for evaluation in result["evaluations"]:
            formatted = await service.get_evaluation_for_api_response(evaluation.id)
            if formatted:
                formatted_evaluations.append(formatted)
        
        return {
            "evaluations": formatted_evaluations,
            "total": result["total"],
            "page": result["page"],
            "limit": result["limit"]
        }


@router.get(
    "/{evaluation_id}",
    summary="Get evaluation by ID",
    description="Get a specific evaluation by its ID or task ID"
)
async def get_evaluation_by_id(
    evaluation_id: str,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Get a specific evaluation by ID or task ID (Node.js API compatibility).
    """
    logger.info(f"Getting evaluation: {evaluation_id}")
    
    if settings.deployment_tier == "development":
        # Tier 1: Check in-memory storage
        if evaluation_id in evaluation_storage:
            return evaluation_storage[evaluation_id]
        
        # Return "running" status for new evaluations
        return {
            "evaluation_id": evaluation_id,
            "experiment_id": "c37be40f-ec72-4861-a0e9-df7275e0735a",
            "experiment_name": "Test Experiment - Evaluation",
            "agent_name": "GPT-4",
            "dataset_name": "Test Dataset",
            "evaluation_type": "comprehensive",
            "status": "running",
            "evaluations": [],
            "summary": {},
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None
        }
    else:
        # Tier 2+: Fetch from database
        try:
            # First try to find evaluations by task_id
            service = EvaluationService(db)
            evaluations_by_task = await service.get_evaluations_by_task_id(evaluation_id)
            
            if evaluations_by_task:
                # Aggregate multiple evaluations into a single response
                first_eval = evaluations_by_task[0]
                
                # Check if all evaluations are complete
                all_completed = all(e.status == "completed" for e in evaluations_by_task)
                any_failed = any(e.status == "failed" for e in evaluations_by_task)
                any_running = any(e.status in ["running", "queued"] for e in evaluations_by_task)
                
                # Determine overall status based on individual evaluation statuses
                if any_running:
                    status = "running"
                elif any_failed:
                    status = "failed"
                elif all_completed:
                    status = "completed"
                else:
                    # Default to running if status is unclear
                    status = "running"
                
                # Aggregate evaluation results
                evaluation_results = []
                for eval in evaluations_by_task:
                    evaluation_results.append({
                        "evaluator_id": eval.evaluator_id,
                        "metric_name": eval.evaluator_name,
                        "score": (eval.score or 0) * 100,  # Convert to percentage
                        "status": eval.status,
                        "details": eval.results.get("details", []) if eval.results else []
                    })
                
                # Calculate summary
                completed_evals = [e for e in evaluations_by_task if e.status == "completed"]
                avg_score = 0
                if completed_evals:
                    avg_score = sum((e.score or 0) * 100 for e in completed_evals) / len(completed_evals)
                
                return {
                    "evaluation_id": evaluation_id,  # Use task_id as evaluation_id
                    "experiment_id": str(first_eval.experiment_id),
                    "experiment_name": first_eval.experiment.name if first_eval.experiment else "Unknown",
                    "agent_name": first_eval.experiment.agent_config.get("name", "Unknown") if first_eval.experiment and first_eval.experiment.agent_config else "Unknown",
                    "dataset_name": first_eval.experiment.dataset.name if first_eval.experiment and hasattr(first_eval.experiment, 'dataset') and first_eval.experiment.dataset else "Test Dataset",
                    "evaluation_type": "comprehensive",
                    "status": status,
                    "evaluations": evaluation_results,
                    "summary": {
                        "average_score": avg_score,
                        "total_evaluators": len(evaluations_by_task),
                        "completed": len(completed_evals),
                        "failed": len([e for e in evaluations_by_task if e.status == "failed"])
                    },
                    "created_at": first_eval.created_at.isoformat() if first_eval.created_at else None,
                    "completed_at": max((e.completed_at for e in evaluations_by_task if e.completed_at), default=None).isoformat() if any(e.completed_at for e in evaluations_by_task) else None
                }
            
            # If not found by task_id, try as evaluation_id
            eval_uuid = UUID(evaluation_id)
            formatted = await service.get_evaluation_for_api_response(eval_uuid)
            
            if formatted:
                return formatted
            
            # Return failed status if not found after checking
            logger.warning(f"Evaluation {evaluation_id} not found in database")
            return {
                "evaluation_id": evaluation_id,
                "status": "failed",
                "evaluations": [],
                "summary": {"error": "Evaluation not found"},
                "created_at": datetime.utcnow().isoformat(),
                "completed_at": datetime.utcnow().isoformat()
            }
        except ValueError:
            # Not a valid UUID, might be a Celery task ID
            # Check if it's a running Celery task
            from celery.result import AsyncResult
            from app.workers.celery_app import celery_app
            
            result = AsyncResult(evaluation_id, app=celery_app)
            
            if result.state in ['STARTED', 'RETRY']:
                return {
                    "evaluation_id": evaluation_id,
                    "status": "running",
                    "evaluations": [],
                    "summary": {},
                    "created_at": datetime.utcnow().isoformat(),
                    "completed_at": None
                }
            elif result.state == 'PENDING':
                # PENDING usually means the task doesn't exist or hasn't started
                # Check if we can find it in the database first
                service = EvaluationService(db)
                evaluations_by_task = await service.get_evaluations_by_task_id(evaluation_id)
                if not evaluations_by_task:
                    # Task doesn't exist or is expired
                    return {
                        "evaluation_id": evaluation_id,
                        "status": "failed",
                        "evaluations": [],
                        "summary": {"error": "Evaluation task not found or expired"},
                        "created_at": datetime.utcnow().isoformat(),
                        "completed_at": datetime.utcnow().isoformat()
                    }
                # If we found evaluations, process them as normal
                return {
                    "evaluation_id": evaluation_id,
                    "status": "running",
                    "evaluations": [],
                    "summary": {},
                    "created_at": datetime.utcnow().isoformat(),
                    "completed_at": None
                }
            elif result.state == 'SUCCESS':
                # Task completed but we don't have the evaluation data
                # Try to fetch by task_id again
                service = EvaluationService(db)
                evaluations_by_task = await service.get_evaluations_by_task_id(evaluation_id)
                if evaluations_by_task:
                    # Process as above (same aggregation logic)
                    first_eval = evaluations_by_task[0]
                    all_completed = all(e.status == "completed" for e in evaluations_by_task)
                    any_failed = any(e.status == "failed" for e in evaluations_by_task)
                    any_running = any(e.status in ["running", "queued"] for e in evaluations_by_task)
                    
                    # Determine overall status based on individual evaluation statuses
                    if any_running:
                        status = "running"
                    elif any_failed:
                        status = "failed"
                    elif all_completed:
                        status = "completed"
                    else:
                        # Default to running if status is unclear
                        status = "running"
                    
                    evaluation_results = []
                    for eval in evaluations_by_task:
                        evaluation_results.append({
                            "evaluator_id": eval.evaluator_id,
                            "metric_name": eval.evaluator_name,
                            "score": (eval.score or 0) * 100,
                            "status": eval.status,
                            "details": eval.results.get("details", []) if eval.results else []
                        })
                    
                    completed_evals = [e for e in evaluations_by_task if e.status == "completed"]
                    avg_score = sum((e.score or 0) * 100 for e in completed_evals) / len(completed_evals) if completed_evals else 0
                    
                    return {
                        "evaluation_id": evaluation_id,
                        "experiment_id": str(first_eval.experiment_id),
                        "experiment_name": first_eval.experiment.name if first_eval.experiment else "Unknown",
                        "agent_name": first_eval.experiment.agent_config.get("name", "Unknown") if first_eval.experiment and first_eval.experiment.agent_config else "Unknown",
                        "dataset_name": "Test Dataset",
                        "evaluation_type": "comprehensive",
                        "status": status,
                        "evaluations": evaluation_results,
                        "summary": {
                            "average_score": avg_score,
                            "total_evaluators": len(evaluations_by_task),
                            "completed": len(completed_evals),
                            "failed": len([e for e in evaluations_by_task if e.status == "failed"])
                        },
                        "created_at": first_eval.created_at.isoformat() if first_eval.created_at else None,
                        "completed_at": max((e.completed_at for e in evaluations_by_task if e.completed_at), default=None).isoformat() if any(e.completed_at for e in evaluations_by_task) else None
                    }
            
            # Check for FAILURE state
            elif result.state == 'FAILURE':
                return {
                    "evaluation_id": evaluation_id,
                    "status": "failed",
                    "evaluations": [],
                    "summary": {"error": str(result.info) if result.info else "Task failed"},
                    "created_at": datetime.utcnow().isoformat(),
                    "completed_at": datetime.utcnow().isoformat()
                }
            else:
                # Unknown state or task doesn't exist
                return {
                    "evaluation_id": evaluation_id,
                    "status": "failed",
                    "evaluations": [],
                    "summary": {"error": f"Unknown task state: {result.state}"},
                    "created_at": datetime.utcnow().isoformat(),
                    "completed_at": datetime.utcnow().isoformat()
                }


@router.delete(
    "/{evaluation_id}",
    summary="Delete evaluation",
    description="Delete an evaluation by its ID"
)
async def delete_evaluation(
    evaluation_id: str,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Delete an evaluation (Node.js API compatibility).
    """
    logger.info(f"Deleting evaluation: {evaluation_id}")
    
    if settings.deployment_tier == "development":
        # Tier 1: Remove from in-memory storage
        if evaluation_id in evaluation_storage:
            del evaluation_storage[evaluation_id]
            return {
                "message": "Evaluation deleted successfully",
                "evaluation_id": evaluation_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation {evaluation_id} not found"
            )
    else:
        # Tier 2+: Delete from database
        try:
            eval_uuid = UUID(evaluation_id)
            service = EvaluationService(db)
            deleted = await service.delete_evaluation(eval_uuid)
            
            if deleted:
                return {
                    "message": "Evaluation deleted successfully",
                    "evaluation_id": evaluation_id
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Evaluation {evaluation_id} not found"
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid evaluation ID format"
            )


@router.get(
    "/task/{task_id}/status",
    summary="Get evaluation task status",
    description="Check the status of an evaluation task"
)
async def get_evaluation_task_status(
    task_id: str,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Get the status of an evaluation task.
    """
    from celery.result import AsyncResult
    from app.workers.celery_app import celery_app
    
    result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": result.state,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None
    }
    
    # Add result data if available
    if result.ready() and result.successful():
        task_result = result.result
        response.update({
            "result": task_result,
            "completed_at": task_result.get("completed_at") if task_result else None
        })
    elif result.failed():
        response["error"] = str(result.info)
    
    return response