"""
Experiments endpoint for managing AI evaluation experiments.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
import uuid
from datetime import datetime
from enum import Enum
import csv
import io
import json
import asyncio
import httpx

from fastapi import APIRouter, HTTPException, status, Depends, Query, Body, BackgroundTasks, UploadFile, File, Form
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import logger
from app.core.dependencies import (
    get_current_user_optional,
    get_current_user,
    apply_rate_limit,
    audit_log,
    get_db
)
from app.services.experiment_service import ExperimentService
from app.services.evaluation_service import EvaluationService
from app.schemas.experiment import (
    ExperimentStatus,
    ExecutionMode,
    AgentConfig,
    ExperimentBase,
    ExperimentCreate,
    ExperimentUpdate,
    ExperimentResponse,
    ExperimentListResponse,
    ExperimentResultItem,
    ExperimentResults,
    ExperimentExecuteRequest
)

router = APIRouter()


# ==========================================
# In-Memory Storage (Tier 1)
# ==========================================
experiments_storage: Dict[str, Dict[str, Any]] = {}
experiment_results_storage: Dict[str, Dict[str, Any]] = {}
evaluation_storage: Dict[str, Dict[str, Any]] = {}


# ==========================================
# API Endpoints
# ==========================================
@router.get(
    "/",
    response_model=ExperimentListResponse,
    summary="List experiments",
    description="Retrieve a paginated list of experiments with optional filtering"
)
async def list_experiments(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[ExperimentStatus] = Query(None, description="Filter by status"),
    dataset_id: Optional[UUID] = Query(None, description="Filter by dataset"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    List all experiments with pagination and filtering.
    """
    logger.info(f"Listing experiments - page: {page}, page_size: {page_size}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        filtered_experiments = list(experiments_storage.values())
        
        # Apply filters
        if status:
            filtered_experiments = [e for e in filtered_experiments if e.get("status") == status]
        if dataset_id:
            filtered_experiments = [e for e in filtered_experiments if e.get("dataset_id") == str(dataset_id)]
        if search:
            search_lower = search.lower()
            filtered_experiments = [
                e for e in filtered_experiments
                if search_lower in e.get("name", "").lower() or
                   search_lower in e.get("description", "").lower()
            ]
        
        # Pagination
        total = len(filtered_experiments)
        start = (page - 1) * page_size
        end = start + page_size
        paginated = filtered_experiments[start:end]
        
        return ExperimentListResponse(
            experiments=[ExperimentResponse(**e) for e in paginated],
            total=total,
            page=page,
            page_size=page_size
        )
    else:
        # Tier 2+: Database implementation
        service = ExperimentService(db)
        result = await service.list_experiments(
            page=page,
            page_size=page_size,
            status=status,
            dataset_id=dataset_id,
            search=search
        )
        return result


@router.post(
    "/",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create experiment",
    description="Create a new experiment"
)
async def create_experiment(
    experiment: ExperimentCreate,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user) if settings.is_feature_enabled("authentication") else Depends(get_current_user_optional)
):
    """
    Create a new experiment.
    """
    logger.info(f"Creating experiment: {experiment.name}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        from uuid import uuid4
        
        experiment_id = str(uuid4())
        experiment_dict = experiment.model_dump()
        experiment_dict.update({
            "id": experiment_id,
            "status": ExperimentStatus.PENDING,
            "progress": 0.0,
            "started_at": None,
            "completed_at": None,
            "created_at": datetime.utcnow(),
            "updated_at": None
        })
        
        # Convert agent_config to dict if needed
        if isinstance(experiment_dict.get("agent_config"), AgentConfig):
            experiment_dict["agent_config"] = experiment_dict["agent_config"].model_dump()
        
        experiments_storage[experiment_id] = experiment_dict
        
        if settings.is_feature_enabled("audit_logging"):
            await audit_log(
                action="experiment.create",
                resource_id=experiment_id,
                user=current_user.id if current_user else "anonymous",
                details={"name": experiment.name}
            )
        
        return ExperimentResponse(**experiment_dict)
    else:
        # Tier 2+: Database implementation
        service = ExperimentService(db)
        created = await service.create_experiment(
            experiment,
            created_by=current_user.id if current_user else "system"
        )
        return created


@router.get(
    "/all-evaluations/",
    summary="Get all evaluations",
    description="Get all evaluations for evaluation results page"
)
async def get_all_evaluations(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Results per page"),
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Get all evaluations (Node.js API compatibility).
    """
    logger.info(f"Getting all evaluations - page: {page}, limit: {limit}")
    
    if settings.deployment_tier == "development":
        # Tier 1: Return in-memory evaluations
        all_evaluations = list(evaluation_storage.values())
        start = (page - 1) * limit
        end = start + limit
        return {
            "evaluations": all_evaluations[start:end],
            "total": len(all_evaluations),
            "page": page,
            "limit": limit
        }
    else:
        # Tier 2+: Fetch from database
        service = EvaluationService(db)
        result = await service.get_all_evaluations(page=page, limit=limit)
        
        # Format evaluations for API response
        formatted_evaluations = []
        for eval in result["evaluations"]:
            formatted = await service.get_evaluation_for_api_response(eval.id)
            if formatted:
                formatted_evaluations.append(formatted)
        
        return {
            "evaluations": formatted_evaluations,
            "total": result["total"],
            "page": result["page"],
            "limit": result["limit"]
        }


@router.get(
    "/evaluations/{evaluation_id}",
    summary="Get evaluation by ID",
    description="Get a specific evaluation by its ID"
)
async def get_evaluation_by_id(
    evaluation_id: str,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Get a specific evaluation by ID (Node.js API compatibility).
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
            eval_uuid = UUID(evaluation_id)
            service = EvaluationService(db)
            formatted = await service.get_evaluation_for_api_response(eval_uuid)
            
            if formatted:
                return formatted
            
            # Return running status if not found (might still be processing)
            return {
                "evaluation_id": evaluation_id,
                "status": "running",
                "evaluations": [],
                "summary": {},
                "created_at": datetime.utcnow().isoformat(),
                "completed_at": None
            }
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid evaluation ID format"
            )


@router.delete(
    "/all-evaluations/{evaluation_id}",
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
    "/{experiment_id}",
    response_model=ExperimentResponse,
    summary="Get experiment",
    description="Retrieve a specific experiment by ID"
)
async def get_experiment(
    experiment_id: UUID,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Get an experiment by ID.
    """
    logger.info(f"Getting experiment: {experiment_id}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        experiment = experiments_storage.get(str(experiment_id))
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        # Ensure metadata is a dict, not an object
        if "metadata" in experiment and not isinstance(experiment["metadata"], dict):
            experiment["metadata"] = {}
        return ExperimentResponse(**experiment)
    else:
        # Tier 2+: Database implementation
        service = ExperimentService(db)
        experiment = await service.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        # Convert to dict and ensure metadata is properly formatted
        exp_dict = experiment.to_dict()
        return ExperimentResponse(**exp_dict)


@router.put(
    "/{experiment_id}",
    response_model=ExperimentResponse,
    summary="Update experiment",
    description="Update an existing experiment"
)
async def update_experiment(
    experiment_id: UUID,
    experiment_update: ExperimentUpdate,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user) if settings.is_feature_enabled("authentication") else Depends(get_current_user_optional)
):
    """
    Update an experiment.
    """
    logger.info(f"Updating experiment: {experiment_id}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        experiment = experiments_storage.get(str(experiment_id))
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        # Check if experiment is running
        if experiment.get("status") == ExperimentStatus.RUNNING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot update a running experiment"
            )
        
        # Update fields
        update_data = experiment_update.model_dump(exclude_unset=True)
        if "agent_config" in update_data and update_data["agent_config"]:
            if isinstance(update_data["agent_config"], AgentConfig):
                update_data["agent_config"] = update_data["agent_config"].model_dump()
        
        experiment.update(update_data)
        experiment["updated_at"] = datetime.utcnow()
        
        if settings.is_feature_enabled("audit_logging"):
            await audit_log(
                action="experiment.update",
                resource_id=str(experiment_id),
                user=current_user.id if current_user else "anonymous",
                details={"fields": list(update_data.keys())}
            )
        
        return ExperimentResponse(**experiment)
    else:
        # Tier 2+: Database implementation
        service = ExperimentService(db)
        updated = await service.update_experiment(
            experiment_id,
            experiment_update,
            updated_by=current_user.id if current_user else "system"
        )
        return updated


@router.delete(
    "/{experiment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete experiment",
    description="Delete an experiment"
)
async def delete_experiment(
    experiment_id: UUID,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user) if settings.is_feature_enabled("authentication") else Depends(get_current_user_optional)
):
    """
    Delete an experiment.
    """
    logger.info(f"Deleting experiment: {experiment_id}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        if str(experiment_id) not in experiments_storage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        experiment = experiments_storage[str(experiment_id)]
        if experiment.get("status") == ExperimentStatus.RUNNING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete a running experiment"
            )
        
        del experiments_storage[str(experiment_id)]
        
        # Also delete results if any
        if str(experiment_id) in experiment_results_storage:
            del experiment_results_storage[str(experiment_id)]
        
        if settings.is_feature_enabled("audit_logging"):
            await audit_log(
                action="experiment.delete",
                resource_id=str(experiment_id),
                user=current_user.id if current_user else "anonymous"
            )
        
        return None
    else:
        # Tier 2+: Database implementation
        service = ExperimentService(db)
        await service.delete_experiment(
            experiment_id,
            deleted_by=current_user.id if current_user else "system"
        )
        return None


@router.post(
    "/{experiment_id}/execute",
    response_model=Dict[str, Any],
    summary="Execute experiment",
    description="Start executing an experiment"
)
async def execute_experiment(
    experiment_id: UUID,
    request: ExperimentExecuteRequest = Body(default=ExperimentExecuteRequest()),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user) if settings.is_feature_enabled("authentication") else Depends(get_current_user_optional)
):
    """
    Execute an experiment asynchronously.
    """
    logger.info(f"Executing experiment: {experiment_id}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        experiment = experiments_storage.get(str(experiment_id))
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        if experiment.get("status") in [ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Experiment is already {experiment.get('status')}"
            )
        
        # Update status to running
        experiment["status"] = ExperimentStatus.RUNNING
        experiment["started_at"] = datetime.utcnow()
        experiment["progress"] = 0.0
        
        # In tier 1, we'll simulate execution
        async def simulate_execution():
            import asyncio
            import random
            
            # Simulate processing
            await asyncio.sleep(2)
            
            # Update experiment status
            experiment["status"] = ExperimentStatus.COMPLETED
            experiment["completed_at"] = datetime.utcnow()
            experiment["progress"] = 100.0
            
            # Create mock results
            experiment_results_storage[str(experiment_id)] = {
                "experiment_id": str(experiment_id),
                "status": ExperimentStatus.COMPLETED,
                "total_tests": 10,
                "passed_tests": 8,
                "failed_tests": 2,
                "error_tests": 0,
                "execution_time": 2.5,
                "results": [],
                "summary": {
                    "success_rate": 0.8,
                    "avg_execution_time": 0.25
                }
            }
            
            logger.info(f"Experiment {experiment_id} completed")
        
        if settings.enable_background_tasks:
            background_tasks.add_task(simulate_execution)
        
        if settings.is_feature_enabled("audit_logging"):
            await audit_log(
                action="experiment.execute",
                resource_id=str(experiment_id),
                user=current_user.id if current_user else "anonymous",
                details={"batch_size": request.batch_size}
            )
        
        return {
            "message": "Experiment execution started",
            "experiment_id": str(experiment_id),
            "status": "running"
        }
    else:
        # Tier 2+: Database implementation with real execution
        service = ExperimentService(db)
        result = await service.execute_experiment(
            experiment_id,
            batch_size=request.batch_size,
            timeout=request.timeout,
            evaluator_ids=request.evaluator_ids,
            background_tasks=background_tasks
        )
        return result


@router.get(
    "/{experiment_id}/results",
    response_model=ExperimentResults,
    summary="Get experiment results",
    description="Retrieve the results of an experiment"
)
async def get_experiment_results(
    experiment_id: UUID,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Get the results of an experiment.
    """
    logger.info(f"Getting results for experiment: {experiment_id}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        results = experiment_results_storage.get(str(experiment_id))
        if not results:
            experiment = experiments_storage.get(str(experiment_id))
            if not experiment:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Experiment {experiment_id} not found"
                )
            
            if experiment.get("status") != ExperimentStatus.COMPLETED:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Experiment is {experiment.get('status')}, results not available"
                )
            
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Results not found for this experiment"
            )
        
        return ExperimentResults(**results)
    else:
        # Tier 2+: Database implementation
        service = ExperimentService(db)
        results = await service.get_experiment_results(experiment_id)
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Results not found for this experiment"
            )
        return results


@router.get(
    "/{experiment_id}/test-results",
    summary="Get experiment test results",
    description="Get test results for an experiment (Node.js API compatibility)"
)
async def get_experiment_test_results(
    experiment_id: UUID,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Get test results for an experiment in Node.js API format.
    This endpoint provides compatibility with the frontend expecting the Node.js backend format.
    """
    logger.info(f"Getting test results for experiment: {experiment_id}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        experiment = experiments_storage.get(str(experiment_id))
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        # Get results from storage
        results = experiment_results_storage.get(str(experiment_id), {})
        test_results = results.get("results", [])
        
        # Format response to match Node.js API
        return {
            "experiment_id": str(experiment_id),
            "experiment_name": experiment.get("name", ""),
            "results": test_results,
            "count": len(test_results),
            "has_actual_outputs": any(r.get("actual_output") for r in test_results)
        }
    else:
        # Tier 2+: Database implementation
        service = ExperimentService(db)
        experiment = await service.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        # Get test results
        results = await service.get_experiment_results(experiment_id)
        test_results = results.get("results", []) if results else []
        
        # Format response to match Node.js API
        return {
            "experiment_id": str(experiment_id),
            "experiment_name": experiment.name,
            "results": test_results,
            "count": len(test_results),
            "has_actual_outputs": any(
                r.get("actual_output") for r in test_results
            )
        }


@router.post(
    "/{experiment_id}/evaluate",
    summary="Evaluate experiment",
    description="Run evaluation on completed experiment"
)
async def evaluate_experiment(
    experiment_id: UUID,
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Run evaluation on an experiment (Node.js API compatibility).
    """
    logger.info(f"Evaluating experiment: {experiment_id}")
    
    evaluator_ids = request.get("evaluator_ids", [])
    
    if not evaluator_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="evaluator_ids array is required"
        )
    
    # Get the experiment and test results
    if settings.deployment_tier == "development":
        experiment = experiments_storage.get(str(experiment_id))
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        # Get test results from storage
        results = experiment_results_storage.get(str(experiment_id), {})
        test_results = results.get("results", [])
    else:
        service = ExperimentService(db)
        experiment_model = await service.get_experiment(experiment_id)
        if not experiment_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        experiment = experiment_model.to_dict()
        
        # Get test results from database
        results = await service.get_experiment_results(experiment_id)
        test_results = results.get("results", []) if results else []
    
    # Create evaluation record
    if settings.deployment_tier == "development":
        # Tier 1: Use in-memory storage
        evaluation_id = str(uuid.uuid4())
        
        # Store initial evaluation record in memory
        evaluation_storage[evaluation_id] = {
            "evaluation_id": evaluation_id,
            "experiment_id": str(experiment_id),
            "experiment_name": experiment.get("name", ""),
            "agent_name": experiment.get("agent_config", {}).get("name", "Unknown") if isinstance(experiment.get("agent_config"), dict) else "Unknown",
            "dataset_name": experiment.get("dataset_name", "No Dataset"),
            "evaluation_type": "comprehensive",
            "status": "running",
            "evaluations": [],
            "summary": {},
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None
        }
        
        # Run evaluation in background (in-memory)
        background_tasks.add_task(
            run_evaluation_async,
            evaluation_id,
            experiment_id,
            evaluator_ids,
            test_results,
            experiment,
            None  # No DB session for tier 1
        )
    else:
        # Tier 2+: Save to database
        eval_service = EvaluationService(db)
        
        # Create evaluations for each evaluator
        created_evaluations = []
        for evaluator_id in evaluator_ids:
            evaluation = await eval_service.create_evaluation(
                experiment_id=experiment_id,
                evaluator_id=evaluator_id,
                evaluator_name=evaluator_id.replace("_", " ").title(),
                evaluator_config={},
                created_by=current_user.id if current_user else "system"
            )
            created_evaluations.append(evaluation)
        
        # Use the first evaluation's ID as the main evaluation ID
        evaluation_id = str(created_evaluations[0].id) if created_evaluations else str(uuid.uuid4())
        
        # Run evaluation in background (with DB)
        background_tasks.add_task(
            run_evaluation_async,
            evaluation_id,
            experiment_id,
            evaluator_ids,
            test_results,
            experiment,
            created_evaluations  # Pass created evaluation objects
        )
    
    logger.info(f"Created evaluation {evaluation_id} for experiment {experiment_id}")
    
    # Return evaluation info immediately
    return {
        "evaluation_id": evaluation_id,
        "experiment_id": str(experiment_id),
        "status": "running",
        "message": "Evaluation started successfully",
        "evaluator_ids": evaluator_ids
    }


@router.post(
    "/{experiment_id}/cancel",
    response_model=ExperimentResponse,
    summary="Cancel experiment",
    description="Cancel a running experiment"
)
async def cancel_experiment(
    experiment_id: UUID,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user) if settings.is_feature_enabled("authentication") else Depends(get_current_user_optional)
):
    """
    Cancel a running experiment.
    """
    logger.info(f"Cancelling experiment: {experiment_id}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        experiment = experiments_storage.get(str(experiment_id))
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        if experiment.get("status") != ExperimentStatus.RUNNING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel experiment with status {experiment.get('status')}"
            )
        
        experiment["status"] = ExperimentStatus.CANCELLED
        experiment["completed_at"] = datetime.utcnow()
        
        if settings.is_feature_enabled("audit_logging"):
            await audit_log(
                action="experiment.cancel",
                resource_id=str(experiment_id),
                user=current_user.id if current_user else "anonymous"
            )
        
        return ExperimentResponse(**experiment)
    else:
        # Tier 2+: Database implementation
        service = ExperimentService(db)
        cancelled = await service.cancel_experiment(experiment_id)
        return cancelled


@router.post(
    "/import-csv",
    response_model=ExperimentResponse,
    summary="Import CSV results",
    description="Import experiment results from a CSV file"
)
async def import_csv_results(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    dataset_id: Optional[str] = Form(None),
    execution_mode: str = Form("import"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Import experiment results from a CSV file.
    
    The CSV should contain columns like:
    - test_case_id or test_case_index
    - input, input_prompt, or prompt
    - expected_output
    - actual_output or output
    - context
    - latency_ms
    - token_usage_input
    - token_usage_output
    - meta_* fields for metadata
    """
    logger.info(f"Importing CSV results for experiment: {name}")
    
    # Read and decode CSV file
    try:
        contents = await file.read()
        decoded = contents.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(decoded))
        records = list(csv_reader)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {str(e)}"
        )
    
    if not records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file is empty"
        )
    
    # Transform CSV records to experiment results format
    test_results = []
    for index, row in enumerate(records):
        # Extract metadata from meta_* fields
        metadata = {}
        for key, value in row.items():
            if key and key.startswith('meta_'):
                metadata[key] = value
        
        # Parse JSON fields if they exist
        retrieval_context = []
        if row.get('retrieval_context'):
            try:
                retrieval_context = json.loads(row['retrieval_context'])
            except:
                pass
        
        tools_called = []
        if row.get('tools_called'):
            try:
                tools_called = json.loads(row['tools_called'])
            except:
                pass
        
        # Create result item
        result = {
            "test_id": row.get('test_case_id') or row.get('test_id') or f"test_{index}",
            "test_case_type": row.get('test_case_type', 'single_turn'),
            "input": row.get('input_prompt') or row.get('input') or row.get('prompt', ''),
            "expected_output": row.get('expected_output', ''),
            "actual_output": row.get('actual_output') or row.get('output', ''),
            "context": [row['context']] if row.get('context') else [],
            "retrieval_context": retrieval_context,
            "tools_called": tools_called,
            "status": "completed",
            "execution_time": float(row.get('latency_ms', 0)) / 1000 if row.get('latency_ms') else 0,
            "meta_data": metadata
        }
        test_results.append(result)
    
    # Store in database
    service = ExperimentService(db)
    
    # Create experiment
    from app.schemas.experiment import ExperimentCreate, AgentConfig
    from uuid import UUID as UUID_type
    
    experiment_create = ExperimentCreate(
        name=name,
        description=description,
        dataset_id=UUID_type(dataset_id) if dataset_id and dataset_id != 'null' else None,
        agent_config=AgentConfig(
            model="imported",
            provider="csv",
            temperature=0.0,
            max_tokens=1,  # Use minimum valid value
            system_prompt="",
            custom_fields={"source": file.filename}
        ),
        execution_mode="batch" if execution_mode == "import" else execution_mode,
        tags=["imported"],
        metadata={
            "source": "csv_import",
            "filename": file.filename,
            "record_count": len(test_results)
        }
    )
    
    experiment = await service.create_experiment(
        experiment_create,
        created_by=current_user.id if current_user else "import"
    )
    
    # Add test results directly without the problematic method
    from app.models.experiment import TestResult, ExperimentStatus
    for result_data in test_results:
        test_result = TestResult(
            experiment_id=experiment.id,
            test_id=result_data.get("test_id", f"test_{len(test_results)}"),
            test_case_type=result_data.get("test_case_type", "single_turn"),
            input=result_data.get("input", {}),
            expected_output=result_data.get("expected_output"),
            actual_output=result_data.get("actual_output"),
            context=result_data.get("context", []),
            retrieval_context=result_data.get("retrieval_context", []),
            tools_called=result_data.get("tools_called", []),
            status=result_data.get("status", "completed"),
            execution_time=result_data.get("execution_time", 0),
            error=result_data.get("error"),
            meta_data=result_data.get("meta_data", {})
        )
        db.add(test_result)
    
    # Update experiment status
    experiment.status = ExperimentStatus.COMPLETED
    experiment.completed_at = datetime.utcnow()
    experiment.progress = 1.0
    
    await db.commit()
    await db.refresh(experiment)
    
    experiment_data = experiment.to_dict()
    
    logger.info(f"Successfully imported {len(test_results)} results into experiment {experiment.id}")
    
    return experiment_data


async def run_evaluation_async(
    evaluation_id: str,
    experiment_id: UUID,
    evaluator_ids: List[str],
    test_results: List[Dict[str, Any]],
    experiment: Dict[str, Any],
    db_evaluations: Optional[List[Any]] = None
):
    """
    Run evaluation asynchronously by calling the evaluator service.
    """
    logger.info(f"Starting async evaluation {evaluation_id} for experiment {experiment_id}")
    
    # Get evaluator service URL
    evaluator_service_url = settings.evaluator_service_url or "http://localhost:9002"
    
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
    
    evaluation_results = []
    missing_evaluators = []
    
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
            missing_evaluators = []  # Initialize to empty if fetch fails
        
        # Run each evaluator
        for evaluator_id in evaluator_ids:
            if evaluator_id in missing_evaluators:
                continue
                
            try:
                logger.info(f"Running evaluator {evaluator_id}")
                
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
    
    # Update evaluation record
    if db_evaluations:
        # Tier 2+: Update database records
        from app.core.database import AsyncSessionLocal
        from app.services.evaluation_service import EvaluationService
        
        async with AsyncSessionLocal() as db_session:
            eval_service = EvaluationService(db_session)
            
            for i, evaluator_id in enumerate(evaluator_ids):
                if i < len(db_evaluations):
                    db_eval = db_evaluations[i]
                    # Find the corresponding result
                    result = next((r for r in evaluation_results if r["evaluator_id"] == evaluator_id), None)
                    
                    if result:
                        # Update the evaluation in database
                        await eval_service.complete_evaluation(
                            evaluation_id=db_eval.id,
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
                        # Mark as failed if no result
                        await eval_service.fail_evaluation(
                            evaluation_id=db_eval.id,
                            error_message="Evaluation failed to produce results"
                        )
    else:
        # Tier 1: Update in-memory storage
        if evaluation_id in evaluation_storage:
            evaluation_storage[evaluation_id]["status"] = "completed"
            evaluation_storage[evaluation_id]["evaluations"] = evaluation_results
            evaluation_storage[evaluation_id]["completed_at"] = datetime.utcnow().isoformat()
            
            # Calculate summary
            total = len(evaluation_results)
            passed = sum(1 for r in evaluation_results if r["status"] == "completed")
            failed = sum(1 for r in evaluation_results if r["status"] == "failed")
            avg_score = sum(r["score"] for r in evaluation_results if r["status"] == "completed") / max(passed, 1)
            
            evaluation_storage[evaluation_id]["summary"] = {
                "total_evaluated": total,
                "passed": passed,
                "failed": failed,
                "average_score": avg_score
            }
    
    logger.info(f"Completed evaluation {evaluation_id} with {len(evaluation_results)} results")