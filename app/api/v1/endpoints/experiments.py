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
from app.utils.csv_parser import csv_parser
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
    ExperimentExecuteRequest,
    AutomatedExecutionRequest,
    HTTPEndpointConfig
)

router = APIRouter()


# ==========================================
# In-Memory Storage (Tier 1)
# ==========================================
experiments_storage: Dict[str, Dict[str, Any]] = {}
experiment_results_storage: Dict[str, Dict[str, Any]] = {}

# Upload status tracking for experiment CSV imports
experiment_upload_status: Dict[str, Dict[str, Any]] = {}


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
        
        # Auto-trigger experiment if it has an endpoint URL
        endpoint_url = None
        if experiment.agent_config and experiment.agent_config.parameters:
            endpoint_url = experiment.agent_config.parameters.get("endpoint_url")
        if not endpoint_url and experiment.metadata:
            endpoint_url = experiment.metadata.get("endpoint_url")
        
        if endpoint_url and experiment.dataset_id:
            logger.info(f"Auto-triggering experiment {created.id} with endpoint {endpoint_url}")
            
            # Load dataset to get test cases
            from app.services.dataset_service import DatasetService
            dataset_service = DatasetService(db)
            dataset = await dataset_service.get_dataset(experiment.dataset_id)
            
            if dataset and dataset.data:
                # Transform localhost URLs to use Docker container names
                if "localhost:8003" in endpoint_url:
                    endpoint_url = endpoint_url.replace("localhost:8003", "pilot-api:8000")
                elif "localhost:8000" in endpoint_url and "/api/v1/guardrails" in endpoint_url:
                    endpoint_url = endpoint_url.replace("localhost:8000", "pilot-api:8000")
                
                # Prepare test cases from dataset
                test_cases = []
                for idx, item in enumerate(dataset.data[:100], 1):  # Limit to 100 test cases
                    test_case = {
                        "test_id": f"test_{idx}",
                        "input": item.get("input", ""),
                        "expected_output": item.get("expected_output", ""),
                        "context": item.get("context", []),
                        "metadata": item.get("metadata", {})
                    }
                    test_cases.append(test_case)
                
                # Queue the experiment runner task
                try:
                    from app.workers.tasks.experiment_runner_tasks import run_automated_experiment_task
                    
                    endpoint_config = {
                        "url": endpoint_url,
                        "headers": {},
                        "timeout": 30,
                        "retry_attempts": 2,
                        "retry_delay": 1,
                        "batch_size": 10
                    }
                    
                    task = run_automated_experiment_task.delay(
                        experiment_id=str(created.id),
                        endpoint_config=endpoint_config,
                        test_cases=test_cases,
                        batch_size=10,
                        experiment_name=created.name
                    )
                    
                    logger.info(f"Queued automated experiment task {task.id} for experiment {created.id}")
                except Exception as e:
                    logger.error(f"Failed to queue experiment task: {e}")
                    logger.error(f"Error type: {type(e).__name__}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Update experiment status to running
                await service.update_experiment_status(
                    experiment_id=created.id,
                    status=ExperimentStatus.RUNNING,
                    progress=0.0
                )
                
                # Refresh the experiment to get the updated status
                await db.refresh(created)
        
        return created




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
    description="Queue evaluation tasks for completed experiment"
)
async def evaluate_experiment(
    experiment_id: UUID,
    request: Dict[str, Any],
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Queue evaluation tasks for an experiment using Celery.
    Returns immediately with task ID for tracking progress.
    """
    logger.info(f"Queuing evaluation for experiment: {experiment_id}")
    
    evaluator_ids = request.get("evaluator_ids", [])
    
    if not evaluator_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="evaluator_ids array is required"
        )
    
    # Import Celery task here to avoid circular imports
    from app.workers.tasks import evaluate_experiment_task
    
    # Get the experiment and test results
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
    
    if not test_results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No test results available for evaluation"
        )
    
    # Create evaluation records in database FIRST
    eval_service = EvaluationService(db)
    created_evaluations = []
    evaluation_ids = []
    
    # Create evaluation records without task ID initially
    for evaluator_id in evaluator_ids:
        evaluation = await eval_service.create_evaluation(
            experiment_id=experiment_id,
            evaluator_id=evaluator_id,
            evaluator_name=evaluator_id.replace("_", " ").title(),
            evaluator_config={},
            created_by=current_user.id if current_user else "system",
            task_id=None  # Don't set task_id yet
        )
        created_evaluations.append(evaluation)
        evaluation_ids.append(str(evaluation.id))
    
    # Now queue the evaluation task to Celery with the evaluation IDs
    task = evaluate_experiment_task.delay(
        evaluation_ids=evaluation_ids,  # Pass the actual evaluation IDs
        experiment_id=str(experiment_id),
        evaluator_ids=evaluator_ids,
        test_results=test_results,
        experiment_name=experiment.get("name", "Unknown")
    )
    
    # Update evaluations with the task ID
    for evaluation in created_evaluations:
        await eval_service.update_task_id(evaluation.id, task.id)
    
    logger.info(f"Queued evaluation task {task.id} for experiment {experiment_id}")
    
    # Return task info for tracking
    return {
        "task_id": task.id,
        "evaluation_ids": evaluation_ids,
        "experiment_id": str(experiment_id),
        "status": "queued",
        "message": "Evaluation queued successfully",
        "evaluator_ids": evaluator_ids,
        "websocket_url": f"/ws/evaluations/{task.id}"
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


async def process_experiment_csv_in_background(
    upload_id: str,
    file_content: bytes,
    filename: str,
    experiment_params: Dict[str, Any],
    user_id: Optional[str],
    db: AsyncSession
):
    """
    Process experiment CSV file in background for large files.
    """
    try:
        # Update status to processing
        experiment_upload_status[upload_id] = {
            "status": "processing",
            "message": "Processing CSV file...",
            "progress": 0
        }
        
        # Create a fake UploadFile object for the parser
        file_like = io.BytesIO(file_content)
        upload_file = UploadFile(
            filename=filename,
            file=file_like
        )
        
        # Parse CSV with streaming parser
        test_results, warnings, metadata = await csv_parser.parse_experiment_results(upload_file)
        
        if not test_results:
            experiment_upload_status[upload_id] = {
                "status": "failed",
                "message": "No valid data could be parsed from the CSV file",
                "error": "Empty results",
                "warnings": warnings
            }
            return
        
        # Create experiment
        from app.schemas.experiment import ExperimentCreate, AgentConfig
        from uuid import UUID as UUID_type
        
        experiment_create = ExperimentCreate(
            name=experiment_params['name'],
            description=experiment_params.get('description') or f"Imported from {filename}",
            dataset_id=UUID_type(experiment_params['dataset_id']) if experiment_params.get('dataset_id') and experiment_params['dataset_id'] != 'null' else None,
            agent_config=AgentConfig(
                model="imported",
                provider="csv",
                temperature=0.0,
                max_tokens=1,
                system_prompt="",
                custom_fields={
                    "source": filename,
                    "upload_id": upload_id,
                    "rows_processed": len(test_results)
                }
            ),
            execution_mode="batch" if experiment_params.get('execution_mode') == "import" else experiment_params.get('execution_mode', 'batch'),
            tags=["imported", "async_upload"],
            metadata={
                **metadata,
                "source": "csv_import",
                "filename": filename,
                "record_count": len(test_results),
                "upload_id": upload_id,
                "warnings": warnings[:10] if warnings else []
            }
        )
        
        service = ExperimentService(db)
        experiment = await service.create_experiment(
            experiment_create,
            created_by=user_id or "import"
        )
        
        # Add test results
        from app.models.experiment import TestResult, ExperimentStatus
        for result_data in test_results:
            test_result = TestResult(
                experiment_id=experiment.id,
                test_id=result_data.get("test_id"),
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
        
        # Update upload status to completed
        experiment_upload_status[upload_id] = {
            "status": "completed",
            "message": f"Successfully processed {len(test_results)} test results",
            "experiment_id": str(experiment.id),
            "warnings": warnings,
            "metadata": metadata
        }
        
        logger.info(f"Background CSV processing completed: {len(test_results)} results imported to experiment {experiment.id}")
        
    except Exception as e:
        logger.error(f"Background CSV processing failed: {e}")
        experiment_upload_status[upload_id] = {
            "status": "failed",
            "message": f"Processing failed: {str(e)}",
            "error": str(e)
        }


@router.post(
    "/import-csv",
    response_model=ExperimentResponse,
    summary="Import CSV results",
    description="Import experiment results from a CSV file with streaming support"
)
async def import_csv_results(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file with experiment results"),
    name: str = Form(..., description="Name for the experiment"),
    description: Optional[str] = Form(None, description="Description for the experiment"),
    dataset_id: Optional[str] = Form(None, description="Associated dataset ID"),
    execution_mode: str = Form("import", description="Execution mode"),
    async_processing: bool = Form(False, description="Process large files in background"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Import experiment results from a CSV file with production-ready streaming.
    
    **Features:**
    - Streaming parsing for memory efficiency
    - Chunked processing (1000 rows at a time)
    - Background processing for large files (>5MB)
    - Comprehensive validation with detailed warnings
    
    **Supported CSV columns:**
    - test_case_id or test_id: Unique test identifier
    - input, input_prompt, or prompt: Test input
    - expected_output: Expected result
    - actual_output or output: Actual result from model
    - context: Additional context (JSON array or comma-separated)
    - latency_ms or execution_time: Performance metrics
    - token_usage_input/output: Token usage metrics
    - meta_* fields: Custom metadata
    - retrieval_context: Retrieved documents (JSON)
    - tools_called: Tools used (JSON)
    """
    logger.info(f"Importing CSV results for experiment: {name} (async={async_processing})")
    
    # Validate file type
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a CSV file"
        )
    
    # Check file size for async decision
    file_content = await file.read()
    file_size = len(file_content)
    
    # Reset file position
    file.file.seek(0)
    
    # Auto-enable async for large files
    if file_size > 5 * 1024 * 1024 and not async_processing:
        logger.info(f"File size {file_size} bytes, auto-enabling async processing")
        async_processing = True
    
    if async_processing:
        # Process in background for large files
        upload_id = str(uuid.uuid4())
        
        # Store initial status
        experiment_upload_status[upload_id] = {
            "status": "processing",
            "message": "Upload received, processing in background...",
            "filename": file.filename,
            "file_size": file_size
        }
        
        # Add background task
        background_tasks.add_task(
            process_experiment_csv_in_background,
            upload_id=upload_id,
            file_content=file_content,
            filename=file.filename,
            experiment_params={
                "name": name,
                "description": description,
                "dataset_id": dataset_id,
                "execution_mode": execution_mode
            },
            user_id=current_user.id if current_user else None,
            db=db
        )
        
        # Return immediate response with upload ID
        return {
            "id": upload_id,
            "name": name,
            "description": description or f"Processing {file.filename}...",
            "status": "processing",
            "progress": 0.0,
            "dataset_id": dataset_id,
            "agent_config": {
                "model": "imported",
                "provider": "csv",
                "temperature": 0.0,
                "max_tokens": 1,
                "system_prompt": ""
            },
            "execution_mode": execution_mode,
            "created_at": datetime.utcnow(),
            "metadata": {
                "upload_id": upload_id,
                "processing_message": f"Check status at /api/v1/experiments/import-status/{upload_id}"
            }
        }
    
    # Process synchronously for smaller files
    try:
        # Parse CSV with streaming parser
        test_results, warnings, metadata = await csv_parser.parse_experiment_results(file)
        
        if not test_results:
            error_detail = {
                "error": "CSV Import Failed",
                "message": "No valid data could be parsed from the CSV file",
                "details": warnings[:5] if warnings else ["No rows could be processed"],
                "help": "Expected CSV columns: 'input' (or 'prompt'), 'expected_output', 'actual_output', 'status'. "
                        "Please ensure your CSV has the correct column headers."
            }
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_detail
            )
        
        # Store in database
        service = ExperimentService(db)
        
        # Create experiment
        from app.schemas.experiment import ExperimentCreate, AgentConfig
        from uuid import UUID as UUID_type
        
        # Validate dataset_id if provided
        validated_dataset_id = None
        if dataset_id and dataset_id != 'null' and dataset_id != '' and dataset_id != 'undefined':
            try:
                # Check if it's a valid UUID
                validated_dataset_id = UUID_type(dataset_id)
                # Always check if dataset exists to avoid foreign key issues
                from app.services.dataset_service import DatasetService
                dataset_service = DatasetService(db)
                try:
                    dataset = await dataset_service.get_dataset(validated_dataset_id)
                    if not dataset:
                        logger.warning(f"Dataset {validated_dataset_id} not found, will create placeholder dataset")
                        # Create a placeholder dataset for the import
                        from app.schemas.dataset import DatasetCreate
                        placeholder_dataset = DatasetCreate(
                            name=f"Import Dataset for {name}",
                            description=f"Auto-created dataset for imported experiment {name}",
                            type="imported",
                            schema_version="1.0",
                            items=[],
                            metadata={"auto_created": True, "source": "csv_import"}
                        )
                        created_dataset = await dataset_service.create_dataset(
                            placeholder_dataset,
                            created_by=current_user.id if current_user else "import"
                        )
                        validated_dataset_id = created_dataset.id
                        logger.info(f"Created placeholder dataset {validated_dataset_id} for import")
                except Exception as e:
                    logger.warning(f"Could not verify or create dataset: {e}")
                    validated_dataset_id = None
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid dataset_id format: {dataset_id}")
                validated_dataset_id = None
        
        experiment_create = ExperimentCreate(
            name=name,
            description=description or f"Imported from {file.filename}",
            dataset_id=validated_dataset_id,
            agent_config=AgentConfig(
                model="imported",
                provider="csv",
                temperature=0.0,
                max_tokens=1,  # Use minimum valid value
                system_prompt="",
                custom_fields={
                    "source": file.filename,
                    "rows_processed": len(test_results),
                    "format": metadata.get('format_type')
                }
            ),
            execution_mode="batch" if execution_mode == "import" else execution_mode,
            tags=["imported"],
            metadata={
                **metadata,
                "source": "csv_import",
                "filename": file.filename,
                "record_count": len(test_results),
                "warnings": warnings[:10] if warnings else []  # Include first 10 warnings
            }
        )
        
        experiment = await service.create_experiment(
            experiment_create,
            created_by=current_user.id if current_user else "import"
        )
        
        # Add test results directly
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
        
        # Convert to response directly from SQLAlchemy model
        # This avoids issues with JSONB serialization in psycopg3
        logger.info(f"Successfully imported {len(test_results)} results into experiment {experiment.id}")
        
        # Return the SQLAlchemy model directly - FastAPI will handle conversion
        return experiment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV import failed: {e}", exc_info=True)
        error_detail = {
            "error": "CSV Processing Error",
            "message": "Failed to process the CSV file",
            "details": str(e),
            "help": "Please check that your CSV file: 1) Has proper column headers, 2) Contains valid data in each row, "
                    "3) Uses UTF-8 encoding. Required columns: 'input', 'expected_output' (optional), 'actual_output' (optional)"
        }
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )


@router.get(
    "/import-status/{upload_id}",
    summary="Check CSV import status",
    description="Check the status of a background CSV import"
)
async def check_import_status(
    upload_id: str,
    current_user = Depends(get_current_user_optional)
):
    """
    Check the status of a background CSV import.
    
    Returns the current status and any available results.
    """
    status = experiment_upload_status.get(upload_id)
    
    if not status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload {upload_id} not found"
        )
    
    # If completed, include the experiment details
    if status.get("status") == "completed" and status.get("experiment_id"):
        # In production, would fetch from database
        # For now, just return the status with experiment_id
        return {
            **status,
            "experiment_url": f"/api/v1/experiments/{status['experiment_id']}"
        }
    
    return status


@router.get(
    "/tasks/{task_id}/status",
    summary="Get evaluation task status",
    description="Get the status of a queued or running evaluation task"
)
async def get_task_status(
    task_id: str,
    current_user = Depends(get_current_user_optional)
):
    """
    Get the status of an evaluation task.
    """
    from app.services.queue_service import QueueService
    
    queue_service = QueueService()
    status = await queue_service.get_task_status(task_id)
    
    return status


@router.delete(
    "/tasks/{task_id}",
    summary="Cancel evaluation task",
    description="Cancel a queued or running evaluation task"
)
async def cancel_task(
    task_id: str,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Cancel an evaluation task.
    """
    from app.services.queue_service import QueueService
    
    queue_service = QueueService()
    cancelled = await queue_service.cancel_task(task_id, terminate=True)
    
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Task could not be cancelled (may be already completed)"
        )
    
    # Update evaluation status in database
    if db:
        eval_service = EvaluationService(db)
        evaluations = await eval_service.get_evaluations_by_task_id(task_id)
        for evaluation in evaluations:
            await eval_service.fail_evaluation(
                evaluation.id,
                "Task cancelled by user"
            )
    
    return {
        "task_id": task_id,
        "status": "cancelled",
        "message": "Task cancelled successfully"
    }


@router.get(
    "/tasks/queue/stats",
    summary="Get queue statistics",
    description="Get statistics about the evaluation task queue"
)
async def get_queue_stats(
    current_user = Depends(get_current_user_optional)
):
    """
    Get queue statistics including active tasks, workers, etc.
    """
    from app.services.queue_service import QueueService
    
    queue_service = QueueService()
    stats = await queue_service.get_queue_stats()
    
    return stats


@router.post(
    "/validate-experiment-csv",
    summary="Validate experiment CSV file",
    description="Validate an experiment CSV file without importing"
)
async def validate_experiment_csv(
    file: UploadFile = File(..., description="CSV file to validate"),
    current_user = Depends(get_current_user_optional)
):
    """
    Validate an experiment CSV file without creating an experiment.
    
    Useful for checking format and data before actual import.
    """
    logger.info(f"Validating experiment CSV file: {file.filename}")
    
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a CSV file"
        )
    
    try:
        # Parse with validation only
        items, warnings, metadata = await csv_parser.parse_experiment_results(file, validate_only=True)
        
        return {
            "valid": len(items) > 0,
            "format_type": "experiment_results",
            "total_rows": metadata.get('total_rows'),
            "valid_items": len(items),
            "warnings": warnings,
            "sample": items[:5] if items else [],  # Include sample of parsed items
            "detected_fields": list(items[0].keys()) if items else []
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Experiment CSV validation failed: {e}")
        return {
            "valid": False,
            "error": str(e),
            "warnings": csv_parser.warnings
        }


@router.post(
    "/{experiment_id}/run-automated",
    response_model=Dict[str, Any],
    summary="Run automated experiment execution",
    description="Execute experiment test cases through HTTP endpoint with retry logic"
)
async def run_automated_experiment(
    experiment_id: UUID,
    request: AutomatedExecutionRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Run automated experiment execution through HTTP endpoint.
    
    This endpoint:
    1. Validates experiment exists and has test cases
    2. Queues experiment runner task with Celery
    3. Executes test cases against provided HTTP endpoint
    4. Stores actual outputs and updates test results
    5. Optionally triggers evaluations after completion
    
    **Features:**
    - Concurrent request execution with configurable batch size
    - Automatic retry with exponential backoff
    - Real-time progress updates via WebSocket
    - Result storage compatible with evaluation system
    """
    logger.info(f"Starting automated execution for experiment {experiment_id}")
    
    # Get experiment service
    service = ExperimentService(db)
    
    # Validate experiment exists
    experiment = await service.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )
    
    # Check experiment status
    if experiment.status == ExperimentStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Experiment is already running"
        )
    
    # Get test results (input data from CSV)
    test_results = await service.get_test_results(experiment_id)
    if not test_results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No test cases found for this experiment. Please upload CSV data first."
        )
    
    # Prepare test cases for execution
    test_cases = []
    for result in test_results:
        test_case = {
            "test_id": result.get("test_id", str(uuid.uuid4())),
            "input": result.get("input", ""),
            "expected_output": result.get("expected_output", ""),
            "context": result.get("context", []),
            "metadata": result.get("metadata", {})
        }
        test_cases.append(test_case)
    
    # Update experiment status to running
    await service.update_experiment_status(
        experiment_id=experiment_id,
        status=ExperimentStatus.RUNNING,
        progress=0.0
    )
    
    # Queue the experiment runner task
    from app.workers.tasks.experiment_runner_tasks import run_automated_experiment_task
    
    task = run_automated_experiment_task.delay(
        experiment_id=str(experiment_id),
        endpoint_config=request.endpoint_config.model_dump(),
        test_cases=test_cases,
        batch_size=request.endpoint_config.batch_size,
        experiment_name=experiment.name
    )
    
    logger.info(f"Queued automated experiment task {task.id} for experiment {experiment_id}")
    
    # If evaluations are requested, store the config for post-execution
    if request.run_evaluations and request.evaluator_ids:
        experiment.metadata["pending_evaluations"] = {
            "evaluator_ids": request.evaluator_ids,
            "trigger_after_execution": True
        }
        await db.commit()
    
    return {
        "message": "Automated experiment execution started",
        "experiment_id": str(experiment_id),
        "task_id": task.id,
        "status": "running",
        "test_count": len(test_cases),
        "endpoint_url": request.endpoint_config.url,
        "batch_size": request.endpoint_config.batch_size,
        "run_evaluations": request.run_evaluations,
        "websocket_url": f"/ws/experiments/{task.id}"
    }


@router.get(
    "/{experiment_id}/execution-status/{task_id}",
    response_model=Dict[str, Any],
    summary="Get automated execution status",
    description="Check the status of an automated experiment execution task"
)
async def get_execution_status(
    experiment_id: UUID,
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Get the status of an automated experiment execution task.
    """
    from celery.result import AsyncResult
    from app.workers.celery_app import celery_app
    
    # Check task status
    result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "experiment_id": str(experiment_id),
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None
    }
    
    # Add result data if available
    if result.ready() and result.successful():
        task_result = result.result
        response.update({
            "summary": task_result.get("summary"),
            "completed_at": task_result.get("completed_at"),
            "results_count": len(task_result.get("results", []))
        })
    elif result.failed():
        response["error"] = str(result.info)
    
    return response


@router.post(
    "/{experiment_id}/cancel-execution/{task_id}",
    response_model=Dict[str, Any],
    summary="Cancel automated execution",
    description="Cancel a running automated experiment execution"
)
async def cancel_execution(
    experiment_id: UUID,
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Cancel a running automated experiment execution.
    """
    from app.workers.tasks.experiment_runner_tasks import cancel_experiment_execution_task
    
    # Cancel the task
    success = cancel_experiment_execution_task(task_id)
    
    if success:
        # Update experiment status
        service = ExperimentService(db)
        await service.update_experiment_status(
            experiment_id=experiment_id,
            status=ExperimentStatus.CANCELLED
        )
        
        return {
            "message": "Experiment execution cancelled",
            "experiment_id": str(experiment_id),
            "task_id": task_id,
            "status": "cancelled"
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel experiment execution"
        )


# Note: The old run_evaluation_async function has been replaced with Celery tasks
# See app/workers/tasks/evaluation_tasks.py for the new implementation