"""
Experiment service layer for database operations.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from sqlalchemy.exc import IntegrityError
from fastapi import BackgroundTasks

from app.core.logging import logger
from app.core.config import settings
from app.models.experiment import Experiment, ExperimentStatus, ExecutionMode, TestResult
from app.models.dataset import Dataset
from app.models.evaluation import Evaluation
from app.schemas.experiment import ExperimentCreate, ExperimentUpdate


class ExperimentService:
    """Service class for experiment database operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_experiments(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[ExperimentStatus] = None,
        dataset_id: Optional[UUID] = None,
        search: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List experiments with pagination and filtering.
        """
        query = select(Experiment).options(joinedload(Experiment.dataset))
        
        # Apply filters
        if status:
            query = query.where(Experiment.status == status)
        if dataset_id:
            query = query.where(Experiment.dataset_id == dataset_id)
        if search:
            search_pattern = f"%{search}%"
            query = query.where(
                or_(
                    Experiment.name.ilike(search_pattern),
                    Experiment.description.ilike(search_pattern)
                )
            )
        
        # Get total count
        count_query = select(func.count()).select_from(Experiment)
        if status:
            count_query = count_query.where(Experiment.status == status)
        if dataset_id:
            count_query = count_query.where(Experiment.dataset_id == dataset_id)
        if search:
            count_query = count_query.where(
                or_(
                    Experiment.name.ilike(search_pattern),
                    Experiment.description.ilike(search_pattern)
                )
            )
        
        total_result = await self.db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        query = query.order_by(Experiment.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await self.db.execute(query)
        experiments = result.scalars().unique().all()
        
        # Format response
        experiments_data = []
        for exp in experiments:
            exp_dict = exp.to_dict()
            if exp.dataset:
                exp_dict["dataset_name"] = exp.dataset.name
                exp_dict["dataset_size"] = exp.dataset.record_count
            experiments_data.append(exp_dict)
        
        return {
            "experiments": experiments_data,
            "total": total,
            "page": page,
            "page_size": page_size
        }
    
    async def get_experiment(self, experiment_id: UUID) -> Optional[Experiment]:
        """
        Get an experiment by ID.
        """
        query = select(Experiment).options(
            joinedload(Experiment.dataset)
        ).where(Experiment.id == experiment_id)
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def create_experiment(
        self,
        experiment_data: ExperimentCreate,
        created_by: str = "system"
    ) -> Experiment:
        """
        Create a new experiment.
        """
        experiment = Experiment(
            name=experiment_data.name,
            description=experiment_data.description,
            dataset_id=experiment_data.dataset_id,
            agent_config=experiment_data.agent_config.model_dump() if hasattr(experiment_data.agent_config, 'model_dump') else experiment_data.agent_config,
            execution_mode=ExecutionMode(experiment_data.execution_mode) if experiment_data.execution_mode else ExecutionMode.BATCH,
            tags=experiment_data.tags,
            metadata=experiment_data.metadata,
            created_by=created_by,
            status=ExperimentStatus.PENDING
        )
        
        try:
            self.db.add(experiment)
            await self.db.commit()
            await self.db.refresh(experiment)
            logger.info(f"Created experiment: {experiment.id}")
            return experiment
        except IntegrityError as e:
            await self.db.rollback()
            logger.error(f"Failed to create experiment: {e}")
            raise
    
    async def update_experiment(
        self,
        experiment_id: UUID,
        update_data: ExperimentUpdate,
        updated_by: str = "system"
    ) -> Optional[Experiment]:
        """
        Update an existing experiment.
        """
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            return None
        
        # Don't allow updates to running experiments
        if experiment.status == ExperimentStatus.RUNNING:
            logger.warning(f"Cannot update running experiment: {experiment_id}")
            return None
        
        # Update fields
        if update_data.name is not None:
            experiment.name = update_data.name
        if update_data.description is not None:
            experiment.description = update_data.description
        if update_data.agent_config is not None:
            experiment.agent_config = update_data.agent_config.model_dump() if hasattr(update_data.agent_config, 'model_dump') else update_data.agent_config
        if update_data.tags is not None:
            experiment.tags = update_data.tags
        if update_data.metadata is not None:
            experiment.metadata = update_data.metadata
        
        experiment.updated_by = updated_by
        
        try:
            await self.db.commit()
            await self.db.refresh(experiment)
            logger.info(f"Updated experiment: {experiment_id}")
            return experiment
        except IntegrityError as e:
            await self.db.rollback()
            logger.error(f"Failed to update experiment: {e}")
            raise
    
    async def delete_experiment(
        self,
        experiment_id: UUID,
        deleted_by: str = "system"
    ) -> bool:
        """
        Delete an experiment.
        """
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            return False
        
        # Don't allow deletion of running experiments
        if experiment.status == ExperimentStatus.RUNNING:
            logger.warning(f"Cannot delete running experiment: {experiment_id}")
            return False
        
        # Delete related evaluations first (they have foreign key to experiment)
        evaluations_query = select(Evaluation).where(Evaluation.experiment_id == experiment_id)
        evaluations = await self.db.execute(evaluations_query)
        for evaluation in evaluations.scalars():
            await self.db.delete(evaluation)
        
        # Delete related test results using ORM
        test_results_query = select(TestResult).where(TestResult.experiment_id == experiment_id)
        test_results = await self.db.execute(test_results_query)
        for test_result in test_results.scalars():
            await self.db.delete(test_result)
        
        # Delete the experiment
        await self.db.delete(experiment)
        await self.db.commit()
        
        logger.info(f"Deleted experiment: {experiment_id}")
        return True
    
    async def execute_experiment(
        self,
        experiment_id: UUID,
        batch_size: Optional[int] = None,
        timeout: Optional[int] = None,
        evaluator_ids: Optional[List[str]] = None,
        background_tasks: Optional[BackgroundTasks] = None
    ) -> Dict[str, Any]:
        """
        Execute an experiment.
        """
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if experiment.status in [ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED]:
            raise ValueError(f"Experiment is already {experiment.status.value}")
        
        # Update status to running
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.utcnow()
        experiment.progress = 0.0
        
        await self.db.commit()
        
        # In production, this would trigger actual execution
        # For now, we'll just return a status message
        logger.info(f"Started execution of experiment: {experiment_id}")
        
        # If background tasks are enabled, we would queue the actual execution
        if background_tasks and settings.enable_background_tasks:
            background_tasks.add_task(
                self._execute_experiment_async,
                experiment_id,
                batch_size,
                timeout,
                evaluator_ids
            )
        
        return {
            "message": "Experiment execution started",
            "experiment_id": str(experiment_id),
            "status": "running"
        }
    
    async def _execute_experiment_async(
        self,
        experiment_id: UUID,
        batch_size: Optional[int],
        timeout: Optional[int],
        evaluator_ids: Optional[List[str]]
    ):
        """
        Async execution of experiment (background task).
        """
        # This would contain the actual execution logic
        # For now, it's a placeholder
        pass
    
    async def cancel_experiment(self, experiment_id: UUID) -> Optional[Experiment]:
        """
        Cancel a running experiment.
        """
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            return None
        
        if experiment.status != ExperimentStatus.RUNNING:
            logger.warning(f"Cannot cancel experiment with status: {experiment.status}")
            return None
        
        experiment.status = ExperimentStatus.CANCELLED
        experiment.completed_at = datetime.utcnow()
        
        await self.db.commit()
        logger.info(f"Cancelled experiment: {experiment_id}")
        return experiment
    
    async def get_experiment_results(self, experiment_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get experiment results.
        """
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            return None
        
        # Get test results
        query = select(TestResult).where(TestResult.experiment_id == experiment_id)
        result = await self.db.execute(query)
        test_results = result.scalars().all()
        
        # Calculate summary statistics
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.status == "passed")
        failed_tests = sum(1 for r in test_results if r.status == "failed")
        error_tests = sum(1 for r in test_results if r.status == "error")
        
        # Calculate execution time
        execution_times = [r.execution_time for r in test_results if r.execution_time]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "experiment_id": str(experiment_id),
            "status": experiment.status.value if experiment.status else "unknown",
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "error_tests": error_tests,
            "execution_time": avg_execution_time,
            "results": [
                {
                    "test_id": r.test_id,
                    "input": r.input,
                    "expected_output": r.expected_output,
                    "actual_output": r.actual_output,
                    "status": r.status,
                    "execution_time": r.execution_time,
                    "error": r.error,
                    "meta_data": r.meta_data if r.meta_data else {}
                }
                for r in test_results
            ],
            "summary": {
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "avg_execution_time": avg_execution_time
            }
        }
    
    async def add_test_results(self, experiment_id: UUID, test_results: List[Dict[str, Any]]) -> bool:
        """
        Add test results to an experiment.
        """
        # Get experiment without joinedload to avoid async issues
        query = select(Experiment).where(Experiment.id == experiment_id)
        result = await self.db.execute(query)
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Create TestResult objects
        for result_data in test_results:
            test_result = TestResult(
                experiment_id=experiment_id,
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
            self.db.add(test_result)
        
        # Update experiment status
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = datetime.utcnow()
        experiment.progress = 1.0
        
        await self.db.commit()
        logger.info(f"Added {len(test_results)} test results to experiment {experiment_id}")
        return True