"""
Evaluation service for managing evaluation database operations.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.evaluation import Evaluation

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service class for evaluation operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_evaluation(
        self,
        experiment_id: UUID,
        evaluator_id: str,
        evaluator_name: str,
        evaluator_config: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> Evaluation:
        """Create a new evaluation."""
        evaluation = Evaluation(
            experiment_id=experiment_id,
            evaluator_id=evaluator_id,
            evaluator_name=evaluator_name,
            evaluator_config=evaluator_config or {},
            status="running",
            meta_data={"created_by": created_by} if created_by else {}
        )
        
        self.db.add(evaluation)
        await self.db.commit()
        await self.db.refresh(evaluation)
        
        logger.info(f"Created evaluation {evaluation.id} for experiment {experiment_id}")
        return evaluation
    
    async def get_evaluation(self, evaluation_id: UUID) -> Optional[Evaluation]:
        """Get evaluation by ID."""
        query = select(Evaluation).where(
            Evaluation.id == evaluation_id
        ).options(selectinload(Evaluation.experiment))
        
        result = await self.db.execute(query)
        evaluation = result.scalar_one_or_none()
        
        if evaluation:
            logger.debug(f"Found evaluation {evaluation_id}")
        else:
            logger.warning(f"Evaluation {evaluation_id} not found")
        
        return evaluation
    
    async def get_evaluations_by_experiment(
        self, 
        experiment_id: UUID
    ) -> List[Evaluation]:
        """Get all evaluations for an experiment."""
        query = select(Evaluation).where(
            Evaluation.experiment_id == experiment_id
        ).order_by(Evaluation.created_at.desc())
        
        result = await self.db.execute(query)
        evaluations = result.scalars().all()
        
        logger.debug(f"Found {len(evaluations)} evaluations for experiment {experiment_id}")
        return evaluations
    
    async def get_all_evaluations(
        self,
        page: int = 1,
        limit: int = 50,
        status_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get paginated list of all evaluations."""
        offset = (page - 1) * limit
        
        # Base query
        query = select(Evaluation).options(selectinload(Evaluation.experiment))
        count_query = select(func.count(Evaluation.id))
        
        # Apply filters
        if status_filter:
            query = query.where(Evaluation.status == status_filter)
            count_query = count_query.where(Evaluation.status == status_filter)
        
        # Apply pagination
        query = query.order_by(Evaluation.created_at.desc()).offset(offset).limit(limit)
        
        # Execute queries
        result = await self.db.execute(query)
        evaluations = result.scalars().all()
        
        count_result = await self.db.execute(count_query)
        total = count_result.scalar()
        
        logger.debug(f"Retrieved {len(evaluations)} evaluations (page {page} of {(total + limit - 1) // limit})")
        
        return {
            "evaluations": evaluations,
            "total": total,
            "page": page,
            "limit": limit
        }
    
    async def update_evaluation(
        self,
        evaluation_id: UUID,
        updates: Dict[str, Any]
    ) -> Optional[Evaluation]:
        """Update an evaluation."""
        # Get existing evaluation
        evaluation = await self.get_evaluation(evaluation_id)
        if not evaluation:
            return None
        
        # Update fields
        for key, value in updates.items():
            if hasattr(evaluation, key):
                setattr(evaluation, key, value)
        
        # Update timestamp
        evaluation.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(evaluation)
        
        logger.info(f"Updated evaluation {evaluation_id}")
        return evaluation
    
    async def start_evaluation(
        self,
        evaluation_id: UUID
    ) -> Optional[Evaluation]:
        """Mark evaluation as started."""
        return await self.update_evaluation(
            evaluation_id,
            {
                "status": "running",
                "started_at": datetime.utcnow()
            }
        )
    
    async def complete_evaluation(
        self,
        evaluation_id: UUID,
        results: Dict[str, Any],
        score: Optional[float] = None,
        total_tests: int = 0,
        passed_tests: int = 0,
        failed_tests: int = 0,
        error_tests: int = 0
    ) -> Optional[Evaluation]:
        """Mark evaluation as completed with results."""
        completed_at = datetime.utcnow()
        evaluation = await self.get_evaluation(evaluation_id)
        if not evaluation:
            return None
        
        # Calculate execution time if started_at is set
        execution_time = None
        if evaluation.started_at:
            execution_time = (completed_at - evaluation.started_at).total_seconds()
        
        return await self.update_evaluation(
            evaluation_id,
            {
                "status": "completed",
                "results": results,
                "score": score,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "execution_time": execution_time,
                "completed_at": completed_at
            }
        )
    
    async def fail_evaluation(
        self,
        evaluation_id: UUID,
        error_message: str
    ) -> Optional[Evaluation]:
        """Mark evaluation as failed."""
        return await self.update_evaluation(
            evaluation_id,
            {
                "status": "failed",
                "error_message": error_message,
                "completed_at": datetime.utcnow()
            }
        )
    
    async def delete_evaluation(
        self,
        evaluation_id: UUID
    ) -> bool:
        """Delete an evaluation."""
        # Use ORM to get the evaluation
        evaluation = await self.get_evaluation(evaluation_id)
        if not evaluation:
            logger.warning(f"Evaluation {evaluation_id} not found for deletion")
            return False
        
        # Delete using ORM
        await self.db.delete(evaluation)
        await self.db.commit()
        
        logger.info(f"Deleted evaluation {evaluation_id}")
        return True
    
    async def get_evaluation_for_api_response(
        self,
        evaluation_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Get evaluation formatted for API response (Node.js compatibility)."""
        evaluation = await self.get_evaluation(evaluation_id)
        if not evaluation:
            return None
        
        # Format for API response matching Node.js structure
        return {
            "evaluation_id": str(evaluation.id),
            "experiment_id": str(evaluation.experiment_id),
            "experiment_name": evaluation.experiment.name if evaluation.experiment else "Unknown",
            "agent_name": evaluation.experiment.agent_config.get("name", "Unknown") if evaluation.experiment and evaluation.experiment.agent_config else "Unknown",
            "dataset_name": "Test Dataset",  # Would need to join with dataset table
            "evaluation_type": "comprehensive",
            "status": evaluation.status,
            "evaluations": [
                {
                    "evaluator_id": evaluation.evaluator_id,
                    "metric_name": evaluation.evaluator_name,
                    "score": evaluation.score or 0,
                    "status": evaluation.status,
                    "details": evaluation.results.get("details", []) if evaluation.results else []
                }
            ] if evaluation.results else [],
            "summary": evaluation.results.get("summary", {}) if evaluation.results else {},
            "created_at": evaluation.created_at.isoformat() if evaluation.created_at else None,
            "completed_at": evaluation.completed_at.isoformat() if evaluation.completed_at else None
        }