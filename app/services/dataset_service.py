"""
Dataset service layer for database operations.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from app.core.logging import logger
from app.models.dataset import Dataset, DatasetStatus, DatasetType
from app.schemas.dataset import DatasetCreate, DatasetUpdate


class DatasetService:
    """Service class for dataset database operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_datasets(
        self,
        page: int = 1,
        page_size: int = 20,
        type: Optional[str] = None,
        status: Optional[str] = None,
        search: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List datasets with pagination and filtering.
        """
        query = select(Dataset).where(Dataset.status != DatasetStatus.DELETED)
        
        # Apply filters
        if type:
            query = query.where(Dataset.type == type)
        if status:
            query = query.where(Dataset.status == status)
        if search:
            search_pattern = f"%{search}%"
            query = query.where(
                or_(
                    Dataset.name.ilike(search_pattern),
                    Dataset.description.ilike(search_pattern)
                )
            )
        
        # Get total count
        count_query = select(func.count()).select_from(Dataset).where(Dataset.status != DatasetStatus.DELETED)
        if type:
            count_query = count_query.where(Dataset.type == type)
        if status:
            count_query = count_query.where(Dataset.status == status)
        if search:
            count_query = count_query.where(
                or_(
                    Dataset.name.ilike(search_pattern),
                    Dataset.description.ilike(search_pattern)
                )
            )
        
        total_result = await self.db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        query = query.order_by(Dataset.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await self.db.execute(query)
        datasets = result.scalars().all()
        
        return {
            "datasets": [d.to_dict() for d in datasets],
            "total": total,
            "page": page,
            "page_size": page_size
        }
    
    async def get_dataset(self, dataset_id: UUID) -> Optional[Dataset]:
        """
        Get a dataset by ID.
        """
        query = select(Dataset).where(
            and_(
                Dataset.id == dataset_id,
                Dataset.status != DatasetStatus.DELETED
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def create_dataset(
        self,
        dataset_data: DatasetCreate,
        created_by: str = "system"
    ) -> Dataset:
        """
        Create a new dataset.
        """
        dataset = Dataset(
            name=dataset_data.name,
            description=dataset_data.description,
            type=DatasetType(dataset_data.type) if dataset_data.type else DatasetType.CUSTOM,
            data=[d.model_dump() if hasattr(d, 'model_dump') else d for d in dataset_data.data],
            record_count=len(dataset_data.data),
            tags=dataset_data.tags,
            metadata=dataset_data.metadata,
            created_by=created_by
        )
        
        try:
            self.db.add(dataset)
            await self.db.commit()
            await self.db.refresh(dataset)
            logger.info(f"Created dataset: {dataset.id}")
            return dataset
        except IntegrityError as e:
            await self.db.rollback()
            logger.error(f"Failed to create dataset: {e}")
            raise
    
    async def update_dataset(
        self,
        dataset_id: UUID,
        update_data: DatasetUpdate,
        updated_by: str = "system"
    ) -> Optional[Dataset]:
        """
        Update an existing dataset.
        """
        dataset = await self.get_dataset(dataset_id)
        if not dataset:
            return None
        
        # Update fields
        if update_data.name is not None:
            dataset.name = update_data.name
        if update_data.description is not None:
            dataset.description = update_data.description
        if update_data.data is not None:
            dataset.data = update_data.data if isinstance(update_data.data, list) else [d.model_dump() for d in update_data.data]
            dataset.record_count = len(update_data.data)
        if update_data.tags is not None:
            dataset.tags = update_data.tags
        if update_data.metadata is not None:
            dataset.metadata = update_data.metadata
        
        dataset.updated_by = updated_by
        
        try:
            await self.db.commit()
            await self.db.refresh(dataset)
            logger.info(f"Updated dataset: {dataset_id}")
            return dataset
        except IntegrityError as e:
            await self.db.rollback()
            logger.error(f"Failed to update dataset: {e}")
            raise
    
    async def delete_dataset(
        self,
        dataset_id: UUID,
        deleted_by: str = "system"
    ) -> bool:
        """
        Soft delete a dataset.
        """
        dataset = await self.get_dataset(dataset_id)
        if not dataset:
            return False
        
        dataset.status = DatasetStatus.DELETED
        dataset.updated_by = deleted_by
        
        await self.db.commit()
        logger.info(f"Deleted dataset: {dataset_id}")
        return True
    
    async def get_dataset_entries(
        self,
        dataset_id: UUID,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get dataset entries with pagination.
        """
        dataset = await self.get_dataset(dataset_id)
        if not dataset:
            return {
                "entries": [],
                "total": 0,
                "offset": offset,
                "limit": limit
            }
        
        entries = dataset.data or []
        total = len(entries)
        
        if limit:
            entries = entries[offset:offset + limit]
        else:
            entries = entries[offset:]
        
        return {
            "entries": entries,
            "total": total,
            "offset": offset,
            "limit": limit
        }
    
    async def duplicate_dataset(
        self,
        dataset_id: UUID,
        new_name: str,
        created_by: str = "system"
    ) -> Optional[Dataset]:
        """
        Create a copy of an existing dataset.
        """
        original = await self.get_dataset(dataset_id)
        if not original:
            return None
        
        new_dataset = Dataset(
            name=new_name,
            description=original.description,
            type=original.type,
            data=original.data.copy() if original.data else [],
            record_count=original.record_count,
            tags=original.tags.copy() if original.tags else [],
            metadata=original.metadata.copy() if original.metadata else {},
            created_by=created_by
        )
        
        try:
            self.db.add(new_dataset)
            await self.db.commit()
            await self.db.refresh(new_dataset)
            logger.info(f"Duplicated dataset {dataset_id} as {new_dataset.id}")
            return new_dataset
        except IntegrityError as e:
            await self.db.rollback()
            logger.error(f"Failed to duplicate dataset: {e}")
            raise