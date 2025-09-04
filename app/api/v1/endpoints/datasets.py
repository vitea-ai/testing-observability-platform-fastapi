"""
Datasets endpoint for managing test datasets.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, Query, Body
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
from app.services.dataset_service import DatasetService
from app.schemas.dataset import (
    DatasetItemBase,
    DatasetBase,
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    DatasetListResponse
)

router = APIRouter()


# ==========================================
# In-Memory Storage (Tier 1)
# ==========================================
# For tier 1 (development), we use in-memory storage
# This will be replaced with database operations in tier 2+
datasets_storage: Dict[str, Dict[str, Any]] = {}


# ==========================================
# API Endpoints
# ==========================================
@router.get(
    "/",
    response_model=DatasetListResponse,
    summary="List datasets",
    description="Retrieve a paginated list of datasets with optional filtering"
)
async def list_datasets(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    type: Optional[str] = Query(None, description="Filter by dataset type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    List all datasets with pagination and filtering.
    
    Returns a paginated list of datasets based on the provided filters.
    """
    logger.info(f"Listing datasets - page: {page}, page_size: {page_size}, filters: type={type}, status={status}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        filtered_datasets = list(datasets_storage.values())
        
        # Apply filters
        if type:
            filtered_datasets = [d for d in filtered_datasets if d.get("type") == type]
        if status:
            filtered_datasets = [d for d in filtered_datasets if d.get("status") == status]
        if search:
            search_lower = search.lower()
            filtered_datasets = [
                d for d in filtered_datasets
                if search_lower in d.get("name", "").lower() or
                   search_lower in d.get("description", "").lower()
            ]
        
        # Pagination
        total = len(filtered_datasets)
        start = (page - 1) * page_size
        end = start + page_size
        paginated = filtered_datasets[start:end]
        
        return DatasetListResponse(
            datasets=[DatasetResponse(**d) for d in paginated],
            total=total,
            page=page,
            page_size=page_size
        )
    else:
        # Tier 2+: Database implementation
        service = DatasetService(db)
        result = await service.list_datasets(
            page=page,
            page_size=page_size,
            type=type,
            status=status,
            search=search
        )
        return DatasetListResponse(
            datasets=[DatasetResponse(**d) for d in result["datasets"]],
            total=result["total"],
            page=result["page"],
            page_size=result["page_size"]
        )


@router.post(
    "/",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create dataset",
    description="Create a new dataset with test cases"
)
async def create_dataset(
    dataset: DatasetCreate,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user) if settings.is_feature_enabled("authentication") else Depends(get_current_user_optional)
):
    """
    Create a new dataset.
    
    Creates a dataset with the provided test cases and metadata.
    """
    logger.info(f"Creating dataset: {dataset.name}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        from uuid import uuid4
        
        dataset_id = str(uuid4())
        dataset_dict = dataset.model_dump()
        dataset_dict.update({
            "id": dataset_id,
            "status": "active",
            "record_count": len(dataset.data),
            "created_at": datetime.utcnow(),
            "updated_at": None,
            "version": "1.0.0"
        })
        
        datasets_storage[dataset_id] = dataset_dict
        
        if settings.is_feature_enabled("audit_logging"):
            await audit_log(
                action="dataset.create",
                resource_id=dataset_id,
                user=current_user.id if current_user else "anonymous",
                details={"name": dataset.name}
            )
        
        return DatasetResponse(**dataset_dict)
    else:
        # Tier 2+: Database implementation
        service = DatasetService(db)
        created = await service.create_dataset(
            dataset,
            created_by=current_user.id if current_user else "system"
        )
        return DatasetResponse(**created.to_dict())


@router.get(
    "/{dataset_id}",
    response_model=DatasetResponse,
    summary="Get dataset",
    description="Retrieve a specific dataset by ID"
)
async def get_dataset(
    dataset_id: UUID,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Get a dataset by ID.
    
    Returns the full dataset including all test cases.
    """
    logger.info(f"Getting dataset: {dataset_id}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        dataset = datasets_storage.get(str(dataset_id))
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        return DatasetResponse(**dataset)
    else:
        # Tier 2+: Database implementation
        service = DatasetService(db)
        dataset = await service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        return DatasetResponse(**dataset.to_dict())


@router.put(
    "/{dataset_id}",
    response_model=DatasetResponse,
    summary="Update dataset",
    description="Update an existing dataset"
)
async def update_dataset(
    dataset_id: UUID,
    dataset_update: DatasetUpdate,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user) if settings.is_feature_enabled("authentication") else Depends(get_current_user_optional)
):
    """
    Update a dataset.
    
    Updates the specified fields of a dataset.
    """
    logger.info(f"Updating dataset: {dataset_id}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        dataset = datasets_storage.get(str(dataset_id))
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        # Update fields
        update_data = dataset_update.model_dump(exclude_unset=True)
        if "data" in update_data:
            update_data["record_count"] = len(update_data["data"])
        
        dataset.update(update_data)
        dataset["updated_at"] = datetime.utcnow()
        
        if settings.is_feature_enabled("audit_logging"):
            await audit_log(
                action="dataset.update",
                resource_id=str(dataset_id),
                user=current_user.id if current_user else "anonymous",
                details={"fields": list(update_data.keys())}
            )
        
        return DatasetResponse(**dataset)
    else:
        # Tier 2+: Database implementation
        service = DatasetService(db)
        updated = await service.update_dataset(
            dataset_id,
            dataset_update,
            updated_by=current_user.id if current_user else "system"
        )
        return DatasetResponse(**updated.to_dict())


@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete dataset",
    description="Delete a dataset (soft delete in production)"
)
async def delete_dataset(
    dataset_id: UUID,
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user) if settings.is_feature_enabled("authentication") else Depends(get_current_user_optional)
):
    """
    Delete a dataset.
    
    Performs a soft delete in production environments.
    """
    logger.info(f"Deleting dataset: {dataset_id}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation (hard delete)
        if str(dataset_id) not in datasets_storage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        del datasets_storage[str(dataset_id)]
        
        if settings.is_feature_enabled("audit_logging"):
            await audit_log(
                action="dataset.delete",
                resource_id=str(dataset_id),
                user=current_user.id if current_user else "anonymous"
            )
        
        return None
    else:
        # Tier 2+: Database implementation (soft delete)
        service = DatasetService(db)
        await service.delete_dataset(
            dataset_id,
            deleted_by=current_user.id if current_user else "system"
        )
        return None


@router.get(
    "/{dataset_id}/entries",
    summary="Get dataset entries",
    description="Retrieve the test cases from a dataset"
)
async def get_dataset_entries(
    dataset_id: UUID,
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Limit results"),
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Get dataset entries.
    
    Returns the test cases from a dataset with optional pagination.
    """
    logger.info(f"Getting entries for dataset: {dataset_id}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        dataset = datasets_storage.get(str(dataset_id))
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        entries = dataset.get("data", [])
        
        # Apply pagination
        if limit:
            entries = entries[offset:offset + limit]
        else:
            entries = entries[offset:]
        
        return {
            "entries": entries,
            "total": len(dataset.get("data", [])),
            "offset": offset,
            "limit": limit
        }
    else:
        # Tier 2+: Database implementation
        service = DatasetService(db)
        entries = await service.get_dataset_entries(
            dataset_id,
            offset=offset,
            limit=limit
        )
        return entries


@router.get(
    "/{dataset_id}/preview/",
    summary="Preview dataset entries",
    description="Get a preview of dataset entries with optional limit"
)
async def preview_dataset(
    dataset_id: UUID,
    limit: int = Query(100, ge=1, le=1000, description="Maximum entries to return"),
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Get a preview of dataset entries.
    
    Returns the dataset info with a limited number of entries for preview.
    """
    logger.info(f"Previewing dataset: {dataset_id} with limit: {limit}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        dataset = datasets_storage.get(str(dataset_id))
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        entries = dataset.get("data", [])[:limit]
        
        return {
            "id": dataset.get("id"),
            "name": dataset.get("name"),
            "description": dataset.get("description"),
            "type": dataset.get("type"),
            "total_count": dataset.get("record_count", len(dataset.get("data", []))),
            "preview_count": len(entries),
            "preview": entries
        }
    else:
        # Tier 2+: Database implementation
        service = DatasetService(db)
        dataset = await service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        entries = (dataset.data or [])[:limit]
        
        return {
            "id": str(dataset.id),
            "name": dataset.name,
            "description": dataset.description,
            "type": dataset.type,
            "total_count": dataset.record_count,
            "preview_count": len(entries),
            "preview": entries
        }


@router.post(
    "/{dataset_id}/duplicate",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Duplicate dataset",
    description="Create a copy of an existing dataset"
)
async def duplicate_dataset(
    dataset_id: UUID,
    new_name: str = Body(..., description="Name for the duplicated dataset"),
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user) if settings.is_feature_enabled("authentication") else Depends(get_current_user_optional)
):
    """
    Duplicate a dataset.
    
    Creates a new dataset with the same data as the original.
    """
    logger.info(f"Duplicating dataset: {dataset_id} as {new_name}")
    
    if settings.deployment_tier == "development":
        # Tier 1: In-memory implementation
        original = datasets_storage.get(str(dataset_id))
        if not original:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        from uuid import uuid4
        
        new_id = str(uuid4())
        duplicated = original.copy()
        duplicated.update({
            "id": new_id,
            "name": new_name,
            "created_at": datetime.utcnow(),
            "updated_at": None,
            "created_by": current_user.id if current_user else "system"
        })
        
        datasets_storage[new_id] = duplicated
        
        if settings.is_feature_enabled("audit_logging"):
            await audit_log(
                action="dataset.duplicate",
                resource_id=new_id,
                user=current_user.id if current_user else "anonymous",
                details={"original_id": str(dataset_id), "new_name": new_name}
            )
        
        return DatasetResponse(**duplicated)
    else:
        # Tier 2+: Database implementation
        service = DatasetService(db)
        duplicated = await service.duplicate_dataset(
            dataset_id,
            new_name,
            created_by=current_user.id if current_user else "system"
        )
        return DatasetResponse(**duplicated.to_dict())