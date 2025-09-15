"""
Datasets endpoint for managing test datasets.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
import json
from enum import Enum

from fastapi import APIRouter, HTTPException, status, Depends, Query, Body, UploadFile, File, Form, BackgroundTasks
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
from app.utils.csv_parser import csv_parser
from app.schemas.dataset import (
    DatasetItemBase,
    DatasetBase,
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    DatasetListResponse,
    CSVUploadResponse,
    ConversationTurn,
    ConversationInput,
    SingleTurnInput
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


@router.post(
    "/validate-csv",
    summary="Validate CSV file",
    description="Validate a CSV file without creating a dataset"
)
async def validate_csv(
    file: UploadFile = File(..., description="CSV file to validate"),
    current_user = Depends(get_current_user_optional)
):
    """
    Validate a CSV file without creating a dataset.
    
    Useful for checking format and data before actual upload.
    """
    logger.info(f"Validating CSV file: {file.filename}")
    
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a CSV file"
        )
    
    try:
        import pandas as pd
        import io
        
        # Read the CSV file directly first for preview
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Get column names and basic info
        columns = df.columns.tolist()
        total_rows = len(df)
        
        # Create sample data for preview (first 5 rows)
        sample_data = []
        for idx in range(min(5, len(df))):
            row_dict = {}
            for col in columns:
                val = df.iloc[idx][col]
                row_dict[col] = str(val) if pd.notna(val) else ""
            sample_data.append(row_dict)
        
        # Detect format type
        format_type = "simple"
        if "conversation_id" in columns and "role" in columns:
            format_type = "conversation"
        elif "input" in columns and "expected_output" in columns:
            format_type = "structured"
        
        # For now, always mark as valid if we can read the CSV
        is_valid = total_rows > 0
        
        return {
            "valid": is_valid,
            "format_type": format_type,
            "total_rows": total_rows,
            "valid_items": total_rows,
            "warnings": [],
            "columns": columns,
            "sample": sample_data,  # Include sample of parsed items
            "preview_data": sample_data,  # For backward compatibility
            "metadata": {
                "format_type": format_type,
                "total_rows": total_rows,
                "detected_columns": columns,
                "processed_items": total_rows
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV validation failed: {e}")
        return {
            "valid": False,
            "error": str(e),
            "warnings": []
        }


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


# ==========================================
# CSV Upload Status Tracking
# ==========================================
class UploadStatus(str, Enum):
    """Upload status enumeration."""
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# In-memory tracking for upload status (Tier 1)
# In production (Tier 3+), this would use Redis or database
upload_status_storage: Dict[str, Dict[str, Any]] = {}


async def process_csv_in_background(
    upload_id: str,
    file_content: bytes,
    filename: str,
    dataset_params: Dict[str, Any],
    user_id: Optional[str] = None
):
    """
    Process CSV file in background for large files.
    """
    try:
        # Update status to processing
        upload_status_storage[upload_id] = {
            "status": UploadStatus.PROCESSING,
            "message": "Processing CSV file...",
            "progress": 0
        }
        
        # Create a fake UploadFile object for the parser
        import io
        from fastapi import UploadFile
        
        file_like = io.BytesIO(file_content)
        upload_file = UploadFile(
            filename=filename,
            file=file_like
        )
        
        # Parse CSV with production parser
        items, warnings, metadata = await csv_parser.parse_csv_file(upload_file)
        
        # Create dataset
        dataset_create = DatasetCreate(
            name=dataset_params['name'],
            description=dataset_params.get('description', f"Dataset uploaded from {filename}"),
            type=dataset_params.get('type', 'custom'),
            data=items,
            tags=dataset_params.get('tags', []),
            metadata={
                **metadata,
                "source": "csv_upload",
                "original_filename": filename,
                "upload_id": upload_id
            },
            created_by=user_id or "system"
        )
        
        # Store dataset (simplified for Tier 1)
        from uuid import uuid4
        dataset_id = str(uuid4())
        dataset_dict = dataset_create.model_dump()
        dataset_dict.update({
            "id": dataset_id,
            "status": "active",
            "record_count": len(items),
            "created_at": datetime.utcnow(),
            "updated_at": None,
            "version": "1.0.0"
        })
        
        datasets_storage[dataset_id] = dataset_dict
        
        # Update status to completed
        upload_status_storage[upload_id] = {
            "status": UploadStatus.COMPLETED,
            "message": f"Successfully processed {len(items)} items",
            "dataset_id": dataset_id,
            "warnings": warnings,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Background CSV processing failed: {e}")
        upload_status_storage[upload_id] = {
            "status": UploadStatus.FAILED,
            "message": f"Processing failed: {str(e)}",
            "error": str(e)
        }


@router.post(
    "/upload-csv",
    response_model=CSVUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload CSV dataset",
    description="Upload a CSV file to create a new dataset with production-ready streaming and validation"
)
async def upload_csv_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file to upload"),
    name: str = Form(..., description="Name for the dataset"),
    description: Optional[str] = Form(None, description="Description for the dataset"),
    type: str = Form("custom", description="Dataset type"),
    tags: Optional[str] = Form(None, description="Comma-separated tags for the dataset"),
    async_processing: bool = Form(False, description="Process large files in background"),
    db: Optional[AsyncSession] = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """
    Upload a CSV file to create a dataset with production-ready features.
    
    **Features:**
    - Streaming parsing for memory efficiency
    - Chunked processing (1000 rows at a time)
    - Automatic format detection
    - Background processing for large files (>5MB)
    - Comprehensive validation with early failure
    
    **Supported CSV Formats:**
    
    1. **Simple Format** (single-turn):
       ```csv
       input,expected_output,context,tags
       "What is 2+2?","4","math","arithmetic,basic"
       ```
    
    2. **Conversation Format** (multi-turn):
       ```csv
       role,content,scenario,expected_outcome,test_id
       user,"Hello","greeting","polite_response","test_1"
       assistant,"Hi there! How can I help?","greeting","","test_1"
       ```
    
    3. **Flexible Format** (auto-detected):
       - Supports alternative column names: prompt, question, text
       - Metadata columns: meta_* fields
       - JSON arrays in context field
    
    **File Limits:**
    - Maximum file size: 50MB
    - For files >5MB, use async_processing=true
    """
    logger.info(f"Uploading CSV dataset: {name} (async={async_processing})")
    
    # Validate file type
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a CSV file"
        )
    
    # Parse tags
    dataset_tags = []
    if tags:
        dataset_tags = [t.strip() for t in tags.split(',') if t.strip()]
    
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
        from uuid import uuid4
        upload_id = str(uuid4())
        
        # Store initial status
        upload_status_storage[upload_id] = {
            "status": UploadStatus.PROCESSING,
            "message": "Upload received, processing in background...",
            "filename": file.filename,
            "file_size": file_size
        }
        
        # Add background task
        background_tasks.add_task(
            process_csv_in_background,
            upload_id=upload_id,
            file_content=file_content,
            filename=file.filename,
            dataset_params={
                "name": name,
                "description": description,
                "type": type,
                "tags": dataset_tags
            },
            user_id=current_user.id if current_user else None
        )
        
        # Return immediate response with upload ID (not using strict schema validation)
        return {
            "message": f"Large file ({file_size // 1024}KB) queued for background processing",
            "dataset": {
                "id": upload_id,
                "name": name,
                "description": description or f"Processing {file.filename}...",
                "type": type,
                "status": "processing",
                "record_count": 0,
                "created_by": current_user.id if current_user else "system",
                "created_at": datetime.utcnow(),
                "updated_at": None,
                "version": "1.0.0",
                "data": [{"input": "Processing...", "metadata": {"status": "processing"}}],  # Placeholder to satisfy schema
                "tags": dataset_tags,
                "metadata": {"upload_id": upload_id}
            },
            "rows_processed": 0,
            "warnings": [f"Check status at /api/v1/datasets/upload-status/{upload_id}"]
        }
    
    # Process synchronously for smaller files
    try:
        # Parse CSV with production parser
        items, warnings, metadata = await csv_parser.parse_csv_file(file)
        
        if not items:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid data could be parsed from the CSV file",
                headers={"X-Warnings": json.dumps(warnings)} if warnings else None
            )
        
        # Create dataset
        dataset_create = DatasetCreate(
            name=name,
            description=description or f"Dataset uploaded from {file.filename}",
            type=type,
            data=items,
            tags=dataset_tags,
            metadata={
                **metadata,
                "source": "csv_upload",
                "original_filename": file.filename
            },
            created_by=current_user.id if current_user else "system"
        )
        
        if settings.deployment_tier == "development":
            # Tier 1: In-memory implementation
            from uuid import uuid4
            
            dataset_id = str(uuid4())
            dataset_dict = dataset_create.model_dump()
            dataset_dict.update({
                "id": dataset_id,
                "status": "active",
                "record_count": len(items),
                "created_at": datetime.utcnow(),
                "updated_at": None,
                "version": "1.0.0"
            })
            
            datasets_storage[dataset_id] = dataset_dict
            
            if settings.is_feature_enabled("audit_logging"):
                await audit_log(
                    action="dataset.upload_csv",
                    resource_id=dataset_id,
                    user=current_user.id if current_user else "anonymous",
                    details={
                        "name": name,
                        "filename": file.filename,
                        "rows_processed": len(items),
                        "format": metadata.get('format_type')
                    }
                )
            
            response_dataset = DatasetResponse(**dataset_dict)
        else:
            # Tier 2+: Database implementation
            service = DatasetService(db)
            created = await service.create_dataset(
                dataset_create,
                created_by=current_user.id if current_user else "system"
            )
            response_dataset = DatasetResponse(**created.to_dict())
        
        return CSVUploadResponse(
            message=f"Successfully uploaded dataset from {file.filename}",
            dataset=response_dataset,
            rows_processed=len(items),
            warnings=warnings
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process CSV file: {str(e)}"
        )


@router.get(
    "/upload-status/{upload_id}",
    summary="Check CSV upload status",
    description="Check the status of a background CSV upload"
)
async def check_upload_status(
    upload_id: str,
    current_user = Depends(get_current_user_optional)
):
    """
    Check the status of a background CSV upload.
    
    Returns the current status and any available results.
    """
    status = upload_status_storage.get(upload_id)
    
    if not status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload {upload_id} not found"
        )
    
    # If completed, include the dataset
    if status.get("status") == UploadStatus.COMPLETED and status.get("dataset_id"):
        dataset = datasets_storage.get(status["dataset_id"])
        if dataset:
            status["dataset"] = DatasetResponse(**dataset)
    
    return status


