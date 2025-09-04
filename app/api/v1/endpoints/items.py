"""
Items endpoint demonstrating CRUD operations with tier-based features.
"""

from typing import List, Optional, Dict, Any
from uuid import uuid4
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, Query
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import logger
from app.core.dependencies import (
    get_current_user_optional,
    get_current_user,
    apply_rate_limit,
    audit_log
)

router = APIRouter()


# ==========================================
# Schemas
# ==========================================
class ItemBase(BaseModel):
    """Base item schema."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    price: float = Field(..., gt=0)
    tax: Optional[float] = Field(None, ge=0)
    tags: List[str] = Field(default_factory=list)


class ItemCreate(ItemBase):
    """Schema for creating items."""
    pass


class ItemUpdate(BaseModel):
    """Schema for updating items."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    price: Optional[float] = Field(None, gt=0)
    tax: Optional[float] = Field(None, ge=0)
    tags: Optional[List[str]] = None


class ItemResponse(ItemBase):
    """Schema for item responses."""
    id: str
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    
    class Config:
        from_attributes = True


# ==========================================
# In-Memory Storage (Tier 1)
# ==========================================
# In production, this would be replaced with database operations
items_db: Dict[str, Dict[str, Any]] = {}


# ==========================================
# Endpoints
# ==========================================
@router.get(
    "/",
    response_model=List[ItemResponse],
    summary="List all items"
)
async def list_items(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None, description="Search in name and description"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    current_user=Depends(get_current_user_optional)
) -> List[ItemResponse]:
    """
    Retrieve a list of items with optional filtering.
    
    - **skip**: Number of items to skip
    - **limit**: Maximum number of items to return
    - **search**: Search term for name and description
    - **tags**: Filter by tags (items must have all specified tags)
    """
    # Apply rate limiting if enabled
    if settings.is_feature_enabled("rate_limiting"):
        await apply_rate_limit()
    
    # Filter items
    filtered_items = []
    for item_id, item in items_db.items():
        # Apply search filter
        if search:
            search_lower = search.lower()
            if (search_lower not in item["name"].lower() and 
                search_lower not in (item.get("description") or "").lower()):
                continue
        
        # Apply tag filter
        if tags:
            item_tags = item.get("tags", [])
            if not all(tag in item_tags for tag in tags):
                continue
        
        filtered_items.append({**item, "id": item_id})
    
    # Apply pagination
    paginated_items = filtered_items[skip:skip + limit]
    
    # Log if audit logging is enabled
    if settings.is_feature_enabled("audit_logging"):
        await audit_log(
            action="LIST_ITEMS",
            user=current_user["username"] if current_user else "anonymous",
            details={"count": len(paginated_items), "skip": skip, "limit": limit}
        )
    
    return paginated_items


@router.get(
    "/{item_id}",
    response_model=ItemResponse,
    summary="Get item by ID"
)
async def get_item(
    item_id: str,
    current_user=Depends(get_current_user_optional)
) -> ItemResponse:
    """
    Retrieve a specific item by ID.
    """
    if item_id not in items_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with id '{item_id}' not found"
        )
    
    item = {**items_db[item_id], "id": item_id}
    
    # Log if audit logging is enabled
    if settings.is_feature_enabled("audit_logging"):
        await audit_log(
            action="GET_ITEM",
            user=current_user["username"] if current_user else "anonymous",
            resource_id=item_id
        )
    
    return item


@router.post(
    "/",
    response_model=ItemResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new item"
)
async def create_item(
    item: ItemCreate,
    current_user=Depends(get_current_user if settings.is_feature_enabled("authentication") else get_current_user_optional)
) -> ItemResponse:
    """
    Create a new item.
    
    Requires authentication if enabled in configuration.
    """
    # Generate unique ID
    item_id = str(uuid4())
    
    # Create item with metadata
    now = datetime.utcnow()
    new_item = {
        **item.model_dump(),
        "created_at": now,
        "updated_at": now,
        "created_by": current_user["username"] if current_user else None
    }
    
    # Store item
    items_db[item_id] = new_item
    
    # Log creation
    logger.info(
        "Item created",
        item_id=item_id,
        name=item.name,
        user=current_user["username"] if current_user else "anonymous"
    )
    
    # Audit log if enabled
    if settings.is_feature_enabled("audit_logging"):
        await audit_log(
            action="CREATE_ITEM",
            user=current_user["username"] if current_user else "anonymous",
            resource_id=item_id,
            details={"name": item.name, "price": item.price}
        )
    
    return {**new_item, "id": item_id}


@router.put(
    "/{item_id}",
    response_model=ItemResponse,
    summary="Update an item"
)
async def update_item(
    item_id: str,
    item_update: ItemUpdate,
    current_user=Depends(get_current_user if settings.is_feature_enabled("authentication") else get_current_user_optional)
) -> ItemResponse:
    """
    Update an existing item.
    
    Requires authentication if enabled in configuration.
    """
    if item_id not in items_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with id '{item_id}' not found"
        )
    
    # Get existing item
    existing_item = items_db[item_id]
    
    # Update only provided fields
    update_data = item_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        existing_item[field] = value
    
    # Update metadata
    existing_item["updated_at"] = datetime.utcnow()
    
    # Log update
    logger.info(
        "Item updated",
        item_id=item_id,
        updates=update_data,
        user=current_user["username"] if current_user else "anonymous"
    )
    
    # Audit log if enabled
    if settings.is_feature_enabled("audit_logging"):
        await audit_log(
            action="UPDATE_ITEM",
            user=current_user["username"] if current_user else "anonymous",
            resource_id=item_id,
            details=update_data
        )
    
    return {**existing_item, "id": item_id}


@router.delete(
    "/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an item"
)
async def delete_item(
    item_id: str,
    current_user=Depends(get_current_user if settings.is_feature_enabled("authentication") else get_current_user_optional)
) -> None:
    """
    Delete an item.
    
    Requires authentication if enabled in configuration.
    """
    if item_id not in items_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with id '{item_id}' not found"
        )
    
    # Delete item
    deleted_item = items_db.pop(item_id)
    
    # Log deletion
    logger.info(
        "Item deleted",
        item_id=item_id,
        name=deleted_item["name"],
        user=current_user["username"] if current_user else "anonymous"
    )
    
    # Audit log if enabled
    if settings.is_feature_enabled("audit_logging"):
        await audit_log(
            action="DELETE_ITEM",
            user=current_user["username"] if current_user else "anonymous",
            resource_id=item_id,
            details={"name": deleted_item["name"]}
        )
