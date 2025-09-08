"""
Dataset schemas for request/response validation.
"""

from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, field_validator


class ConversationTurn(BaseModel):
    """Schema for a single turn in a conversation."""
    role: Literal["user", "assistant", "system"] = Field(..., description="Role of the speaker")
    content: str = Field(..., description="Content of the message")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata for the turn")


class ConversationInput(BaseModel):
    """Schema for multi-turn conversation input."""
    scenario: str = Field(..., description="Description of the conversation scenario")
    turns: List[ConversationTurn] = Field(..., min_length=1, description="List of conversation turns")


class SingleTurnInput(BaseModel):
    """Schema for single-turn input."""
    question: Optional[str] = Field(None, description="Question for the test case")
    prompt: Optional[str] = Field(None, description="Prompt for the test case")
    text: Optional[str] = Field(None, description="Text input for the test case")
    
    class Config:
        extra = "allow"  # Allow additional fields


class DatasetItemBase(BaseModel):
    """Base schema for dataset items (test cases)."""
    # Core input field - flexible to support various formats
    input: Union[str, SingleTurnInput, ConversationInput, Dict[str, Any]] = Field(
        ..., 
        description="Test case input (string, object, or conversation)"
    )
    
    # Expected outputs
    expected_output: Optional[Any] = Field(None, description="Expected output for single-turn")
    expected_outcome: Optional[str] = Field(None, description="Expected outcome for conversations")
    
    # Optional context
    context: Optional[List[str]] = Field(default_factory=list, description="Background context")
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    test_id: Optional[str] = Field(None, description="Unique identifier for the test case")
    tags: Optional[List[str]] = Field(default_factory=list)


class DatasetBase(BaseModel):
    """Base dataset schema."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    type: str = Field(default="custom", description="Dataset type")
    data: List[DatasetItemBase] = Field(..., min_items=1, description="Dataset entries")
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetCreate(DatasetBase):
    """Schema for creating datasets."""
    created_by: Optional[str] = Field(default="system", max_length=100)


class DatasetUpdate(BaseModel):
    """Schema for updating datasets."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    data: Optional[List[DatasetItemBase]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetResponse(DatasetBase):
    """Response schema for datasets."""
    id: UUID
    status: str
    record_count: int
    created_by: str
    created_at: datetime
    updated_at: Optional[datetime]
    version: str = "1.0.0"
    
    model_config = ConfigDict(from_attributes=True)


class DatasetListResponse(BaseModel):
    """Response schema for dataset list."""
    datasets: List[DatasetResponse]
    total: int
    page: int
    page_size: int


class CSVUploadResponse(BaseModel):
    """Response schema for CSV upload."""
    message: str
    dataset: DatasetResponse
    rows_processed: int
    warnings: Optional[List[str]] = Field(default_factory=list)