"""
Experiment schemas for request/response validation.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class ExperimentStatus(str, Enum):
    """Experiment status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionMode(str, Enum):
    """Experiment execution mode."""
    BATCH = "batch"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class AgentConfig(BaseModel):
    """Configuration for AI agent/model."""
    model: str = Field(..., description="Model identifier (e.g., gpt-4, claude-3)")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1)
    system_prompt: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ExperimentBase(BaseModel):
    """Base experiment schema."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    dataset_id: Optional[UUID] = Field(None, description="Associated dataset ID")
    agent_config: AgentConfig
    execution_mode: ExecutionMode = Field(default=ExecutionMode.BATCH)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentCreate(ExperimentBase):
    """Schema for creating experiments."""
    created_by: Optional[str] = Field(default="system", max_length=100)
    evaluator_ids: Optional[List[str]] = Field(default_factory=list, description="Evaluators to run")


class ExperimentUpdate(BaseModel):
    """Schema for updating experiments."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    agent_config: Optional[AgentConfig] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ExperimentResponse(ExperimentBase):
    """Response schema for experiments."""
    id: UUID
    status: ExperimentStatus
    progress: float = Field(default=0.0, ge=0, le=100)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: str
    created_at: datetime
    updated_at: Optional[datetime]
    dataset_name: Optional[str] = None
    dataset_size: Optional[int] = None
    
    model_config = ConfigDict(from_attributes=True)


class ExperimentListResponse(BaseModel):
    """Response schema for experiment list."""
    experiments: List[ExperimentResponse]
    total: int
    page: int
    page_size: int


class ExperimentResultItem(BaseModel):
    """Individual test result within an experiment."""
    test_id: str
    input: Any
    expected_output: Optional[Any]
    actual_output: Optional[Any]
    status: str
    execution_time: Optional[float]
    error: Optional[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentResults(BaseModel):
    """Experiment execution results."""
    experiment_id: UUID
    status: ExperimentStatus
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    execution_time: float
    results: List[ExperimentResultItem]
    summary: Dict[str, Any]


class ExperimentExecuteRequest(BaseModel):
    """Request to execute an experiment."""
    batch_size: Optional[int] = Field(None, ge=1, le=100, description="Batch size for processing")
    timeout: Optional[int] = Field(None, ge=1, description="Timeout in seconds")
    evaluator_ids: Optional[List[str]] = Field(default_factory=list, description="Evaluators to run")


class HTTPEndpointConfig(BaseModel):
    """Configuration for HTTP endpoint automation."""
    url: str = Field(..., description="HTTP endpoint URL to call")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers including auth tokens")
    timeout: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, ge=1, le=10, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.1, le=60, description="Initial retry delay in seconds")
    batch_size: int = Field(default=10, ge=1, le=50, description="Number of concurrent requests")


class AutomatedExecutionRequest(BaseModel):
    """Request to run automated experiment execution."""
    endpoint_config: HTTPEndpointConfig
    run_evaluations: bool = Field(default=False, description="Run evaluations after execution")
    evaluator_ids: Optional[List[str]] = Field(default_factory=list, description="Evaluators to run if enabled")