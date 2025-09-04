"""
Evaluators endpoint for listing available evaluation metrics.
"""

from typing import List, Dict, Any
import httpx
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import logger
from app.core.dependencies import get_current_user_optional

router = APIRouter()


# ==========================================
# Schemas
# ==========================================
class EvaluatorInfo(BaseModel):
    """Information about an available evaluator."""
    id: str = Field(..., description="Unique evaluator identifier")
    name: str = Field(..., description="Human-readable evaluator name")
    description: str = Field(..., description="Evaluator description")
    category: str = Field(..., description="Evaluator category")
    type: str = Field(..., description="Evaluator type (llm_judge, rule_based, etc.)")
    config_schema: Dict[str, Any] = Field(default_factory=dict, description="Configuration schema")
    supported_metrics: List[str] = Field(default_factory=list, description="Supported metrics")
    version: str = Field(default="1.0.0", description="Evaluator version")


# ==========================================
# Static Evaluator Definitions
# ==========================================
AVAILABLE_EVALUATORS = [
    EvaluatorInfo(
        id="security_evaluator",
        name="Security Evaluator",
        description="Evaluates responses for security vulnerabilities and risks",
        category="security",
        type="rule_based",
        supported_metrics=["prompt_injection", "data_leakage", "encoded_content"],
        config_schema={
            "sensitivity_level": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"}
        }
    ),
    EvaluatorInfo(
        id="quality_evaluator",
        name="Quality Evaluator",
        description="Evaluates response quality, accuracy, and relevance",
        category="quality",
        type="llm_judge",
        supported_metrics=["accuracy", "relevance", "completeness", "coherence"],
        config_schema={
            "judge_model": {"type": "string", "default": "gpt-4"},
            "temperature": {"type": "number", "default": 0.3}
        }
    ),
    EvaluatorInfo(
        id="healthcare_compliance",
        name="Healthcare Compliance Evaluator",
        description="Evaluates compliance with healthcare regulations and standards",
        category="compliance",
        type="rule_based",
        supported_metrics=["hipaa_compliance", "phi_detection", "medical_accuracy"],
        config_schema={
            "strict_mode": {"type": "boolean", "default": True},
            "regulation": {"type": "string", "enum": ["hipaa", "gdpr", "both"], "default": "hipaa"}
        }
    ),
    EvaluatorInfo(
        id="performance_evaluator",
        name="Performance Evaluator",
        description="Evaluates response time and computational efficiency",
        category="performance",
        type="metric_based",
        supported_metrics=["response_time", "token_usage", "cost_efficiency"],
        config_schema={
            "time_threshold_ms": {"type": "number", "default": 1000},
            "token_limit": {"type": "number", "default": 4096}
        }
    ),
    EvaluatorInfo(
        id="sentiment_evaluator",
        name="Sentiment Evaluator",
        description="Evaluates emotional tone and sentiment of responses",
        category="behavioral",
        type="ml_based",
        supported_metrics=["sentiment_score", "emotion_detection", "toxicity"],
        config_schema={
            "model": {"type": "string", "default": "distilbert-base-uncased"},
            "threshold": {"type": "number", "default": 0.5}
        }
    ),
    EvaluatorInfo(
        id="factuality_evaluator",
        name="Factuality Evaluator",
        description="Evaluates factual accuracy and hallucination detection",
        category="quality",
        type="llm_judge",
        supported_metrics=["factual_accuracy", "hallucination_detection", "citation_quality"],
        config_schema={
            "fact_checking_model": {"type": "string", "default": "gpt-4-turbo"},
            "require_citations": {"type": "boolean", "default": False}
        }
    ),
    EvaluatorInfo(
        id="bias_evaluator",
        name="Bias & Fairness Evaluator",
        description="Evaluates responses for bias and fairness issues",
        category="ethics",
        type="ml_based",
        supported_metrics=["gender_bias", "racial_bias", "fairness_score"],
        config_schema={
            "bias_categories": {"type": "array", "default": ["gender", "race", "age"]},
            "sensitivity": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"}
        }
    ),
    EvaluatorInfo(
        id="custom_llm_judge",
        name="Custom LLM Judge",
        description="Customizable LLM-based evaluation with user-defined criteria",
        category="custom",
        type="llm_judge",
        supported_metrics=["custom_score"],
        config_schema={
            "judge_model": {"type": "string", "default": "gpt-4"},
            "evaluation_prompt": {"type": "string", "required": True},
            "scoring_criteria": {"type": "object", "required": True}
        }
    )
]


# ==========================================
# API Endpoints
# ==========================================
class EvaluatorsResponse(BaseModel):
    """Response model for evaluators list."""
    evaluators: List[EvaluatorInfo]
    total: int


@router.get(
    "/",
    response_model=EvaluatorsResponse,
    summary="List available evaluators",
    description="Get a list of all available evaluation metrics and evaluators"
)
async def list_evaluators(
    category: str = None,
    type: str = None,
    current_user = Depends(get_current_user_optional)
):
    """
    List all available evaluators.
    
    Returns information about available evaluation metrics that can be used
    in experiments and evaluations.
    """
    logger.info(f"Listing evaluators - category: {category}, type: {type}")
    
    # Try to fetch from evaluator service if available
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.evaluator_service_url}/evaluators",
                timeout=5.0
            )
            if response.status_code == 200:
                evaluator_data = response.json()
                # Handle different response formats
                if isinstance(evaluator_data, dict) and "evaluators" in evaluator_data:
                    evaluator_list = evaluator_data["evaluators"]
                elif isinstance(evaluator_data, list):
                    evaluator_list = evaluator_data
                else:
                    raise ValueError("Unexpected response format from evaluator service")
                
                # Convert to EvaluatorInfo objects with field mapping
                service_evaluators = []
                for e in evaluator_list:
                    # Map fields from evaluator service to our schema
                    evaluator_info = {
                        "id": e.get("id"),
                        "name": e.get("name"),
                        "description": e.get("description"),
                        "category": e.get("category", "custom"),
                        "type": e.get("type", "custom"),  # Default to custom if not provided
                        "config_schema": e.get("config_schema", {}),
                        "supported_metrics": e.get("supported_metrics", []),
                        "version": e.get("version", "1.0.0")
                    }
                    try:
                        service_evaluators.append(EvaluatorInfo(**evaluator_info))
                    except Exception as ex:
                        logger.debug(f"Skipping evaluator {e.get('id')}: {ex}")
                
                # Combine with static evaluators (avoiding duplicates by ID)
                service_ids = {e.id for e in service_evaluators}
                static_evaluators = [e for e in AVAILABLE_EVALUATORS if e.id not in service_ids]
                evaluators = service_evaluators + static_evaluators
                
                # Apply filters
                if category:
                    evaluators = [e for e in evaluators if e.category == category]
                
                if type:
                    evaluators = [e for e in evaluators if e.type == type]
                
                return EvaluatorsResponse(evaluators=evaluators, total=len(evaluators))
    except Exception as e:
        logger.warning(f"Failed to fetch evaluators from service: {e}. Using static list.")
    
    # Fallback to static list if service is unavailable
    evaluators = AVAILABLE_EVALUATORS.copy()
    
    # Apply filters
    if category:
        evaluators = [e for e in evaluators if e.category == category]
    
    if type:
        evaluators = [e for e in evaluators if e.type == type]
    
    return EvaluatorsResponse(evaluators=evaluators, total=len(evaluators))


@router.get(
    "/{evaluator_id}",
    response_model=EvaluatorInfo,
    summary="Get evaluator details",
    description="Get detailed information about a specific evaluator"
)
async def get_evaluator(
    evaluator_id: str,
    current_user = Depends(get_current_user_optional)
):
    """
    Get details about a specific evaluator.
    
    Returns comprehensive information about the evaluator including
    its configuration schema and supported metrics.
    """
    logger.info(f"Getting evaluator: {evaluator_id}")
    
    # Try to fetch from evaluator service if available
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.evaluator_service_url}/evaluators/{evaluator_id}",
                timeout=5.0
            )
            if response.status_code == 200:
                return EvaluatorInfo(**response.json())
            elif response.status_code == 404:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Evaluator {evaluator_id} not found"
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Failed to fetch evaluator from service: {e}. Using static list.")
    
    # Fallback to static list if service is unavailable
    evaluator = next((e for e in AVAILABLE_EVALUATORS if e.id == evaluator_id), None)
    
    if not evaluator:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluator {evaluator_id} not found"
        )
    
    return evaluator


@router.get(
    "/categories",
    response_model=List[str],
    summary="List evaluator categories",
    description="Get a list of all available evaluator categories"
)
async def list_evaluator_categories(
    current_user = Depends(get_current_user_optional)
):
    """
    List all available evaluator categories.
    """
    categories = list(set(e.category for e in AVAILABLE_EVALUATORS))
    return sorted(categories)


@router.get(
    "/types",
    response_model=List[str],
    summary="List evaluator types",
    description="Get a list of all available evaluator types"
)
async def list_evaluator_types(
    current_user = Depends(get_current_user_optional)
):
    """
    List all available evaluator types.
    """
    types = list(set(e.type for e in AVAILABLE_EVALUATORS))
    return sorted(types)