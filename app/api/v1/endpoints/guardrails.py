"""
Simple guardrails wrapper endpoints.
Each endpoint takes a string input and returns a processed string output.
These endpoints wrap the actual guardrail services running separately.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.core.logging import logger
import httpx
import time
import os

router = APIRouter()

# Service URLs - use localhost for local development, Docker service names for Docker
PIIGUARD_URL = os.getenv("PIIGUARD_SERVICE_URL", "http://localhost:8080")
SECRETS_DETECTOR_URL = os.getenv("SECRETS_DETECTOR_SERVICE_URL", "http://secrets-detector:8081")
PROFANITY_GUARD_URL = os.getenv("PROFANITY_GUARD_SERVICE_URL", "http://profanity-guardrail:8082")
JAILBREAK_GUARD_URL = os.getenv("JAILBREAK_GUARD_SERVICE_URL", "http://jailbreak-guard:8086")


class GuardrailRequest(BaseModel):
    """Simple request model - just a text string."""
    text: str = Field(..., description="The text to process")


class GuardrailResponse(BaseModel):
    """Simple response model - processed text string."""
    text: str = Field(..., description="The processed text")


@router.post(
    "/pii-redaction",
    response_model=GuardrailResponse,
    summary="PII/PHI Redaction",
    description="Redact Protected Health Information (PHI) and PII from text using PiiGuard service"
)
async def process_pii_redaction(request: GuardrailRequest):
    """
    Redact PII/PHI entities from text using PiiGuard service.
    Calls the PiiGuard /check endpoint which detects and redacts both PII and PHI.
    """
    logger.info(f"Processing PII/PHI redaction request via PiiGuard service at {PIIGUARD_URL}")
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Call PiiGuard service /check endpoint
            response = await client.post(
                f"{PIIGUARD_URL}/check",
                json={
                    "correlation_id": "pii-redaction",
                    "content": request.text,
                    "content_type": "text/plain"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                # PiiGuard returns modified_content in the response
                processed_text = result.get("modified_content", request.text)
            else:
                logger.error(f"PiiGuard service returned status {response.status_code}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"PiiGuard service returned error: {response.status_code}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Error calling PiiGuard service: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PiiGuard service is unavailable: {str(e)}"
        )
    except httpx.TimeoutException:
        logger.error("Timeout calling PiiGuard service")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="PiiGuard service request timed out"
        )
    
    processing_time = (time.time() - start_time) * 1000
    logger.info(f"PII/PHI redaction completed in {processing_time:.2f}ms")
    
    return GuardrailResponse(text=processed_text)


@router.post(
    "/secrets-detection",
    response_model=GuardrailResponse,
    summary="Secrets Detection",
    description="Detect and redact API keys, passwords, tokens, and other secrets"
)
async def process_secrets_detection(request: GuardrailRequest):
    """
    Detect and redact secrets using the secrets-detector service.
    Handles API keys, passwords, tokens, and other sensitive credentials.
    """
    logger.info("Processing secrets detection request via secrets-detector service")
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Call secrets-detector service /check endpoint
            response = await client.post(
                f"{SECRETS_DETECTOR_URL}/check",
                json={
                    "correlation_id": "secrets-check",
                    "content": request.text,
                    "content_type": "text/plain"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                # If modified_content is None, use the original text
                processed_text = result.get("modified_content")
                if processed_text is None:
                    processed_text = request.text
            else:
                logger.error(f"Secrets-detector service returned status {response.status_code}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Secrets-detector service returned error: {response.status_code}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Error calling secrets-detector service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Secrets-detector service is unavailable: {str(e)}"
        )
    except httpx.TimeoutException:
        logger.error("Timeout calling secrets-detector service")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Secrets-detector service request timed out"
        )
    
    processing_time = (time.time() - start_time) * 1000
    logger.info(f"Secrets detection completed in {processing_time:.2f}ms")
    
    return GuardrailResponse(text=processed_text)


@router.post(
    "/content-moderation",
    response_model=GuardrailResponse,
    summary="Content Moderation",
    description="Moderate content for inappropriate or harmful material using profanity-guardrail service"
)
async def process_content_moderation(request: GuardrailRequest):
    """
    Moderate content for inappropriate material using profanity-guardrail service.
    """
    logger.info("Processing content moderation request via profanity-guardrail service")
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Call profanity-guardrail service /check endpoint
            response = await client.post(
                f"{PROFANITY_GUARD_URL}/check",
                json={
                    "correlation_id": "content-moderation",
                    "content": request.text,
                    "content_type": "text/plain"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                # If modified_content is None, use the original text
                processed_text = result.get("modified_content")
                if processed_text is None:
                    processed_text = request.text
            else:
                logger.error(f"Profanity-guardrail service returned status {response.status_code}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Profanity-guardrail service returned error: {response.status_code}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Error calling profanity-guardrail service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Profanity-guardrail service is unavailable: {str(e)}"
        )
    except httpx.TimeoutException:
        logger.error("Timeout calling profanity-guardrail service")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Profanity-guardrail service request timed out"
        )
    
    processing_time = (time.time() - start_time) * 1000
    logger.info(f"Content moderation completed in {processing_time:.2f}ms")
    
    return GuardrailResponse(text=processed_text)


@router.post(
    "/prompt-injection",
    response_model=GuardrailResponse,
    summary="Prompt Injection Detection",
    description="Detect and prevent prompt injection attacks using jailbreak-guard service"
)
async def process_prompt_injection(request: GuardrailRequest):
    """
    Detect potential prompt injection attempts using jailbreak-guard service.
    """
    logger.info("Processing prompt injection detection request via jailbreak-guard service")
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Call jailbreak-guard service /check endpoint
            response = await client.post(
                f"{JAILBREAK_GUARD_URL}/check",
                json={
                    "correlation_id": "prompt-injection",
                    "content": request.text,
                    "content_type": "text/plain"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                # If modified_content is None, use the original text
                processed_text = result.get("modified_content")
                if processed_text is None:
                    processed_text = request.text
            else:
                logger.error(f"Jailbreak-guard service returned status {response.status_code}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Jailbreak-guard service returned error: {response.status_code}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Error calling jailbreak-guard service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Jailbreak-guard service is unavailable: {str(e)}"
        )
    except httpx.TimeoutException:
        logger.error("Timeout calling jailbreak-guard service")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Jailbreak-guard service request timed out"
        )
    
    processing_time = (time.time() - start_time) * 1000
    logger.info(f"Prompt injection detection completed in {processing_time:.2f}ms")
    
    return GuardrailResponse(text=processed_text)



@router.get("/test-piiguard")
async def test_piiguard():
    """Test PiiGuard connection directly."""
    import requests
    try:
        response = requests.get("http://127.0.0.1:8080/health", timeout=5.0)
        return {"status": "success", "piiguard_status": response.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e), "type": type(e).__name__}


@router.get(
    "/health",
    summary="Guardrails Health Check",
    description="Check if guardrails service is running and which services are available"
)
async def health_check():
    """Health check endpoint for guardrails service."""
    # Check which services are available
    available_services = []
    service_status = {}
    
    # Check PiiGuard
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{PIIGUARD_URL}/health")
            if response.status_code == 200:
                available_services.append("piiguard")
                service_status["piiguard"] = "healthy"
            else:
                service_status["piiguard"] = f"unhealthy (status: {response.status_code})"
    except Exception as e:
        service_status["piiguard"] = f"unavailable ({type(e).__name__})"
    
    # Check Secrets Detector
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{SECRETS_DETECTOR_URL}/health")
            if response.status_code == 200:
                available_services.append("secrets_detector")
                service_status["secrets_detector"] = "healthy"
            else:
                service_status["secrets_detector"] = f"unhealthy (status: {response.status_code})"
    except Exception as e:
        service_status["secrets_detector"] = f"unavailable ({type(e).__name__})"
    
    # Check Profanity Guardrail
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{PROFANITY_GUARD_URL}/health")
            if response.status_code == 200:
                available_services.append("profanity_guardrail")
                service_status["profanity_guardrail"] = "healthy"
            else:
                service_status["profanity_guardrail"] = f"unhealthy (status: {response.status_code})"
    except Exception as e:
        service_status["profanity_guardrail"] = f"unavailable ({type(e).__name__})"
    
    # Check Jailbreak Guard
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{JAILBREAK_GUARD_URL}/health")
            if response.status_code == 200:
                available_services.append("jailbreak_guard")
                service_status["jailbreak_guard"] = "healthy"
            else:
                service_status["jailbreak_guard"] = f"unhealthy (status: {response.status_code})"
    except Exception as e:
        service_status["jailbreak_guard"] = f"unavailable ({type(e).__name__})"
    
    return {
        "status": "healthy",
        "service": "guardrails",
        "available_endpoints": [
            "pii_redaction",
            "secrets_detection",
            "content_moderation",
            "prompt_injection"
        ],
        "connected_services": available_services,
        "service_status": service_status
    }