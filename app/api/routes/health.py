"""
Health check routes
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException

from app.services.gauss_client import GaussChatClient
from app.api.dependencies import get_gauss_client
from app.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def root():
    """
    Root endpoint with API information
    """
    settings = get_settings()
    
    return {
        "message": "Gauss OpenAI Compatible API",
        "version": settings.api_version,
        "status": "running",
        "endpoints": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
            "health": "/health",
            "health_detailed": "/health/detailed"
        },
        "documentation": "/docs"
    }


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint
    """
    return {
        "status": "healthy",
        "service": "gauss-openai-api",
        "timestamp": None  # Will be set by middleware if needed
    }


@router.get("/health/detailed")
async def detailed_health_check(
    gauss_client: GaussChatClient = Depends(get_gauss_client)
) -> Dict[str, Any]:
    """
    Detailed health check with dependency verification
    """
    health_status = {
        "status": "healthy",
        "service": "gauss-openai-api",
        "checks": {}
    }
    
    # Check Gauss API connectivity
    try:
        is_gauss_healthy = gauss_client.health_check()
        health_status["checks"]["gauss_api"] = {
            "status": "healthy" if is_gauss_healthy else "unhealthy",
            "message": "Gauss API is accessible" if is_gauss_healthy else "Gauss API is not accessible"
        }
        
        if not is_gauss_healthy:
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["checks"]["gauss_api"] = {
            "status": "unhealthy",
            "message": f"Gauss API check failed: {str(e)}"
        }
        health_status["status"] = "unhealthy"
    
    # Check configuration
    try:
        settings = get_settings()
        config_check = {
            "status": "healthy",
            "message": "Configuration loaded successfully"
        }
        
        # Verify critical settings
        if not settings.gauss_pass_key or not settings.gauss_client_key:
            config_check["status"] = "unhealthy"
            config_check["message"] = "Missing required Gauss API credentials"
            health_status["status"] = "unhealthy"
            
        health_status["checks"]["configuration"] = config_check
        
    except Exception as e:
        health_status["checks"]["configuration"] = {
            "status": "unhealthy",
            "message": f"Configuration check failed: {str(e)}"
        }
        health_status["status"] = "unhealthy"
    
    # Return appropriate HTTP status
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    elif health_status["status"] == "degraded":
        raise HTTPException(status_code=200, detail=health_status)  # Still return 200 for degraded
    
    return health_status