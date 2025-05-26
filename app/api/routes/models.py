"""
Models routes
"""

import logging
from fastapi import APIRouter, HTTPException, Depends

from app.models.openai_models import ModelListResponse, ErrorResponse
from app.services.gauss_client import GaussChatClient
from app.services.converter import OpenAIGaussConverter
from app.api.dependencies import get_gauss_client, get_converter
from app.core.exceptions import BaseAPIException, GaussAPIError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/models",
    response_model=ModelListResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        502: {"model": ErrorResponse, "description": "Bad Gateway"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)
async def list_models(
    gauss_client: GaussChatClient = Depends(get_gauss_client),
    converter: OpenAIGaussConverter = Depends(get_converter)
):
    """
    List available models
    
    This endpoint is compatible with OpenAI's models API.
    Returns a list of available models that can be used with the chat completions endpoint.
    """
    try:
        logger.info("Fetching available models")
        
        # Try to get models from Gauss API
        try:
            gauss_models = gauss_client.get_models()
        except Exception as e:
            logger.warning(f"Failed to fetch models from Gauss API: {e}")
            # Fall back to default model list
            gauss_models = []
        
        # Convert to OpenAI format
        openai_models = converter.gauss_models_to_openai(gauss_models)
        
        logger.info(f"Successfully returned {len(openai_models.data)} models")
        return openai_models
    
    except BaseAPIException as e:
        logger.error(f"API error in list models: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.to_dict())
    
    except Exception as e:
        logger.error(f"Unexpected error in list models: {str(e)}")
        error = GaussAPIError(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=error.status_code, detail=error.to_dict())