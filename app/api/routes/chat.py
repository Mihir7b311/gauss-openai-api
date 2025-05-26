"""
Chat completion routes
"""

import json
import uuid
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from app.models.openai_models import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ErrorResponse
)
from app.services.gauss_client import GaussChatClient
from app.services.converter import OpenAIGaussConverter
from app.api.dependencies import get_gauss_client, get_converter
from app.core.exceptions import (
    BaseAPIException, 
    ValidationError, 
    GaussAPIError,
    ModelNotFoundError
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Model Not Found"},
        429: {"model": ErrorResponse, "description": "Rate Limit Exceeded"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        502: {"model": ErrorResponse, "description": "Bad Gateway"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    gauss_client: GaussChatClient = Depends(get_gauss_client),
    converter: OpenAIGaussConverter = Depends(get_converter)
):
    """
    Create a chat completion, optionally streamed
    
    This endpoint is compatible with OpenAI's chat completions API.
    """
    try:
        # Validate request
        validation_errors = converter.validate_openai_request(request)
        if validation_errors:
            raise ValidationError(f"Validation errors: {', '.join(validation_errors)}")
        
        # Log request (without sensitive data)
        logger.info(f"Chat completion request: model={request.model}, "
                   f"messages_count={len(request.messages)}, stream={request.stream}")
        
        # Convert OpenAI format to Gauss format
        contents, system_prompt = converter.openai_messages_to_gauss_contents(request.messages)
        llm_config = converter.openai_params_to_gauss_config(request)
        
        # Generate completion ID
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        
        if request.stream:
            # Streaming response
            logger.info(f"Starting streaming chat completion: {completion_id}")
            return StreamingResponse(
                generate_stream_response(
                    contents=contents,
                    llm_config=llm_config,
                    system_prompt=system_prompt,
                    model=request.model,
                    completion_id=completion_id,
                    gauss_client=gauss_client,
                    converter=converter
                ),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/plain; charset=utf-8"
                }
            )
        else:
            # Non-streaming response
            logger.info(f"Starting non-streaming chat completion: {completion_id}")
            gauss_response = gauss_client.chat_completion(
                contents=contents,
                llm_config=llm_config,
                system_prompt=system_prompt,
                stream=False
            )
            
            # Convert Gauss response to OpenAI format
            openai_response = converter.gauss_response_to_openai(
                gauss_response=gauss_response,
                model=request.model,
                completion_id=completion_id
            )
            
            logger.info(f"Chat completion successful: {completion_id}")
            return openai_response
    
    except BaseAPIException as e:
        logger.error(f"API error in chat completion: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.to_dict())
    
    except Exception as e:
        logger.error(f"Unexpected error in chat completion: {str(e)}")
        error = GaussAPIError(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=error.status_code, detail=error.to_dict())


async def generate_stream_response(
    contents: list,
    llm_config: Any,
    system_prompt: str,
    model: str,
    completion_id: str,
    gauss_client: GaussChatClient,
    converter: OpenAIGaussConverter
):
    """Generate streaming response in OpenAI format"""
    try:
        logger.info(f"Starting stream generation: {completion_id}")
        
        # Send first chunk with role
        first_chunk = converter.create_openai_stream_chunk(
            content="",
            model=model,
            completion_id=completion_id,
            is_first=True
        )
        yield f"data: {first_chunk.json()}\n\n"
        
        # Get streaming response from Gauss
        stream = gauss_client.chat_completion_stream(
            contents=contents,
            llm_config=llm_config,
            system_prompt=system_prompt
        )
        
        # Process each line from the stream
        for line in stream:
            try:
                parsed = converter.parse_gauss_stream_line(line)
                
                if parsed is None:
                    continue
                
                if parsed.get('type') == 'done':
                    # End of stream
                    break
                elif parsed.get('type') == 'content':
                    # Direct content
                    content = parsed.get('content', '')
                    if content:
                        chunk = converter.create_openai_stream_chunk(
                            content=content,
                            model=model,
                            completion_id=completion_id
                        )
                        yield f"data: {chunk.json()}\n\n"
                elif parsed.get('type') == 'data':
                    # Structured data
                    data = parsed.get('data', {})
                    content = data.get('content', '')
                    
                    if content:
                        chunk = converter.create_openai_stream_chunk(
                            content=content,
                            model=model,
                            completion_id=completion_id
                        )
                        yield f"data: {chunk.json()}\n\n"
                
            except Exception as e:
                logger.warning(f"Error processing stream line: {e}")
                continue
        
        # Send final chunk
        final_chunk = converter.create_openai_stream_chunk(
            content="",
            model=model,
            completion_id=completion_id,
            is_last=True,
            finish_reason="stop"
        )
        yield f"data: {final_chunk.json()}\n\n"
        yield "data: [DONE]\n\n"
        
        logger.info(f"Stream generation completed: {completion_id}")
        
    except Exception as e:
        logger.error(f"Error in stream generation: {e}")
        # Send error chunk
        error_chunk = converter.create_openai_stream_chunk(
            content="",
            model=model,
            completion_id=completion_id,
            is_last=True,
            finish_reason="error"
        )
        yield f"data: {error_chunk.json()}\n\n"
        yield "data: [DONE]\n\n"