"""
Conversion service between OpenAI and Gauss formats
"""

import json
import uuid
import time
from typing import List, Dict, Any, Tuple, Optional

from app.models.openai_models import (
    ChatMessage, 
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionChoice, 
    ChatCompletionUsage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ModelInfo,
    ModelListResponse
)
from app.models.gauss_models import (
    GaussLLMConfig, 
    GaussChatResponse
)
from app.config.settings import get_settings


class OpenAIGaussConverter:
    """Converter between OpenAI and Gauss formats"""
    
    def __init__(self, settings: Optional[Any] = None):
        if settings is None:
            settings = get_settings()
        self.settings = settings
    
    def openai_messages_to_gauss_contents(
        self, 
        messages: List[ChatMessage]
    ) -> Tuple[List[str], Optional[str]]:
        """
        Convert OpenAI messages format to Gauss contents format
        
        Args:
            messages: List of OpenAI ChatMessage objects
            
        Returns:
            Tuple of (contents_list, system_prompt)
        """
        contents = []
        system_prompt = None
        
        for message in messages:
            if message.role == "system":
                # Use the last system message as system prompt
                system_prompt = message.content
            elif message.role in ["user", "assistant"]:
                # Add user and assistant messages to contents
                if message.content:
                    contents.append(message.content)
        
        return contents, system_prompt
    
    def openai_params_to_gauss_config(
        self, 
        request: ChatCompletionRequest
    ) -> GaussLLMConfig:
        """
        Convert OpenAI parameters to Gauss LLM config
        Following the exact constraints from the PDF manual
        
        Args:
            request: OpenAI ChatCompletionRequest
            
        Returns:
            GaussLLMConfig object
        """
        # Handle frequency penalty -> repetition penalty conversion
        # Manual constraint: repetition_penalty affects repeated words appearing
        # Higher frequency penalty should reduce repetition (inverse relationship)
        repetition_penalty = 1.04  # Default from manual
        if request.frequency_penalty is not None:
            # Convert frequency penalty (-2 to 2) to repetition penalty
            # Higher frequency penalty = lower repetition penalty
            repetition_penalty = max(0.5, min(2.0, 1.04 - (request.frequency_penalty * 0.1)))
        
        # Handle presence penalty (combine with repetition penalty effect)
        if request.presence_penalty is not None:
            presence_effect = request.presence_penalty * 0.05
            repetition_penalty = max(0.5, min(2.0, repetition_penalty - presence_effect))
        
        # Ensure temperature is within manual constraints (0 < temperature < 1)
        temperature = request.temperature or self.settings.temperature_default
        temperature = max(0.01, min(0.99, temperature))  # Ensure 0 < temp < 1
        
        # Ensure top_p is within manual constraints (0.0 < top_p < 1.0)
        top_p = request.top_p or self.settings.top_p_default
        top_p = max(0.01, min(0.99, top_p))  # Ensure 0.0 < top_p < 1.0
        
        return GaussLLMConfig(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=min(
                request.max_tokens or 2024, 
                self.settings.max_tokens_limit
            ),
            repetition_penalty=repetition_penalty,
            decoder_input_details=True,
            return_full_text=False,
            seed=None,  # Manual shows this as optional, defaulting to null
            top_k=14,   # Manual default, constraint: 1 <= top_k
            do_sample=True  # From manual example
        )
    
    def gauss_response_to_openai(
        self, 
        gauss_response: GaussChatResponse,
        model: str,
        completion_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        """
        Convert Gauss response to OpenAI format
        
        Args:
            gauss_response: GaussChatResponse object
            model: Model name to use in response
            completion_id: Optional completion ID (generated if not provided)
            
        Returns:
            ChatCompletionResponse object
        """
        if completion_id is None:
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        
        # Map Gauss finish reason to OpenAI format
        finish_reason = "stop"
        if gauss_response.finish_reason:
            if "length" in gauss_response.finish_reason.lower():
                finish_reason = "length"
            elif "filter" in gauss_response.finish_reason.lower():
                finish_reason = "content_filter"
        
        # Create the assistant message
        assistant_message = ChatMessage(
            role="assistant",
            content=gauss_response.content or ""
        )
        
        # Create choice
        choice = ChatCompletionChoice(
            index=0,
            message=assistant_message,
            finish_reason=finish_reason
        )
        
        # Create usage info
        usage = ChatCompletionUsage(
            prompt_tokens=gauss_response.prompt_token or 0,
            completion_tokens=gauss_response.completion_token or 0,
            total_tokens=(
                (gauss_response.prompt_token or 0) + 
                (gauss_response.completion_token or 0)
            )
        )
        
        return ChatCompletionResponse(
            id=completion_id,
            model=model,
            choices=[choice],
            usage=usage,
            created=int(time.time())
        )
    
    def create_openai_stream_chunk(
        self,
        content: str,
        model: str,
        completion_id: str,
        is_first: bool = False,
        is_last: bool = False,
        finish_reason: Optional[str] = None
    ) -> ChatCompletionStreamResponse:
        """
        Create an OpenAI streaming response chunk
        
        Args:
            content: Content to include in the chunk
            model: Model name
            completion_id: Completion ID
            is_first: Whether this is the first chunk (includes role)
            is_last: Whether this is the last chunk
            finish_reason: Reason for completion finish
            
        Returns:
            ChatCompletionStreamResponse object
        """
        # Create delta based on chunk type
        if is_first:
            # First chunk includes role
            delta = ChatMessage(role="assistant", content=content if content else None)
        elif is_last:
            # Last chunk is empty with finish reason
            delta = ChatMessage(role=None, content=None)
        else:
            # Regular content chunk
            delta = ChatMessage(role=None, content=content)
        
        choice = ChatCompletionStreamChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason if is_last else None
        )
        
        return ChatCompletionStreamResponse(
            id=completion_id,
            model=model,
            choices=[choice],
            created=int(time.time())
        )
    
    def parse_gauss_stream_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a line from Gauss streaming response
        
        Args:
            line: Raw line from stream
            
        Returns:
            Parsed data or None if invalid
        """
        try:
            # Handle Server-Sent Events format
            if line.startswith('data: '):
                data_str = line[6:].strip()
                if data_str == '[DONE]':
                    return {'type': 'done'}
                
                try:
                    data = json.loads(data_str)
                    return {'type': 'data', 'data': data}
                except json.JSONDecodeError:
                    # Handle plain text content
                    return {'type': 'content', 'content': data_str}
            
            # Handle plain JSON lines
            elif line.strip():
                try:
                    data = json.loads(line)
                    return {'type': 'data', 'data': data}
                except json.JSONDecodeError:
                    # Handle plain text content
                    return {'type': 'content', 'content': line.strip()}
            
            return None
            
        except Exception as e:
            # Log error but don't break streaming
            return None
    
    def gauss_models_to_openai(self, gauss_models: List[Dict[str, Any]]) -> ModelListResponse:
        """
        Convert Gauss models to OpenAI format
        
        Args:
            gauss_models: List of Gauss model dictionaries
            
        Returns:
            ModelListResponse object
        """
        openai_models = []
        
        # Add default Gauss model
        openai_models.append(ModelInfo(
            id="gauss",
            object="model",
            created=int(time.time()),
            owned_by="samsung"
        ))
        
        # Add any additional models from Gauss API
        for model in gauss_models:
            if isinstance(model, dict):
                model_id = model.get('modelId') or model.get('id', 'gauss')
                openai_models.append(ModelInfo(
                    id=model_id,
                    object="model",
                    created=int(time.time()),
                    owned_by="samsung"
                ))
        
        return ModelListResponse(
            object="list",
            data=openai_models
        )
    
    def validate_openai_request(self, request: ChatCompletionRequest) -> List[str]:
        """
        Validate OpenAI request according to Gauss manual constraints
        
        Args:
            request: ChatCompletionRequest to validate
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Check required fields
        if not request.messages:
            errors.append("messages field is required")
        
        if not request.model:
            errors.append("model field is required")
        
        # Check message format
        for i, message in enumerate(request.messages):
            if not message.role:
                errors.append(f"message[{i}].role is required")
            
            if message.role not in ["system", "user", "assistant", "function"]:
                errors.append(f"message[{i}].role must be one of: system, user, assistant, function")
            
            if not message.content and message.role != "function":
                errors.append(f"message[{i}].content is required for role {message.role}")
        
        # Check parameter ranges according to manual constraints
        if request.temperature is not None:
            if request.temperature <= 0 or request.temperature >= 1:
                errors.append("temperature must be between 0 and 1 (exclusive) - manual constraint: 0 < temperature < 1")
        
        if request.top_p is not None:
            if request.top_p <= 0 or request.top_p >= 1:
                errors.append("top_p must be between 0 and 1 (exclusive) - manual constraint: 0.0 < top_p < 1.0")
        
        if request.max_tokens is not None and request.max_tokens < 1:
            errors.append("max_tokens must be greater than 0")
        
        if request.n is not None and request.n != 1:
            errors.append("n parameter must be 1 (Gauss limitation)")
        
        # Validate frequency and presence penalties
        if request.frequency_penalty is not None and (request.frequency_penalty < -2 or request.frequency_penalty > 2):
            errors.append("frequency_penalty must be between -2 and 2")
            
        if request.presence_penalty is not None and (request.presence_penalty < -2 or request.presence_penalty > 2):
            errors.append("presence_penalty must be between -2 and 2")
        
        return errors