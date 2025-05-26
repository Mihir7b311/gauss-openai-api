"""
OpenAI-compatible Pydantic models
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
import time


class ChatMessage(BaseModel):
    """OpenAI chat message model"""
    role: Literal["system", "user", "assistant", "function"]
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request model"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=128)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None


class ChatCompletionChoice(BaseModel):
    """OpenAI chat completion choice model"""
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "function_call", "content_filter"]] = None


class ChatCompletionUsage(BaseModel):
    """OpenAI chat completion usage model"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response model"""
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    system_fingerprint: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    """OpenAI chat completion stream choice model"""
    index: int
    delta: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "function_call", "content_filter"]] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI chat completion stream response model"""
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]
    system_fingerprint: Optional[str] = None


class ModelInfo(BaseModel):
    """OpenAI model info model"""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str


class ModelListResponse(BaseModel):
    """OpenAI model list response model"""
    object: str = "list"
    data: List[ModelInfo]


class ErrorDetail(BaseModel):
    """OpenAI error detail model"""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """OpenAI error response model"""
    error: ErrorDetail