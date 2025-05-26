"""
Gauss-specific Pydantic models
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field



class GaussLLMConfig(BaseModel):
    """Gauss LLM configuration model - matches PDF manual exactly"""
    temperature: float = Field(default=0.4, ge=0.0, le=1.0)  # Manual shows 0 < temperature < 1
    repetition_penalty: float = Field(default=1.04, ge=0.0, le=2.0)
    decoder_input_details: bool = True
    return_full_text: bool = False
    seed: Optional[int] = None
    top_k: int = Field(default=14, ge=1)  # Manual shows 1 <= top_k
    top_p: float = Field(default=0.94, ge=0.0, le=1.0)  # Manual shows 0.0 < top_p < 1.0
    max_new_tokens: int = Field(default=2024, ge=1)
    do_sample: bool = True


class GaussChatRequest(BaseModel):
    """Gauss chat request model - exactly as per PDF manual"""
    llm_id: int = Field(default=1, alias="llmId")  # Optional, default: basic LLM
    llm_name: str = Field(default="Gauss", alias="llmName")  # Deprecated, default: basic LLM
    contents: List[str]  # Required - Contents array
    is_stream: bool = Field(default=False, alias="isStream")  # Optional, default: true (manual says true, but we default to false)
    llm_config: GaussLLMConfig = Field(alias="llmConfig")  # Optional - LLM Config info
    system_prompt: Optional[str] = Field(default=None, alias="systemPrompt")  # Optional - System prompt for LLM to use


class GaussChatResponse(BaseModel):
    """Gauss chat response model - comprehensive fields from PDF manual"""
    id: Optional[int] = None
    parent_message_id: Optional[str] = Field(default=None, alias="parentMessageId")
    parent_message_created_at: Optional[str] = Field(default=None, alias="parentMessageCreatedAt")
    chat_id: Optional[int] = Field(default=None, alias="chatId")
    user_id: Optional[int] = Field(default=None, alias="userId")
    model_id: Optional[str] = Field(default=None, alias="modelId")
    model_type: Optional[str] = Field(default=None, alias="modelType")
    content: str  # Answer
    created_at: Optional[str] = Field(default=None, alias="createdAt")
    completion_token: Optional[int] = Field(default=None, alias="completionToken")  # Answer Token Usage
    prompt_token: Optional[int] = Field(default=None, alias="promptToken")  # Question Token Usage
    truncated: Optional[str] = None  # Whether the question is truncated or not
    finish_reason: Optional[str] = Field(default=None, alias="finishReason")  # Finish Reason (stop: normal, length: Answer length exceeded)
    
    # Filter block reasons
    filter_block_reason_ko: Optional[str] = Field(default=None, alias="filterBlockReason.ko")
    filter_block_reason_en: Optional[str] = Field(default=None, alias="filterBlockReason.en")
    filter_block_reason_policy_id: Optional[str] = Field(default=None, alias="filterBlockReason.policy_id")
    filter_block_reason_message: Optional[str] = Field(default=None, alias="filterBlockReason.message")
    filter_block_reason_result_code: Optional[str] = Field(default=None, alias="filterBlockReason.result_code")
    filter_block_reason_filter_log_id: Optional[str] = Field(default=None, alias="filterBlockReason.filter_log_id")
    
    status: Optional[str] = None
    response_code: Optional[str] = Field(default=None, alias="responseCode")
    plugins: Optional[List[str]] = None  # Requested plugin
    references: Optional[List[str]] = None  # References used while generating the answer
    catalogs: Optional[List[int]] = None  # Requested Catalog
    event_status: Optional[str] = Field(default=None, alias="eventStatus")
    event_data: Optional[str] = Field(default=None, alias="eventData")
    filter_validation: Optional[bool] = Field(default=None, alias="filterValidation")  # Filter success or not
    success_yn: Optional[bool] = Field(default=None, alias="successYn")  # Success or failure
    
    class Config:
        allow_population_by_field_name = True


class GaussModelInfo(BaseModel):
    """Gauss model info model"""
    model_id: str = Field(alias="modelId")
    model_name: str = Field(alias="modelName")
    model_label_en: str = Field(alias="modelLabel.en")
    model_label_ko: str = Field(alias="modelLabel.ko")
    model_description_en: str = Field(alias="modelDescription.en")
    model_description_ko: str = Field(alias="modelDescription.ko")


class GaussModelListResponse(BaseModel):
    """Gauss model list response model"""
    models: List[GaussModelInfo] = []


class GaussStreamChunk(BaseModel):
    """Gauss streaming response chunk model"""
    content: Optional[str] = None
    finish_reason: Optional[str] = Field(alias="finishReason")
    completion_token: Optional[int] = Field(alias="completionToken")
    prompt_token: Optional[int] = Field(alias="promptToken")
    event_status: Optional[str] = Field(alias="eventStatus")
    
    class Config:
        allow_population_by_field_name = True