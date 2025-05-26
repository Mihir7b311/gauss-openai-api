"""
Configuration settings for the Gauss OpenAI API
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    api_title: str = "Gauss OpenAI Compatible API"
    api_description: str = "OpenAI-compatible API wrapper for Samsung Gauss LLM"
    api_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    reload: bool = Field(default=True, env="API_RELOAD")
    
    # Gauss API Configuration
    gauss_pass_key: str = Field(..., env="GAUSS_PASS_KEY")
    gauss_client_key: str = Field(..., env="GAUSS_CLIENT_KEY")
    gauss_base_url: str = Field(
        default="https://fabrix-dx.sec.samsung.net/apim-dev/fssedx/dx_dev_chat_v1/1/openapi/chat/v1",
        env="GAUSS_BASE_URL"
    )
    
    # Proxy Configuration
    gauss_proxy_ips: List[str] = Field(
        default=["107.99.237.66"],
        env="GAUSS_PROXY_IPS"
    )
    gauss_proxy_port: int = Field(default=9000, env="GAUSS_PROXY_PORT")
    gauss_request_timeout: int = Field(default=30, env="GAUSS_REQUEST_TIMEOUT")
    
    # OpenAI API Configuration - Updated to match manual constraints
    default_model: str = Field(default="gauss", env="DEFAULT_MODEL")
    max_tokens_limit: int = Field(default=4096, env="MAX_TOKENS_LIMIT")
    temperature_default: float = Field(default=0.4, env="TEMPERATURE_DEFAULT")  # Manual example uses 0.4
    top_p_default: float = Field(default=0.94, env="TOP_P_DEFAULT")  # Manual example uses 0.94
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["*"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(
        default=["*"],
        env="CORS_ALLOW_METHODS"
    )
    cors_allow_headers: List[str] = Field(
        default=["*"],
        env="CORS_ALLOW_HEADERS"
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings