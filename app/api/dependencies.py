"""
FastAPI dependencies
"""

from functools import lru_cache
from app.services.gauss_client import GaussChatClient
from app.services.converter import OpenAIGaussConverter
from app.config.settings import get_settings


@lru_cache()
def get_gauss_client() -> GaussChatClient:
    """
    Get Gauss client instance (cached)
    """
    settings = get_settings()
    return GaussChatClient(settings)


@lru_cache()
def get_converter() -> OpenAIGaussConverter:
    """
    Get converter instance (cached)
    """
    settings = get_settings()
    return OpenAIGaussConverter(settings)