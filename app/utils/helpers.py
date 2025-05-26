"""
Helper utility functions
"""

import time
import uuid
from typing import Dict, List, Any, Optional


def generate_completion_id() -> str:
    """Generate a unique completion ID in OpenAI format"""
    return f"chatcmpl-{uuid.uuid4().hex}"


def generate_model_id() -> str:
    """Generate a unique model ID"""
    return f"model-{uuid.uuid4().hex}"


def current_timestamp() -> int:
    """Get current Unix timestamp"""
    return int(time.time())


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary"""
    return data.get(key, default) if data else default


def flatten_dict(data: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten a nested dictionary
    
    Args:
        data: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def chunk_list(data: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size
    
    Args:
        data: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def sanitize_string(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize a string for logging or display
    
    Args:
        text: String to sanitize
        max_length: Maximum length (truncate if exceeded)
        
    Returns:
        Sanitized string
    """
    if not text:
        return ""
    
    # Remove control characters
    sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Truncate if needed
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length - 3] + "..."
    
    return sanitized


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def validate_model_name(model: str) -> bool:
    """
    Validate if a model name is acceptable
    
    Args:
        model: Model name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not model:
        return False
    
    # Allow common OpenAI model names and Gauss
    valid_models = {
        "gauss",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-turbo-preview",
        "gpt-4-vision-preview"
    }
    
    return model.lower() in valid_models or model.startswith("gauss")


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count for a text string
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Rough approximation: 1 token ≈ 4 characters for English text
    # This is a very rough estimate and not accurate for production use
    return max(1, len(text) // 4)