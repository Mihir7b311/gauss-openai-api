"""
Custom exceptions for the Gauss OpenAI API
"""

from typing import Optional, Dict, Any


class BaseAPIException(Exception):
    """Base exception for API errors"""
    
    def __init__(
        self, 
        message: str, 
        status_code: int = 500,
        error_type: str = "internal_error",
        error_code: Optional[str] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format"""
        error_dict = {
            "message": self.message,
            "type": self.error_type
        }
        
        if self.error_code:
            error_dict["code"] = self.error_code
            
        return {"error": error_dict}


class GaussAPIError(BaseAPIException):
    """Exception for Gauss API errors"""
    
    def __init__(
        self, 
        message: str, 
        status_code: int = 502,
        error_code: Optional[str] = None
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            error_type="gauss_api_error",
            error_code=error_code
        )


class GaussConnectionError(BaseAPIException):
    """Exception for Gauss connection errors"""
    
    def __init__(
        self, 
        message: str, 
        status_code: int = 503,
        error_code: Optional[str] = None
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            error_type="gauss_connection_error",
            error_code=error_code
        )


class ValidationError(BaseAPIException):
    """Exception for request validation errors"""
    
    def __init__(
        self, 
        message: str, 
        status_code: int = 400,
        error_code: Optional[str] = None
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            error_type="invalid_request_error",
            error_code=error_code
        )


class ModelNotFoundError(BaseAPIException):
    """Exception for model not found errors"""
    
    def __init__(
        self, 
        model: str,
        status_code: int = 404,
        error_code: Optional[str] = "model_not_found"
    ):
        message = f"Model '{model}' not found"
        super().__init__(
            message=message,
            status_code=status_code,
            error_type="invalid_request_error",
            error_code=error_code
        )


class RateLimitError(BaseAPIException):
    """Exception for rate limit errors"""
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded",
        status_code: int = 429,
        error_code: Optional[str] = "rate_limit_exceeded"
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            error_type="rate_limit_error",
            error_code=error_code
        )


class AuthenticationError(BaseAPIException):
    """Exception for authentication errors"""
    
    def __init__(
        self, 
        message: str = "Invalid authentication credentials",
        status_code: int = 401,
        error_code: Optional[str] = "invalid_api_key"
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            error_type="invalid_request_error",
            error_code=error_code
        )