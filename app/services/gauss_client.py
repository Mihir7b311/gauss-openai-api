"""
Gauss API client service
"""

import json
import logging
from typing import List, Dict, Any, Iterator, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config.settings import get_settings
from app.models.gauss_models import GaussChatRequest, GaussChatResponse, GaussLLMConfig
from app.core.exceptions import GaussAPIError, GaussConnectionError


logger = logging.getLogger(__name__)


class GaussChatClient:
    """Client for interacting with the Gauss Chat API"""
    
    def __init__(self, settings: Optional[Any] = None):
        """Initialize the Gauss Chat client"""
        if settings is None:
            settings = get_settings()
            
        self.settings = settings
        self.base_url = settings.gauss_base_url
        self.chat_endpoint = f"{self.base_url}/messages"
        self.models_endpoint = f"{self.base_url}/models"
        
        # Headers for authentication
        self.headers = {
            "x-openapi-token": f"Bearer {settings.gauss_pass_key}",
            "x-generative-ai-client": settings.gauss_client_key,
            "Content-Type": "application/json"
        }
        
        # Proxy configuration
        self.proxy_ips = settings.gauss_proxy_ips
        self.proxy_port = settings.gauss_proxy_port
        self.timeout = settings.gauss_request_timeout
        self.working_ip = None
        
        # Setup session with retry strategy
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy"""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "POST"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_proxy_config(self, ip: str) -> Dict[str, str]:
        """Get proxy configuration for given IP"""
        proxy_url = f"http://{ip}:{self.proxy_port}"
        return {
            "http": proxy_url,
            "https": proxy_url
        }
    
    def _make_request_with_proxy_fallback(
        self, 
        method: str, 
        url: str, 
        **kwargs
    ) -> requests.Response:
        """Make a request with automatic proxy IP fallback"""
        
        # Try working IP first if available
        if self.working_ip:
            try:
                proxies = self._get_proxy_config(self.working_ip)
                logger.debug(f"Using previously working IP: {self.working_ip}")
                
                response = self.session.request(
                    method=method,
                    url=url,
                    proxies=proxies,
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                logger.debug(f"Success with working IP {self.working_ip}")
                return response
                
            except Exception as e:
                logger.warning(f"Previously working IP {self.working_ip} failed: {str(e)}")
                self.working_ip = None
        
        # Try each proxy IP
        last_exception = None
        for ip in self.proxy_ips:
            try:
                proxies = self._get_proxy_config(ip)
                logger.debug(f"Trying request with proxy IP: {ip}")
                
                response = self.session.request(
                    method=method,
                    url=url,
                    proxies=proxies,
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                
                # Store working IP for future requests
                self.working_ip = ip
                logger.info(f"Success with IP {ip}, saved as working IP")
                return response
                
            except Exception as e:
                logger.warning(f"Failed with IP {ip}: {str(e)}")
                last_exception = e
                continue
        
        # All IPs failed
        error_msg = f"All proxy IPs failed. Last error: {str(last_exception)}"
        logger.error(error_msg)
        raise GaussConnectionError(error_msg)
    
    def chat_completion(
        self,
        contents: List[str],
        llm_config: GaussLLMConfig,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> GaussChatResponse:
        """Create a chat completion (non-streaming)"""
        
        request_data = GaussChatRequest(
            contents=contents,
            llm_config=llm_config,
            system_prompt=system_prompt,
            is_stream=stream
        )
        
        try:
            logger.info(f"Sending chat request to: {self.chat_endpoint}")
            logger.debug(f"Request data: {request_data.json()}")
            
            response = self._make_request_with_proxy_fallback(
                method="POST",
                url=self.chat_endpoint,
                headers=self.headers,
                json=request_data.dict(by_alias=True, exclude_none=True)
            )
            
            response_data = response.json()
            logger.info("Chat request successful")
            logger.debug(f"Response data: {json.dumps(response_data, indent=2)}")
            
            return GaussChatResponse(**response_data)
            
        except requests.RequestException as e:
            logger.error(f"Chat request failed: {e}")
            raise GaussAPIError(f"Chat completion failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in chat completion: {e}")
            raise GaussAPIError(f"Unexpected error: {str(e)}")
    
    def chat_completion_stream(
        self,
        contents: List[str],
        llm_config: GaussLLMConfig,
        system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        """Create a streaming chat completion"""
        
        request_data = GaussChatRequest(
            contents=contents,
            llm_config=llm_config,
            system_prompt=system_prompt,
            is_stream=True
        )
        
        try:
            logger.info(f"Sending streaming chat request to: {self.chat_endpoint}")
            logger.debug(f"Request data: {request_data.json()}")
            
            # Use working IP if available, otherwise first IP
            proxy_ip = self.working_ip or self.proxy_ips[0]
            proxies = self._get_proxy_config(proxy_ip)
            
            response = self.session.post(
                self.chat_endpoint,
                headers=self.headers,
                json=request_data.dict(by_alias=True, exclude_none=True),
                proxies=proxies,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Update working IP
            if proxy_ip != self.working_ip:
                self.working_ip = proxy_ip
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line:
                            yield decoded_line
                    except Exception as e:
                        logger.warning(f"Error processing stream line: {e}")
                        continue
                        
        except requests.RequestException as e:
            logger.error(f"Streaming chat request failed: {e}")
            raise GaussAPIError(f"Streaming chat completion failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in streaming chat: {e}")
            raise GaussAPIError(f"Unexpected error: {str(e)}")
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get available models"""
        try:
            logger.info(f"Fetching models from: {self.models_endpoint}")
            
            response = self._make_request_with_proxy_fallback(
                method="GET",
                url=self.models_endpoint,
                headers=self.headers
            )
            
            models_data = response.json()
            logger.info("Models request successful")
            logger.debug(f"Models data: {json.dumps(models_data, indent=2)}")
            
            return models_data
            
        except requests.RequestException as e:
            logger.error(f"Models request failed: {e}")
            raise GaussAPIError(f"Failed to fetch models: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error fetching models: {e}")
            raise GaussAPIError(f"Unexpected error: {str(e)}")
    
    def health_check(self) -> bool:
        """Check if the Gauss API is healthy"""
        try:
            self.get_models()
            return True
        except Exception:
            return False