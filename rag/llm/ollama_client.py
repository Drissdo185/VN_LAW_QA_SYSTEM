import json
import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import httpx
from llama_index.core.llms import (
    CompletionResponse, 
    CompletionResponseGen, 
    LLMMetadata,
    CustomLLM
)

logger = logging.getLogger(__name__)

@dataclass
class OllamaConfig:
    """Configuration for Ollama server."""
    api_url: str = "http://192.168.100.125:11434"
    model_name: str = "qwen2.5:32b"
    temperature: float = 0.7
    max_tokens: int = 65536
    top_p: float = 0.95
    timeout: float = 120.0
    request_timeout: float = 120.0
    
class OllamaClient(CustomLLM):
    """Client for Ollama API serving local models."""
    
    def __init__(
        self,
        api_url: str = "http://192.168.100.125:11434",
        model_name: str = "qwen2.5:32b",
        temperature: float = 0.7,
        max_tokens: int = 65536,
        top_p: float = 0.95,
        timeout: float = 120.0,
        request_timeout: float = 120.0,
    ) -> None:
        """Initialize the Ollama client."""
        super().__init__()
        self._api_url = api_url
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._timeout = timeout
        self._request_timeout = request_timeout
        self._client = httpx.AsyncClient(timeout=request_timeout)
        
        logger.info(f"Initialized Ollama client for model: {model_name} at {api_url}")
    
    @property
    def api_url(self) -> str:
        return self._api_url
        
    @property
    def model_name(self) -> str:
        return self._model_name
        
    @property
    def temperature(self) -> float:
        return self._temperature
        
    @property
    def max_tokens(self) -> int:
        return self._max_tokens
        
    @property
    def top_p(self) -> float:
        return self._top_p
        
    @property
    def timeout(self) -> float:
        return self._timeout
        
    @property
    def request_timeout(self) -> float:
        return self._request_timeout
        
    @classmethod
    def from_config(cls, config: "OllamaConfig") -> "OllamaClient":
        """Create a OllamaClient from a configuration object."""
        try:
            logger.info(f"Creating OllamaClient with config: {config.__dict__}")
            return cls(
                api_url=config.api_url,
                model_name=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                timeout=config.timeout,
                request_timeout=config.request_timeout,
            )
        except Exception as e:
            logger.error(f"Error creating OllamaClient from config: {str(e)}")
            logger.info("Falling back to default OllamaClient initialization")
            return cls()
        
    @property
    def metadata(self) -> LLMMetadata:
        """Return metadata about the LLM."""
        return LLMMetadata(
            model_name=self._model_name,
            max_input_tokens=32768,  # Default context window size
            max_output_tokens=self._max_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
        )
        
    async def _complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponse:
        """Complete the prompt using Ollama API."""
        completion_url = f"{self._api_url}/api/generate"
        
        # Prepare payload for Ollama API
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
            "top_p": kwargs.get("top_p", self._top_p),
            "stream": False,
        }
        
        try:
            logger.debug(f"Sending request to Ollama API: {completion_url}")
            response = await self._client.post(
                completion_url,
                json=payload,
                timeout=self._timeout
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Extract response from Ollama's response format
            generated_text = response_data.get("response", "")
            if not generated_text:
                raise ValueError(f"Unexpected response format: {response_data}")
                
            return CompletionResponse(text=generated_text)
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
            
    async def _acomplete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponse:
        """Asynchronously complete the prompt."""
        return await self._complete(prompt, **kwargs)
        
    async def _acompletion_stream(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        """Stream the completion (not implemented)."""
        raise NotImplementedError("Streaming completion not implemented for Ollama client")
        
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete the prompt (blocking version)."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._acomplete(prompt, **kwargs))
        
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream the completion (not implemented)."""
        raise NotImplementedError("Streaming completion not implemented for Ollama client")