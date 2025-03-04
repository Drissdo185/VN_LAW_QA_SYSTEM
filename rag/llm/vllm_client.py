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
class VLLMConfig:
    """Configuration for vLLM server."""
    api_url: str = "http://192.168.100.125:8000"
    model_name: str = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    timeout: float = 120.0
    request_timeout: float = 120.0
    
class VLLMClient(CustomLLM):
    """Client for vLLM API serving Qwen2.5 models."""
    
    def __init__(
        self,
        api_url: str = "http://192.168.100.125:8000",
        model_name: str = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        timeout: float = 120.0,
        request_timeout: float = 120.0,
    ) -> None:
        """Initialize the vLLM client.
        
        Args:
            api_url: The URL of the vLLM server.
            model_name: Name of the model.
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.
            top_p: Top-p sampling parameter.
            timeout: Timeout for the request in seconds.
            request_timeout: Timeout for the request in seconds.
        """
        super().__init__()
        self.api_url = "http://192.168.100.125:8000"
        self.model_name = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.request_timeout = request_timeout
        self._client = httpx.AsyncClient(timeout=request_timeout)
        
        logger.info(f"Initialized vLLM client for model: {model_name}")
        
    @classmethod
    def from_config(cls, config: VLLMConfig) -> "VLLMClient":
        """Create a VLLMClient from a configuration object."""
        return cls(
            api_url=config.api_url,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            timeout=config.timeout,
            request_timeout=config.request_timeout,
        )
        
    @property
    def metadata(self) -> LLMMetadata:
        """Return metadata about the LLM."""
        return LLMMetadata(
            model_name=self.model_name,
            max_input_tokens=4096,  # Qwen2.5-14B context window size
            max_output_tokens=self.max_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
        )
        
    async def _complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponse:
        """Complete the prompt using vLLM API."""
        completion_url = f"{self.api_url}/v1/completions"
        
        # Prepare payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": False,
        }
        
        try:
            logger.debug(f"Sending request to vLLM API: {completion_url}")
            response = await self._client.post(
                completion_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" not in response_data or not response_data["choices"]:
                raise ValueError(f"Unexpected response format: {response_data}")
                
            generated_text = response_data["choices"][0]["text"]
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
        raise NotImplementedError("Streaming completion not implemented for vLLM client")
        
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete the prompt (blocking version)."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._acomplete(prompt, **kwargs))
        
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream the completion (not implemented)."""
        raise NotImplementedError("Streaming completion not implemented for vLLM client")