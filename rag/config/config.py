from dataclasses import dataclass, field
import os
import torch
from typing import Optional
from enum import Enum

class LLMProvider(str, Enum):
    OPENAI = "openai"
    VLLM = "vllm"
    OLLAMA = "ollama"

@dataclass
class WeaviateConfig:
    url: str
    api_key: str
    collection: str = "ND168"

@dataclass
class VLLMConfig:
    api_url: str = "http://192.168.100.125:8000"
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    temperature: float = 0.2
    max_tokens: int = 32768
    top_p: float = 0.95
    timeout: float = 120.0
    request_timeout: float = 120.0


@dataclass
class OllamaConfig:  # Add new config class
    api_url: str = "http://192.168.100.125:11434"  # Update with your server's address
    model_name: str = "qwen2.5:14b"
    temperature: float = 0.2
    max_tokens: int = 65536
    top_p: float = 0.95
    timeout: float = 120.0
    request_timeout: float = 120.0

@dataclass
class ModelConfig:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedding_model: str = "dangvantuan/vietnamese-document-embedding"
    cross_encoder_model: str = "dangvantuan/vietnamese-document-embedding"
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # LLM configuration
    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # VLLM configuration (used when llm_provider is VLLM)
    vllm_config: VLLMConfig = field(default_factory=lambda: VLLMConfig())
    
    # Ollama configuration (used when llm_provider is OLLAMA)
    ollama_config: OllamaConfig = field(default_factory=lambda: OllamaConfig())
    
@dataclass
class RetrievalConfig:
    vector_store_query_mode: str = "hybrid"
    similarity_top_k: int = 5
    alpha: float = 0.5

@dataclass
class WebSearchConfig:
    google_api_key: str = os.getenv("GOOGLE_API_KEY")
    google_cse_id: str = os.getenv("GOOGLE_CSE_ID")
    fallback_threshold: float = 0.5
    web_search_enabled: bool = True