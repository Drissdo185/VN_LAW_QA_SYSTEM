from dataclasses import dataclass
import os
import torch
from typing import Optional, Literal
from enum import Enum

class Domain(str, Enum):
    TRAFFIC = "traffic"
    STOCK = "stock"

@dataclass
class WeaviateConfig:
    url: str
    api_key: str
    traffic_collection: str = "ND168"
    stock_collection: str = "STOCK"
    
    def get_collection(self, domain: Domain) -> str:
        return self.traffic_collection if domain == Domain.TRAFFIC else self.stock_collection

@dataclass
class ModelConfig:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedding_model: str = "dangvantuan/vietnamese-document-embedding"
    cross_encoder_model: str = "dangvantuan/vietnamese-document-embedding"
    chunk_size: int = 512
    chunk_overlap: int = 50
    llm_model: str = "gpt-4o-mini"
    llm_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
@dataclass
class RetrievalConfig:
    vector_store_query_mode: str = "hybrid"
    similarity_top_k: int = 8
    alpha: float = 0.5
    

@dataclass
class WebSearchConfig:
    google_api_key: str = os.getenv("GOOGLE_API_KEY")
    google_cse_id: str = os.getenv("GOOGLE_CSE_ID")
    fallback_threshold: float = 0.5
    web_search_enabled: bool = True