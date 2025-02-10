from dataclasses import dataclass
import os
from typing import Optional


@dataclass
class WeaviateConfig:
    url: os.getenv("WEAVIATE_TRAFFIC_URL")
    api_key: os.getenv("WEAVIATE_TRAFFIC_KEY")
    collection: "ND168"

@dataclass
class ModelConfig:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedding_model: str = "dangvantuan/vietnamese-document-embedding"
    chunk_size: int = 512
    chunk_overlap: int = 50
    llm_model: str = "gpt-4o-mini"
    llm_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
@dataclass
class RetrievalConfig:
    vector_store_query_mode: str = "hybrid"
    similarity_top_k: int = 5
    alpha: float = 0.5