from dataclasses import dataclass
from typing import Optional


@dataclass
class VectorStoreConfig:
    """Cấu hình cho vector store."""
    weaviate_url: str
    index_name: str