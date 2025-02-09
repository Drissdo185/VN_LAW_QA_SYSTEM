from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
import os

class DomainStatus(Enum):
    """Status of domain availability."""
    ACTIVE = "active"
    IN_DEVELOPMENT = "in_development"

@dataclass
class DomainConfig:
    """Configuration for each legal domain."""
    collection_name: str
    cluster_url: str
    api_key: str
    description: str
    status: DomainStatus = DomainStatus.IN_DEVELOPMENT
    similarity_top_k: int = 10
    dense_weight: float = 0.3
    embedding_model: Optional[str] = None
    
    @property
    def is_available(self) -> bool:
        """Check if domain is available for use."""
        return self.status == DomainStatus.ACTIVE

DOMAIN_CONFIGS: Dict[str, DomainConfig] = {
    "Giao thông": DomainConfig(
        collection_name="ND168",
        cluster_url=os.getenv("WEAVIATE_TRAFFIC_URL"),
        api_key=os.getenv("WEAVIATE_TRAFFIC_KEY"),
        description="Văn bản pháp luật về giao thông",
        status=DomainStatus.ACTIVE,
        embedding_model="dangvantuan/vietnamese-document-embedding"
    ),
    "Doanh nghiệp": DomainConfig(
        collection_name="Business_law",
        cluster_url="WEAVIATE_BUSINESS_URL",
        api_key="WEAVIATE_BUSINESS_KEY",
        description="Văn bản pháp luật về doanh nghiệp"
    ),
    "Chính trị": DomainConfig(
        collection_name="Politics_law",
        cluster_url="WEAVIATE_POLITICS_URL",
        api_key="WEAVIATE_POLITICS_KEY",
        description="Văn bản pháp luật về chính trị"
    )
}