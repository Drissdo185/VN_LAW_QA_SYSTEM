# config/domain_config.py

from dataclasses import dataclass
from typing import Dict

@dataclass
class DomainConfig:
    """Cấu hình cho từng lĩnh vực."""
    collection_name: str
    cluster_url: str
    api_key: str
    description: str


DOMAIN_CONFIGS: Dict[str, DomainConfig] = {
    "Giao thông": DomainConfig(
        collection_name="ND168",
        cluster_url="WEAVIATE_TRAFFIC_URL",  
        api_key="WEAVIATE_TRAFFIC_KEY",      
        description="Văn bản pháp luật về giao thông",
        
    ),
    "Doanh nghiệp": DomainConfig(
        collection_name="Business_law",
        cluster_url="WEAVIATE_BUSINESS_URL",  
        api_key="WEAVIATE_BUSINESS_KEY",      
        description="Văn bản pháp luật về doanh nghiệp",
        
    ),
    "Chính trị": DomainConfig(
        collection_name="Politics_law",
        cluster_url="WEAVIATE_POLITICS_URL",  
        api_key="WEAVIATE_POLITICS_KEY",      
        description="Văn bản pháp luật về chính trị",

    )
}