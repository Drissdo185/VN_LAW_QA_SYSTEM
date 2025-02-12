from dataclasses import dataclass
import os
import torch
from typing import Optional, Dict


@dataclass
class WeaviateConfig:
    """Configuration for Weaviate vector database"""
    url: str
    api_key: str
    collection: str
    description: str = ""
    is_available: bool = False  # Flag to indicate if domain is ready

    def validate(self) -> bool:
        """Validate configuration values"""
        return bool(self.url and self.api_key and self.collection)


# Create domain configurations outside the class with availability flags
DOMAIN_CONFIGS = {
    "giao_thong": WeaviateConfig(
        url=os.getenv("WEAVIATE_TRAFFIC_URL", ""),
        api_key=os.getenv("WEAVIATE_TRAFFIC_KEY", ""),
        collection="ND168",
        description="Tra cứu văn bản pháp luật về giao thông, vận tải, an toàn đường bộ",
        is_available=True  # Only Giao Thông is available
    # ),
    # "chung_khoan": WeaviateConfig(
    #     url=os.getenv("WEAVIATE_STOCK_URL", ""),
    #     api_key=os.getenv("WEAVIATE_STOCK_KEY", ""),
    #     collection="StockLaws",
    #     description="Tra cứu luật chứng khoán, quy định thị trường vốn",
    #     is_available=False  # Not available yet
    # ),
    # "lao_dong": WeaviateConfig(
    #     url=os.getenv("WEAVIATE_LABOR_URL", ""),
    #     api_key=os.getenv("WEAVIATE_LABOR_KEY", ""),
    #     collection="LaborLaws",
    #     description="Tra cứu luật lao động, quy định về việc làm",
    #     is_available=False  # Not available yet
    )
}


@dataclass
class ModelConfig:
    """Configuration for ML models and LLM"""
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedding_model: str = "dangvantuan/vietnamese-document-embedding"
    cross_encoder_model: str = "dangvantuan/vietnamese-document-embedding"
    chunk_size: int = 512
    chunk_overlap: int = 50
    llm_model: str = "gpt-4o-mini"
    llm_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    def validate(self) -> bool:
        """Validate configuration values"""
        return bool(
            self.embedding_model and 
            self.cross_encoder_model and 
            self.llm_model and 
            self.llm_api_key
        )


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval"""
    vector_store_query_mode: str = "hybrid"
    similarity_top_k: int = 7
    alpha: float = 0.7

    def validate(self) -> bool:
        """Validate configuration values"""
        return (
            self.similarity_top_k > 0 and
            0 <= self.alpha <= 1
        )


def load_and_validate_configs():
    """Load and validate all configurations"""
    model_config = ModelConfig()
    retrieval_config = RetrievalConfig()
    
    # Validate configs
    if not model_config.validate():
        raise ValueError("Invalid model configuration")
    
    if not retrieval_config.validate():
        raise ValueError("Invalid retrieval configuration")
    
    # Only validate available domains
    for domain, config in DOMAIN_CONFIGS.items():
        if config.is_available and not config.validate():
            raise ValueError(f"Invalid Weaviate configuration for domain: {domain}")
    
    return model_config, retrieval_config


def get_domain_descriptions() -> Dict[str, str]:
    """Get descriptions for available domains"""
    return {
        domain: config.description 
        for domain, config in DOMAIN_CONFIGS.items()
        if config.is_available
    }


def get_available_domains() -> Dict[str, WeaviateConfig]:
    """Get only available domains"""
    return {
        domain: config 
        for domain, config in DOMAIN_CONFIGS.items()
        if config.is_available
    }