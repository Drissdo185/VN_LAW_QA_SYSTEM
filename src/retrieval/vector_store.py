from typing import Optional
import weaviate
from weaviate.classes.init import Auth
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config.config import WeaviateConfig, ModelConfig


class VectorStoreManager:
    def __init__(
        self,
        weaviate_config: WeaviateConfig,
        model_config: ModelConfig,):
        
        self.weaviate_config = weaviate_config
        self.model_config = model_config
        self.weaviate = None,
        self.vector_store = None
        self.index = None
        
        
    
    def initialize(self):
        """Initialize Weaviate client and vector store"""
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.weaviate_config.url,
            auth_credentials=Auth.api_key(self.weaviate_config.api_key))
        
        
        self.vector_store = WeaviateVectorStore(
            weaviate_client=self.client,
            index_name=self.weaviate_config.collection
        )
        
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model= HuggingFaceEmbedding(
                model_name="dangvantuan/vietnamese-document-embedding",
                max_length=256,
                trust_remote_code=True
            ),
        )
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """Get the vector store index if initialized"""
        return self.index
    
    def cleanup(self):
        """Clean up resources"""
        if self.client:
            self.client.close()

        