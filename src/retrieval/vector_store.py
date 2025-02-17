from typing import Optional, Dict
import weaviate
from weaviate.classes.init import Auth
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from functools import lru_cache

from config.config import WeaviateConfig, ModelConfig, Domain

class VectorStoreManager:
    def __init__(
        self,
        weaviate_config: WeaviateConfig,
        model_config: ModelConfig,
        domain: Domain
    ):
        self.weaviate_config = weaviate_config
        self.model_config = model_config
        self.domain = domain
        self.client = None
        self.vector_stores: Dict[Domain, WeaviateVectorStore] = {}
        self.indices: Dict[Domain, VectorStoreIndex] = {}
        
    def initialize(self):
        """Initialize Weaviate client and vector store for the specified domain"""
        self._ensure_client_connected()
        
        collection = self.weaviate_config.get_collection(self.domain)
        embed_model = self._get_cached_embedding_model()
        
        vector_store = WeaviateVectorStore(
            weaviate_client=self.client,
            index_name=collection
        )
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        
        self.vector_stores[self.domain] = vector_store
        self.indices[self.domain] = index
    
    def _ensure_client_connected(self):
        """Ensure Weaviate client is connected, reconnect if needed"""
        if not self.client:
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.weaviate_config.url,
                auth_credentials=Auth.api_key(self.weaviate_config.api_key)
            )
        else:
            try:
                # Test if client is still connected
                self.client.schema.get()
            except Exception:
                # Reconnect if client is closed
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.weaviate_config.url,
                    auth_credentials=Auth.api_key(self.weaviate_config.api_key)
                )
    
    @lru_cache(maxsize=1)
    def _get_cached_embedding_model(self):
        """Get cached embedding model to avoid reloading"""
        return HuggingFaceEmbedding(
            model_name=self.model_config.embedding_model,
            max_length=256,
            trust_remote_code=True
        )
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """Get the vector store index for the current domain"""
        self._ensure_client_connected()  # Ensure client is connected before returning index
        return self.indices.get(self.domain)
    
    def cleanup(self):
        """Clean up resources"""
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            finally:
                self.client = None