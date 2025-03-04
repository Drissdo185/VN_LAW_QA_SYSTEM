from typing import Optional
import weaviate
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from functools import lru_cache
import logging

from config.config import WeaviateConfig, ModelConfig

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(
        self,
        weaviate_config: WeaviateConfig,
        model_config: ModelConfig
    ):
        self.weaviate_config = weaviate_config
        self.model_config = model_config
        self.client = None
        self.vector_store = None
        self.index = None
        
    def initialize(self):
        """Initialize Weaviate client and vector store"""
        self._ensure_client_connected()
        
        collection = self.weaviate_config.collection
        embed_model = self._get_cached_embedding_model()
        
        vector_store = WeaviateVectorStore(
            weaviate_client=self.client,
            index_name=collection
        )
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        
        self.vector_store = vector_store
        self.index = index
    
    def _ensure_client_connected(self):
        """Ensure Weaviate client is connected, reconnect if needed"""
        if not self.client:
            try:
                # Check which method is available in the installed weaviate version
                if hasattr(weaviate, 'connect_to_weaviate_cloud'):
                    logger.info("Using connect_to_weaviate_cloud method")
                    # Newer weaviate client version
                    self.client = weaviate.connect_to_weaviate_cloud(
                        cluster_url=self.weaviate_config.url,
                        auth_credentials=weaviate.classes.init.Auth.api_key(self.weaviate_config.api_key),
                        skip_init_checks=True
                    )
                else:
                    logger.info("Using Client constructor method")
                    # Older weaviate client version
                    auth_config = weaviate.auth.AuthApiKey(self.weaviate_config.api_key)
                    self.client = weaviate.Client(
                        url=self.weaviate_config.url,
                        auth_client_secret=auth_config
                    )
            except Exception as e:
                logger.error(f"Failed to connect to Weaviate: {str(e)}")
                raise RuntimeError(f"Failed to connect to Weaviate: {str(e)}")
        else:
            try:
                # Test if client is still connected
                self.client.schema.get()
            except Exception as e:
                logger.error(f"Client connection test failed: {str(e)}")
                # Reconnect with appropriate method
                try:
                    if hasattr(weaviate, 'connect_to_weaviate_cloud'):
                        self.client = weaviate.connect_to_weaviate_cloud(
                            cluster_url=self.weaviate_config.url,
                            auth_credentials=weaviate.classes.init.Auth.api_key(self.weaviate_config.api_key),
                            skip_init_checks=True
                        )
                    else:
                        auth_config = weaviate.auth.AuthApiKey(self.weaviate_config.api_key)
                        self.client = weaviate.Client(
                            url=self.weaviate_config.url,
                            auth_client_secret=auth_config
                        )
                except Exception as e:
                    logger.error(f"Failed to reconnect to Weaviate: {str(e)}")
                    raise RuntimeError(f"Failed to reconnect to Weaviate: {str(e)}")
    
    @lru_cache(maxsize=1)
    def _get_cached_embedding_model(self):
        """Get cached embedding model to avoid reloading"""
        return HuggingFaceEmbedding(
            model_name=self.model_config.embedding_model,
            max_length=256,
            trust_remote_code=True
        )
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """Get the vector store index"""
        self._ensure_client_connected()  # Ensure client is connected before returning index
        return self.index
    
    def cleanup(self):
        """Clean up resources"""
        if self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.warning(f"Error closing Weaviate client: {str(e)}")
            finally:
                self.client = None