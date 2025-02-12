from typing import Optional
import logging
import weaviate
from weaviate.classes.init import Auth
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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
        try:
            logger.info("Initializing Weaviate client")
            logger.info(f"Connecting to: {self.weaviate_config.url}")
            
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.weaviate_config.url,
                auth_credentials=Auth.api_key(self.weaviate_config.api_key)
            )
            logger.info("Successfully connected to Weaviate")
            
            self.vector_store = WeaviateVectorStore(
                weaviate_client=self.client,
                index_name=self.weaviate_config.collection
            )
            logger.info(f"Vector store initialized with collection: {self.weaviate_config.collection}")
            
            embed_model = HuggingFaceEmbedding(
                model_name="dangvantuan/vietnamese-document-embedding",
                max_length=256,
                trust_remote_code=True
            )
            logger.info("Embedding model initialized")
            
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=embed_model
            )
            logger.info("Vector store index created successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}", exc_info=True)
            raise
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """Get the vector store index if initialized"""
        if not self.index:
            logger.warning("Attempting to get index before initialization")
        return self.index
    
    def cleanup(self):
        """Clean up resources"""
        if self.client:
            try:
                self.client.close()
                logger.info("Weaviate client closed successfully")
            except Exception as e:
                logger.error(f"Error closing Weaviate client: {e}")