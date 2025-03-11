from typing import Optional
import weaviate
from weaviate.classes.init import Auth
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from functools import lru_cache
import logging
import time
import os

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
            logger.info(f"Initializing connection to Weaviate at {self.weaviate_config.url}")
            logger.info(f"Using collection: {self.weaviate_config.collection}")
            
            # Ensure client is connected
            self._ensure_client_connected()
            
            collection = self.weaviate_config.collection
            embed_model = self._get_cached_embedding_model()
            
            # Initialize vector store
            vector_store = WeaviateVectorStore(
                weaviate_client=self.client,
                index_name=collection,
                text_key="text"  # Explicitly set text key
            )
            
            # Initialize index
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embed_model
            )
            
            self.vector_store = vector_store
            self.index = index
            logger.info(f"Successfully initialized vector store and index for collection {collection}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    def _ensure_client_connected(self):
        """Ensure Weaviate client is connected, reconnect if needed"""
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Check if client exists and is connected
                if self.client:
                    # Test connection with a simple query
                    try:
                        
                        logger.info("Existing Weaviate connection is active")
                        return
                    except Exception as e:
                        logger.warning(f"Existing connection is broken, reconnecting: {str(e)}")
                        try:
                            self.client.close()
                        except:
                            pass
                        self.client = None
                
                # Create new connection if needed
                if not self.client:
                    logger.info(f"Creating new Weaviate client connection (attempt {retry_count+1}/{max_retries})")
                    self.client = weaviate.connect_to_weaviate_cloud(
                        cluster_url=os.getenv("WEAVIATE_TRAFFIC_URL"),
                        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_TRAFFIC_KEY")),
                        skip_init_checks=False  # Changed to validate connection
                    )
                    logger.info("Successfully connected to Weaviate")
                    return
            except Exception as e:
                last_error = e
                logger.error(f"Connection attempt {retry_count+1} failed: {str(e)}")
                retry_count += 1
                time.sleep(1)
                
    
    @lru_cache(maxsize=1)
    def _get_cached_embedding_model(self):
        """Get cached embedding model to avoid reloading"""
        logger.info(f"Loading embedding model: {self.model_config.embedding_model}")
        return HuggingFaceEmbedding(
            model_name=self.model_config.embedding_model,
            device=self.model_config.device,
            max_length=512,  
            trust_remote_code=True
        )
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """Get the vector store index"""
        if not self.index:
            logger.warning("Index requested but not initialized, attempting initialization")
            self.initialize()
            
    
        return self.index
    
    def cleanup(self):
        """Clean up resources"""
        if self.client:
            try:
                logger.info("Closing Weaviate client connection")
                self.client.close()
                logger.info("Weaviate client connection closed")
            except Exception as e:
                logger.warning(f"Error closing Weaviate client: {str(e)}")
            finally:
                self.client = None