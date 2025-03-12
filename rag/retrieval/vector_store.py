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
        self.is_initialized = False
        
    def initialize(self):
        """Initialize Weaviate client and vector store if not already initialized"""
        if self.is_initialized and self.client:
            logger.info("VectorStoreManager already initialized, reusing existing connection")
            return
            
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
            self.is_initialized = True
            logger.info(f"Successfully initialized vector store and index for collection {collection}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    def _ensure_client_connected(self):
        """Ensure Weaviate client is connected, reconnect if needed"""
        max_retries = 3
        retry_count = 0
        last_error = None
        
        # If we already have a client, test it with a simple query
        if self.client:
            try:
                logger.info("Existing Weaviate connection is active")
                return
            except Exception as e:
                logger.warning(f"Existing connection is broken, will reconnect: {str(e)}")
                try:
                    self.client.close()
                except:
                    pass
                self.client = None
        
        # Try to establish connection
        while retry_count < max_retries:
            try:
                logger.info(f"Creating new Weaviate client connection (attempt {retry_count+1}/{max_retries})")
                
                # Use the proper connection method with api key from environment
                # self.client = weaviate.connect_to_weaviate_cloud(
                #     cluster_url=self.weaviate_config.url,
                #     auth_credentials=Auth.api_key(self.weaviate_config.api_key),
                #     skip_init_checks=False  # Validate connection
                # )
                self.client = weaviate.connect_to_local(
                     host="192.168.100.125",
                        port=8080,
                        grpc_port=50051
                )
            
                
                logger.info("Successfully connected to Weaviate")
                return
            except Exception as e:
                last_error = e
                logger.error(f"Connection attempt {retry_count+1} failed: {str(e)}")
                retry_count += 1
                time.sleep(2)  # Wait a bit longer between retries
        
        # If we've exhausted retries, raise an exception
        if last_error:
            logger.error(f"Failed to connect to Weaviate after {max_retries} attempts")
            raise last_error
        else:
            raise ConnectionError("Failed to connect to Weaviate: Unknown error")
    
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
        """Get the vector store index, initializing if necessary"""
        if not self.is_initialized or not self.index:
            logger.info("Index requested but not initialized, initializing now")
            self.initialize()
        
        # Test connection and reinitialize if needed
        if not self.client:
            logger.warning("Client connection lost, reinitializing")
            self.initialize()
        
        return self.index
    
    def cleanup(self):
        """Clean up resources - only call this when shutting down the application"""
        if self.client:
            try:
                logger.info("Closing Weaviate client connection")
                self.client.close()
                logger.info("Weaviate client connection closed")
                self.client = None
                self.is_initialized = False
            except Exception as e:
                logger.warning(f"Error closing Weaviate client: {str(e)}")