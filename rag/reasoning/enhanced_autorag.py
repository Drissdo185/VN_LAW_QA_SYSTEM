from typing import Dict, Any, Optional, List
import logging
from retrieval.query_simplifier import QuerySimplifier
from config.config import ModelConfig
from retrieval.retriever import DocumentRetriever
from retrieval.search_pipline import SearchPipeline
from reasoning.auto_rag import AutoRAG

logger = logging.getLogger(__name__)

class EnhancedAutoRAG:
    """
    Enhanced version of AutoRAG that incorporates query standardization
    before document retrieval to improve relevance.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        retriever: DocumentRetriever,
        search_pipeline: Optional[SearchPipeline] = None,
        max_iterations: int = 3
    ):
        self.model_config = model_config
        self.retriever = retriever
        self.search_pipeline = search_pipeline
        self.max_iterations = max_iterations
        
        # Initialize components
        self.query_simplifier = QuerySimplifier(model_config)
        self.auto_rag = AutoRAG(
            model_config=model_config,
            retriever=retriever,
            search_pipeline=search_pipeline,
            max_iterations=max_iterations
        )
        
        logger.info("Initialized EnhancedAutoRAG with query standardization")
    
    async def get_answer(self, question: str) -> Dict[str, Any]:
        """
        Get answer for a traffic-related question using the enhanced RAG flow:
        1. Standardize the query using LLM
        2. Use standardized query for retrieval
        3. Generate answer based on retrieved documents
        
        Args:
            question: The original user question
        
        Returns:
            Dict containing answer, analysis, and metadata
        """
        try:
            # Step 1: Standardize the query
            logger.info(f"Processing question via standardized flow: {question}")
            simplification_result = await self.query_simplifier.simplify_query(question)
            
            # Get the standardized query from the result
            standardized_query = simplification_result.get("standardized_query", question)
            query_metadata = simplification_result.get("metadata", {})
            
            # Log information about standardization
            logger.info(f"Original query: '{question}'")
            logger.info(f"Standardized query: '{standardized_query}'")
            logger.info(f"Query metadata: {query_metadata}")
            
            # Step 2: Use original AutoRAG with standardized query
            response = await self.auto_rag.get_answer(standardized_query)
            
            # Step 3: Enhance response with query metadata
            response["query_info"] = {
                "original_query": question,
                "standardized_query": standardized_query,
                "metadata": query_metadata
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in EnhancedAutoRAG: {str(e)}")
            return {
                "error": f"Error processing question: {str(e)}",
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }
            }