from typing import Dict, Any, Optional, List
import logging
from retrieval.query_simplifier import QuerySimplifier
from config.config import ModelConfig, RetrievalConfig
from retrieval.retriever import DocumentRetriever
from retrieval.search_pipline import SearchPipeline
from reasoning.auto_rag import AutoRAG
from retrieval.traffic_synonyms import TrafficSynonymExpander

logger = logging.getLogger(__name__)

class EnhancedAutoRAG:
    
    
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
        
        
        self.synonym_expander = TrafficSynonymExpander()
        self.query_simplifier = QuerySimplifier(model_config)
        
        
        self.auto_rag = AutoRAG(
            model_config=model_config,
            retriever=retriever,
            search_pipeline=search_pipeline,
            max_iterations=max_iterations
        )
        
        logger.info("Initialized EnhancedAutoRAG with synonym expansion and query standardization")
    
    async def get_answer(self, question: str) -> Dict[str, Any]:
        try:
            logger.info(f"Original query: '{question}'")
            
    
            expanded_query = self.synonym_expander.expand_query(question)
            logger.info(f"Query after synonym expansion: '{expanded_query}'")
            
           
            logger.info(f"Standardizing expanded query")
            simplification_result = await self.query_simplifier.simplify_query(expanded_query)
            
        
            
           
            
            
            response = await self.auto_rag.get_answer(simplification_result)

            
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