from typing import Dict, Any, Optional, List
import logging
from retrieval.query_simplifier import QuerySimplifier
from config.config import ModelConfig, RetrievalConfig
from retrieval.retriever import DocumentRetriever
from retrieval.search_pipline import SearchPipeline
from reasoning.auto_rag import AutoRAG
from retrieval.traffic_synonyms import TrafficSynonymExpander
from reasoning.prompts import SYSTEM_PROMPT_FOR_VIOLATION, SYSTEM_PROMPT_FOR_REGULATION
from llama_index.core import PromptTemplate

logger = logging.getLogger(__name__)

class EnhancedAutoRAG:
    """
    Enhanced version of AutoRAG that incorporates:
    1. Synonym expansion (using TrafficSynonymExpander)
    2. Query standardization (using QuerySimplifier)
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
        self.synonym_expander = TrafficSynonymExpander()
        self.query_simplifier = QuerySimplifier(model_config)
        
        # Initialize AutoRAG
        # If search_pipeline is not provided, it will be created on-demand in AutoRAG
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
            
            
            question_type = self._classify_question_type(expanded_query)
            logger.info(f"Detected question type: {question_type}")
            
            
            if question_type == "regulation":
                
                self.auto_rag.prompt_template = PromptTemplate(template=SYSTEM_PROMPT_FOR_REGULATION)
                logger.info("Using regulation prompt template")
            else:
                
                self.auto_rag.prompt_template = PromptTemplate(template=SYSTEM_PROMPT_FOR_VIOLATION)
                logger.info("Using violation prompt template")
            
            
            response = await self.auto_rag.get_answer(expanded_query)
            
            response["query_info"] = {
                "original_query": question,
                "expanded_query": expanded_query,
                "question_type": question_type
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
        
    def _classify_question_type(self, query: str) -> str:
        """Phân loại loại câu hỏi mà không chuẩn hóa nó"""
        query_lower = query.lower()
        
        # Từ khóa gợi ý câu hỏi về quy định chung
        regulation_keywords = [
            "điều kiện", "quy định", "độ tuổi", "tuổi", "được phép", 
            "được lái", "yêu cầu", "cần gì", "sức khỏe", "bao nhiêu tuổi",
            "thủ tục", "đủ điều kiện", "cấp giấy phép", "cấp bằng"
        ]
        
        # Từ khóa gợi ý câu hỏi về vi phạm
        violation_keywords = [
            "vi phạm", "phạt", "xử phạt", "bị phạt", "phạt bao nhiêu",
            "hình phạt", "bị xử lý", "bị tước", "trừ điểm", "tịch thu"
        ]
        
        # Kiểm tra từ khóa cho quy định
        for keyword in regulation_keywords:
            if keyword in query_lower:
                logger.info(f"Query classified as 'regulation' via keyword: {keyword}")
                return "regulation"
        
        # Kiểm tra từ khóa cho vi phạm
        for keyword in violation_keywords:
            if keyword in query_lower:
                logger.info(f"Query classified as 'violation' via keyword: {keyword}")
                return "violation"
        
        # Mặc định là vi phạm nếu không phát hiện
        return "violation"