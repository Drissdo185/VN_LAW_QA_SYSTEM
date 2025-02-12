from typing import Optional, Dict, Tuple, List
import logging
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import NodeWithScore

from config.config import *
from retrieval.vector_store import VectorStoreManager
from retrieval.search_pipline import SearchPipeline
from retrieval.retriever import DocumentRetriever

logger = logging.getLogger(__name__)

class DomainRouter:
    def __init__(
        self,
        llm: OpenAI,
        model_config: ModelConfig,
        retrieval_config: RetrievalConfig
    ):
        self.llm = llm
        self.model_config = model_config
        self.retrieval_config = retrieval_config
        self.domain_pipelines: Dict[str, SearchPipeline] = {}
        self.available_domains = get_available_domains()
        logger.info(f"Available domains: {list(self.available_domains.keys())}")
        self.initialize_domain_pipelines()

    def initialize_domain_pipelines(self):
        """Initialize search pipelines for available domains"""
        logger.info("Starting pipeline initialization")
        for domain, config in self.available_domains.items():
            try:
                logger.info(f"Initializing pipeline for domain: {domain}")
                
                # Initialize vector store
                vector_store = VectorStoreManager(config, self.model_config)
                vector_store.initialize()
                logger.info(f"Vector store initialized for {domain}")
                
                # Initialize retriever
                index = vector_store.get_index()
                if not index:
                    logger.error(f"Failed to get index for domain {domain}")
                    continue
                    
                retriever = DocumentRetriever(
                    index,
                    self.retrieval_config
                )
                logger.info(f"Retriever initialized for {domain}")
                
                # Initialize search pipeline
                self.domain_pipelines[domain] = SearchPipeline(
                    retriever=retriever,
                    model_config=self.model_config,
                    retrieval_config=self.retrieval_config
                )
                logger.info(f"Successfully initialized pipeline for {domain}")
                
            except Exception as e:
                logger.error(f"Error initializing pipeline for domain {domain}: {e}", exc_info=True)
                continue

    async def route_and_search(self, question: str) -> Tuple[Optional[List[NodeWithScore]], Optional[str]]:
        """Route question to appropriate domain and perform search"""
        logger.info(f"Processing question: {question}")
        
        # Classify domain
        domain = await self.classify_domain(question)
        logger.info(f"Classified domain: {domain}")
        
        if not domain or domain not in self.available_domains:
            logger.warning(f"Domain not available: {domain}")
            return None, None
            
        # Get search pipeline for domain
        pipeline = self.domain_pipelines.get(domain)
        if not pipeline:
            logger.error(f"No pipeline found for domain: {domain}")
            return None, None
            
        # Perform search
        try:
            logger.info(f"Performing search for domain: {domain}")
            results = pipeline.search(question)
            if results:
                logger.info(f"Found {len(results)} results")
                # Log first result score for debugging
                if results[0].score is not None:
                    logger.info(f"Top result score: {results[0].score}")
            else:
                logger.warning("No results found")
            return results, domain
        except Exception as e:
            logger.error(f"Error performing search: {e}", exc_info=True)
            return None, None

    async def classify_domain(self, question: str) -> Optional[str]:
        """Classify question domain using LLM"""
        domains_str = "giao_thong" if len(self.available_domains) == 1 else ", ".join(self.available_domains.keys())
        
        prompt = f"""Xác định lĩnh vực của câu hỏi sau:

Câu hỏi: {question}

Chỉ trả về một trong các giá trị: {domains_str}
- giao_thong: Nếu liên quan đến luật giao thông, xe cộ
{"" if len(self.available_domains) == 1 else "- chung_khoan: Nếu liên quan đến chứng khoán, thị trường vốn"}
{"" if len(self.available_domains) == 1 else "- lao_dong: Nếu liên quan đến lao động, việc làm"}

Trả lời:"""

        try:
            logger.info("Sending classification prompt to LLM")
            response = await self.llm.acomplete(prompt)
            domain = response.text.strip().lower()
            logger.info(f"LLM classified domain as: {domain}")
            return domain if domain in self.available_domains else None
        except Exception as e:
            logger.error(f"Error classifying domain: {e}", exc_info=True)
            return None