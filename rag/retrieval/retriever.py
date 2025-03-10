from typing import List
import logging
from pyvi import ViTokenizer
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from config.config import RetrievalConfig
from retrieval.traffic_synonyms import TrafficSynonymExpander

logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(
        self,
        index: VectorStoreIndex,
        config: RetrievalConfig
    ):
        self.index = index
        self.config = config
        self._setup_retriever()
        self.synonym_expander = TrafficSynonymExpander()
        
    def _setup_retriever(self):
        """Setup the retriever with configured parameters"""
        try:
            self.retriever = self.index.as_retriever(
                vector_store_query_mode=self.config.vector_store_query_mode,
                similarity_top_k=self.config.similarity_top_k,
                alpha=self.config.alpha
            )
            logger.info(f"Retriever setup complete with similarity_top_k={self.config.similarity_top_k}")
        except Exception as e:
            logger.error(f"Error setting up retriever: {str(e)}")
            raise
    
    def retrieve(self, query: str) -> List[NodeWithScore]:
        """
        Retrieve relevant documents for the given query
        
        Args:
            query: The search query
            
        Returns:
            List of retrieved documents with relevance scores
        """
        try:
            
            expanded_query = self.synonym_expander.expand_query(query)
            logger.info(f"Original query: '{query}'")
            logger.info(f"Expanded query: '{expanded_query}'")
            
         
            tokenized_query = ViTokenizer.tokenize(expanded_query.lower())
            logger.info(f"Tokenized query: '{tokenized_query}'")
            
            results = self.retriever.retrieve(tokenized_query)
            logger.info(f"Retrieved {len(results)} documents for query")
            
            if results:
                logger.info(f"Top result score: {results[0].score}")
                logger.info(f"Score range: {results[-1].score} to {results[0].score}")
            
            return results
        except Exception as e:
            if "closed" in str(e).lower():
                logger.warning("Client connection closed, attempting to reinitialize retriever")
                self._setup_retriever()
                # Dùng lại mở rộng từ đồng nghĩa nếu kết nối lại
                expanded_query = self.synonym_expander.expand_query(query)
                tokenized_query = ViTokenizer.tokenize(expanded_query.lower())
                return self.retriever.retrieve(tokenized_query)
            logger.error(f"Error during retrieval: {str(e)}")
            raise
    
    def get_formatted_context(self, nodes: List[NodeWithScore]) -> str:
        """
        Format retrieved nodes into a context string
        
        Args:
            nodes: List of retrieved nodes
            
        Returns:
            Formatted context string
        """
        if not nodes:
            logger.warning("No nodes provided to format as context")
            return ""
            
        formatted_context = "\n\n".join([
            f"Document {i+1}:\n{self._get_original_text(node)}" 
            for i, node in enumerate(nodes)
        ])
        
        logger.debug(f"Formatted context with {len(nodes)} documents")
        return formatted_context
        
    def _get_original_text(self, node: NodeWithScore) -> str:
        """Get original text from node, handling metadata if available"""
        if hasattr(node.node, 'metadata') and 'original_text' in node.node.metadata:
            return node.node.metadata['original_text']
        return node.text