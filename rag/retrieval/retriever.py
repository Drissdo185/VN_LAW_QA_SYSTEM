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
        self.retriever = self.index.as_retriever(
                vector_store_query_mode="hybrid",
                similarity_top_k=10,
                alpha=0.5
            )
    
    def retrieve(self, query: str) -> List[NodeWithScore]:
        
            
            expanded_query = self.synonym_expander.expand_query(query)
            logger.info(f"Original query: '{query}'")
            logger.info(f"Expanded query: '{expanded_query}'")
            
         
            tokenized_query = ViTokenizer.tokenize(expanded_query.lower())
            logger.info(f"Tokenized query: '{tokenized_query}'")
            
            results = self.retriever.retrieve(tokenized_query)
            logger.info(f"Retrieved {len(results)} documents for query")
            
            
            return results
        
    
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