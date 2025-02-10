from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from config.config import RetrievalConfig


class DocumentRetriever:
    def __init__(
        self,
        index: VectorStoreIndex,
        config: RetrievalConfig
    ):
        self.index = index
        self.config = config
        self.retriever = self._setup_retriever()
        
    def _setup_retriever(self):
        """Setup the retriever with configured parameters"""
        return self.index.as_retriever(
            vector_store_query_mode=self.config.vector_store_query_mode,
            similarity_top_k=self.config.similarity_top_k,
            alpha=self.config.alpha
        )
    
    def retrieve(self, query: str) -> List[NodeWithScore]:
        """
        Retrieve relevant documents for the given query
        
        Args:
            query: The search query
            
        Returns:
            List of retrieved documents with relevance scores
        """
        return self.retriever.retrieve(query)
    
    def get_formatted_context(self, nodes: List[NodeWithScore]) -> str:
        """
        Format retrieved nodes into a context string
        
        Args:
            nodes: List of retrieved nodes
            
        Returns:
            Formatted context string
        """
        return "\n\n".join([node.text for node in nodes])