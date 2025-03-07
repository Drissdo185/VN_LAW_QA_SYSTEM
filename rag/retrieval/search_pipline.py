from typing import List
from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore

from config.config import ModelConfig, RetrievalConfig
from retrieval.retriever import DocumentRetriever

class SearchPipeline:
    def __init__(
        self,
        retriever: DocumentRetriever,
        model_config: ModelConfig,
        retrieval_config: RetrievalConfig
    ):
        self.retriever = retriever
        self.model_config = model_config
        self.retrieval_config = retrieval_config
        self.cross_encoder = CrossEncoder(
            model_config.cross_encoder_model,
            device=model_config.device,
            trust_remote_code=True
        )
        
    def search(self, query: str) -> List[NodeWithScore]:
        """
        Execute the full search pipeline
        
        Steps:
        1. Initial hybrid retrieval (BM25 + Dense)
        2. Cross-encoder reranking
        3. Return top 2 results
        """
        # Step 1: Initial retrieval
        initial_results = self.retriever.retrieve(query)
        
        # Step 2: Cross-encoder reranking
        reranked_results = self._rerank_results(query, initial_results)
        
        return reranked_results[:1]
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Rerank results using cross-encoder"""
        if not results:
            return []
            
        # Prepare pairs for cross-encoder
        pairs = [[query, node.text] for node in results]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Create new NodeWithScore objects with updated scores
        reranked_results = []
        for node, score in zip(results, scores):
            reranked_results.append(
                NodeWithScore(
                    node=node.node,
                    score=float(score)
                )
            )
        
        # Sort by score in descending order
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        return reranked_results