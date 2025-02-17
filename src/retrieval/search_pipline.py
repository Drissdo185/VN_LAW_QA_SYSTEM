from typing import List, Dict, Any
import numpy as np
from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore
from difflib import SequenceMatcher

from config.config import ModelConfig, RetrievalConfig, Domain
from retrieval.retriever import DocumentRetriever

class SearchPipeline:
    def __init__(
        self,
        retriever: DocumentRetriever,
        model_config: ModelConfig,
        retrieval_config: RetrievalConfig,
        domain: Domain
    ):
        self.retriever = retriever
        self.model_config = model_config
        self.retrieval_config = retrieval_config
        self.domain = domain
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
        3. Remove duplicate parent documents
        4. Filter by minimum similarity score
        """
        # Step 1: Initial retrieval
        initial_results = self.retriever.retrieve(query)
        
        # Step 2: Cross-encoder reranking
        reranked_results = self._rerank_results(query, initial_results)
        
        # # Step 3: Remove duplicates
        # deduplicated_results = self._remove_duplicates(reranked_results)
        
        # # Step 4: Filter by score
        # filtered_results = self._filter_by_score(deduplicated_results)
        
        return reranked_results[:10]
    
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
    
    def _remove_duplicates(self, results: List[NodeWithScore]) -> List[NodeWithScore]:
        """Remove duplicate or highly similar parent documents"""
        if not results:
            return []
            
        unique_results = []
        seen_texts = []
        
        for result in results:
            is_duplicate = False
            current_text = result.text
            
            # Check similarity with previously seen texts
            for seen_text in seen_texts:
                similarity = SequenceMatcher(None, current_text, seen_text).ratio()
                if similarity > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_texts.append(current_text)
                
        return unique_results
    
    def _filter_by_score(self, results: List[NodeWithScore]) -> List[NodeWithScore]:
        """Filter results by minimum similarity score"""
        return [
            result for result in results 
            if result.score >= 0.7
        ]