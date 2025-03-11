from typing import List
from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore
from pyvi import ViTokenizer
import logging
from config.config import ModelConfig, RetrievalConfig
from retrieval.retriever import DocumentRetriever
from retrieval.traffic_synonyms import TrafficSynonymExpander

# Setup logger
logger = logging.getLogger(__name__)

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
        self.synonym_expander = TrafficSynonymExpander()
        
        try:
            self.cross_encoder = CrossEncoder(
                model_config.cross_encoder_model,
                device=model_config.device,
                trust_remote_code=True
            )
            logger.info(f"Initialized CrossEncoder with model: {model_config.cross_encoder_model}")
        except Exception as e:
            logger.error(f"Error initializing CrossEncoder: {str(e)}")
            logger.warning("SearchPipeline will operate without reranking")
            self.cross_encoder = None
            
        logger.info("Initialized SearchPipeline")
      
    def search(self, query: str) -> List[NodeWithScore]:
        """
        Execute the full search pipeline
        
        Steps:
        1. Initial hybrid retrieval (BM25 + Dense)
        2. Metadata filtering
        3. Cross-encoder reranking
        4. Term-based ranking
        5. Return top results
        """
      
        expanded_query = self.synonym_expander.expand_query(query)
        tokenized_query = ViTokenizer.tokenize(query.lower())
        
        
        initial_results = self.retriever.retrieve(tokenized_query)
        logger.info(f"Initial retrieval returned {len(initial_results)} documents")
            
            
                
            
        logger.info("Applying metadata filtering")
        filtered_results = self._perform_metadata_filtering(initial_results, query)
            
            
            
        if self.cross_encoder:
            logger.info("Applying cross-encoder reranking")
            reranked_results = self._rerank_results(query, filtered_results)

        else:
            logger.info("Skipping cross-encoder reranking (not available)")
            reranked_results = filtered_results
            
           
            logger.info("Applying term-based ranking")
            final_results = self._rank_results(reranked_results, query)
           
            
            
        top_k = min(self.retrieval_config.similarity_top_k, len(final_results))
        logger.info(f"Returning top {top_k} results")
        return filtered_results[:10]
            
    
    def _rerank_results(self, query: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Rerank results using cross-encoder
        
        """
        if not nodes:
            logger.warning("No nodes to rerank")
            return []
            
        if not self.cross_encoder:
            logger.warning("Cross-encoder not available, skipping reranking")
            return nodes
            
        try:
           
            pairs = [(query, node.text) for node in nodes]
            
     
            scores = self.cross_encoder.predict(pairs)
            
         
            scored_nodes = []
            for i, node in enumerate(nodes):
               
                scored_node = NodeWithScore(
                    node=node.node,
                    score=float(scores[i])
                )
                scored_nodes.append(scored_node)
                
           
            return sorted(scored_nodes, key=lambda x: x.score, reverse=True)
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {str(e)}")
         
            return nodes
    
    def _rank_results(self, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
        """
        Rank results based on term overlap and relevance score
        
        Args:
            nodes: List of nodes to rank
            query: The search query
            
        Returns:
            Ranked list of nodes
        """
        if not nodes:
            logger.warning("No nodes to rank")
            return []
            
        
        query_tokens = set(ViTokenizer.tokenize(query.lower()).split())
        logger.debug(f"Query tokens: {query_tokens}")
        
        scored_nodes = []
        for node in nodes:
           
            node_tokens = set(ViTokenizer.tokenize(node.text.lower()).split())
            
           
            overlap = len(query_tokens.intersection(node_tokens))
            overlap_score = overlap / max(1, len(query_tokens))
            
            
            combined_score = (0.7 * node.score) + (0.3 * overlap_score)
            
           
            scored_node = NodeWithScore(
                node=node.node,
                score=combined_score
            )
            scored_nodes.append(scored_node)
            
            logger.debug(f"Node score: {node.score}, Overlap: {overlap}, Combined: {combined_score}")
        
        # Sort by combined score
        return sorted(scored_nodes, key=lambda x: x.score, reverse=True)

    def _perform_metadata_filtering(self, nodes, query):
        """
        Filter nodes based on vehicle category in query
        """
        query_lower = query.lower()
        
        # Detect vehicle type
        if any(keyword in query_lower for keyword in ["xe máy", "mô tô", "xe gắn máy", "xe Honda",
                                                    "xe Yamaha", "xe Suzuki", "xe Piaggio", "xe SYM",
                                                    "xe Vespa", "xe SH", "xe Air Blade", "xe Wave", "xe SH 350i",
                                                    "xe SH 150i",
                                                    "xe Dream", "xe Future", "xe Click", "xe Lead",
                                                    "xe Vision", "xe Wave Alpha", "xe Wave RSX",
                                                    "xe Wave RSX 110", "xe Wave S", "xe Wave S 110",
                                                    "xe Wave RS", "xe Wave RS 110", "xe Wave Alpha 110",
                                                    "xe Wave Alpha 100", "xe Wave RSX 100", "xe Wave S 100",
                                                    "xe Wave RS 100", "xe Wave RSX 110 Fi", "xe Wave S 110 Fi",
                                                    "xe Wave RS 110 Fi", "xe Wave Alpha 110 Fi", "xe Wave Alpha 100 Fi"]):
            vehicle_type = "mô tô, gắn máy"
        elif any(keyword in query_lower for keyword in ["ô tô", "xe hơi", "xe bốn bánh", "xe con","Mercedes",
                                                    "BMW", "Audi", "Toyota", "Honda", "Hyundai", "Kia",
                                                    "Mazda", "Ford", "Chevrolet", "Nissan", "Suzuki",
                                                    "Peugeot", "Renault", "Lexus", "Volvo", "Volkswagen",
                                                    "Mitsubishi", "Subaru", "Isuzu", "Hino"
                                        ]):
            vehicle_type = "ô tô"
        else:
            vehicle_type = None
        
        # Log detected vehicle type
        if vehicle_type:
            logger.info(f"Detected vehicle type: {vehicle_type}")
            
            # Filter by vehicle type
            filtered_nodes = [
                node for node in nodes 
                if vehicle_type.lower() in node.text.lower()
            ]
            
            if filtered_nodes:
                logger.info(f"Filtered by vehicle type: {len(filtered_nodes)} nodes")
                return filtered_nodes
        
        logger.info("No specific vehicle type detected or no matches found. Using all results.")
        return nodes