from typing import List
from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore
from pyvi import ViTokenizer
import logging
from config.config import ModelConfig, RetrievalConfig
from retrieval.retriever import DocumentRetriever

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
        # Tokenize query for Vietnamese
        tokenized_query = ViTokenizer.tokenize(query.lower())
        logger.info(f"Processing search query: '{query}'")
        logger.info(f"Tokenized query: '{tokenized_query}'")
        
        try:
            # Step 1: Initial retrieval
            logger.info(f"Performing initial retrieval")
            if self.retriever:
                initial_results = self.retriever.retrieve(tokenized_query)
                logger.info(f"Initial retrieval returned {len(initial_results)} documents")
            else:
                logger.error("Retriever not properly initialized")
                return []
            
            if not initial_results:
                logger.warning("Initial retrieval returned no results")
                return []
                
            # Step 2: Apply metadata filtering
            logger.info("Applying metadata filtering")
            filtered_results = self._perform_metadata_filtering(initial_results, query)
            logger.info(f"Metadata filtering returned {len(filtered_results)} documents")
            
            # Step 3: Apply cross-encoder reranking
            if self.cross_encoder:
                logger.info("Applying cross-encoder reranking")
                reranked_results = self._rerank_results(query, filtered_results)
                logger.info(f"Reranking returned {len(reranked_results)} documents")
            else:
                logger.info("Skipping cross-encoder reranking (not available)")
                reranked_results = filtered_results
            
            # Step 4: Apply term-based ranking
            logger.info("Applying term-based ranking")
            final_results = self._rank_results(reranked_results, query)
            logger.info(f"Term-based ranking returned {len(final_results)} documents")
            
            # Log top result for debugging
            if final_results:
                logger.info(f"Top result score: {final_results[0].score}")
                logger.info(f"Top result snippet: {final_results[0].text[:100]}...")
            
            # Return top results
            top_k = min(self.retrieval_config.similarity_top_k, len(final_results))
            logger.info(f"Returning top {top_k} results")
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in search pipeline: {str(e)}")
            return []
    
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
            # Prepare document-query pairs for cross-encoder
            pairs = [(query, node.text) for node in nodes]
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Update node scores
            scored_nodes = []
            for i, node in enumerate(nodes):
                # Create a copy to avoid modifying the original
                scored_node = NodeWithScore(
                    node=node.node,
                    score=float(scores[i])
                )
                scored_nodes.append(scored_node)
                
            # Sort by score (descending)
            return sorted(scored_nodes, key=lambda x: x.score, reverse=True)
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {str(e)}")
            # Return original nodes if reranking fails
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
            
        # Tokenize query using ViTokenizer for Vietnamese
        query_tokens = set(ViTokenizer.tokenize(query.lower()).split())
        logger.debug(f"Query tokens: {query_tokens}")
        
        scored_nodes = []
        for node in nodes:
            # Tokenize node text consistently with query
            node_tokens = set(ViTokenizer.tokenize(node.text.lower()).split())
            
            # Calculate term overlap
            overlap = len(query_tokens.intersection(node_tokens))
            overlap_score = overlap / max(1, len(query_tokens))
            
            # Calculate combined score (weighted average of similarity and term overlap)
            combined_score = (0.7 * node.score) + (0.3 * overlap_score)
            
            # Create a new node with the combined score
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
        Filter nodes based on metadata matching query
        """
        query_lower = query.lower()
        
        # Detect violation types in query with expanded keywords
        violation_types = []
        
        # Child safety violations
        if any(keyword in query_lower for keyword in ["trẻ em", "trẻ nhỏ", "1,35 mét", "mầm non", "học sinh", 
                                                    "dưới 10 tuổi", "chiều cao", "ghế trẻ em"]):
            violation_types.append("trẻ_em")
        
        # Speed violations
        if any(keyword in query_lower for keyword in ["tốc độ", "km/h", "chạy quá", "vượt quá tốc độ", 
                                                    "tốc độ tối đa", "tốc độ tối thiểu", "chạy nhanh"]):
            violation_types.append("tốc_độ")
            
        # Alcohol violations
        if any(keyword in query_lower for keyword in ["nồng độ cồn", "cồn", "rượu", "bia", "miligam", 
                                                    "nồng độ trong máu", "hơi thở", "say xỉn"]):
            violation_types.append("nồng_độ_cồn")
        
        # Drug violations
        if any(keyword in query_lower for keyword in ["ma túy", "chất kích thích", "chất gây nghiện", 
                                                    "chất ma túy", "dương tính"]):
            violation_types.append("ma_túy")
        
        # Parking violations
        if any(keyword in query_lower for keyword in ["đỗ xe", "đậu xe", "dừng xe", "đỗ trái phép", 
                                                    "đỗ sai quy định", "vạch kẻ đường", "lề đường", "vỉa hè"]):
            violation_types.append("đỗ_dừng_xe")
        
        # Vehicle-specific queries
        if any(keyword in query_lower for keyword in ["xe máy", "mô tô", "xe gắn máy"]):
            vehicle_type = "xe máy"
        elif any(keyword in query_lower for keyword in ["ô tô", "xe hơi", "xe bốn bánh"]):
            vehicle_type = "ô tô"
        else:
            vehicle_type = None
        
        # Remove duplicates
        violation_types = list(set(violation_types))
        
        logger.info(f"Detected violation types: {violation_types}")
        if vehicle_type:
            logger.info(f"Detected vehicle type: {vehicle_type}")
        
        # First try filtering by both violation type and vehicle type if both exist
        if violation_types and vehicle_type:
            filtered_nodes = [
                node for node in nodes 
                if (hasattr(node.node, "metadata") and 
                    node.node.metadata.get("violation_type") in violation_types and
                    vehicle_type.lower() in node.text.lower())
            ]
            
            if filtered_nodes:
                logger.info(f"Filtered by both violation and vehicle types: {len(filtered_nodes)} nodes")
                return filtered_nodes
                
        # Otherwise try just violation type
        if violation_types:
            filtered_nodes = [
                node for node in nodes 
                if (hasattr(node.node, "metadata") and 
                    node.node.metadata.get("violation_type") in violation_types)
            ]
            
            if filtered_nodes:
                logger.info(f"Filtered by violation type: {len(filtered_nodes)} nodes")
                return filtered_nodes
                
        # Otherwise try just vehicle type
        if vehicle_type:
            filtered_nodes = [
                node for node in nodes 
                if vehicle_type.lower() in node.text.lower()
            ]
            
            if filtered_nodes:
                logger.info(f"Filtered by vehicle type: {len(filtered_nodes)} nodes")
                return filtered_nodes
        
        logger.info("No specific filtering applied or no matches found. Using all results.")
        return nodes