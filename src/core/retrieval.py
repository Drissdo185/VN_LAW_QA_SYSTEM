from typing import List, Any, Optional
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core import StorageContext
from pyvi import ViTokenizer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class RetrievalManager:
    """Handles search and retrieval operations."""
    
    def __init__(
        self,
        storage,
        post_processing_pipeline,
        similarity_top_k: int = 100,
        dense_weight: float = 0.7
    ):
        self.storage = storage
        self.post_processing_pipeline = post_processing_pipeline
        self.similarity_top_k = similarity_top_k
        self.dense_weight = dense_weight

    def perform_hybrid_search(
        self,
        query: str,
        st_container: Optional[Any] = None
    ) -> List[NodeWithScore]:
        """Perform hybrid search combining dense and sparse retrieval."""
        try:
            if st_container:
                status = st_container.status("**Retrieving relevant passages...**")

            query_bundle = QueryBundle(
                query_str=query,
                custom_embedding_strs=[ViTokenizer.tokenize(query.lower())]
            )

            retriever = self.storage.get_retriever(
                similarity_top_k=self.similarity_top_k,
                alpha=self.dense_weight
            )

            results = retriever.retrieve(query_bundle)
            processed_results = self.post_processing_pipeline.process(
                results,
                query_bundle
            )

            if st_container and status:
                self._update_retrieval_status(status, processed_results, results)

            return processed_results

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise

    def _update_retrieval_status(
        self,
        status: Any,
        processed_results: List[NodeWithScore],
        raw_results: List[NodeWithScore]
    ):
        """Update UI with retrieval information."""
        self._display_results(status, processed_results)
        status.update(label="**Context Retrieval Complete**")
        status.update(state="complete")

    @staticmethod
    def _display_results(status: Any, results: List[NodeWithScore]):
        """Display search results in the UI."""
        for idx, node in enumerate(results):
            status.markdown("----")
            status.markdown(f"**Top {idx+1} Relevant Passage**")
            
            # Display content with highlighting
            content = RetrievalManager._get_highlighted_content(node)
            status.markdown(content, unsafe_allow_html=True)
            
            # Display scores and metadata
            RetrievalManager._display_node_info(status, node)

    @staticmethod
    def _get_highlighted_content(node: NodeWithScore) -> str:
        """Get highlighted content for display."""
        parent_text = node.node.get_content()
        if hasattr(node.node, 'start_char_idx') and hasattr(node.node, 'end_char_idx'):
            return (
                f"{parent_text[:node.node.start_char_idx]}"
                f":green[{parent_text[node.node.start_char_idx:node.node.end_char_idx].replace('\n', ' ')}]"
                f"{parent_text[node.node.end_char_idx:]}"
            )
        return parent_text

    @staticmethod
    def _display_node_info(status: Any, node: NodeWithScore):
        """Display node scores and metadata."""
        # Display scores
        scores = []
        for score_type in ['dense_score', 'bm25_score', 'retrieval_score']:
            if score_type in node.node.metadata:
                score = node.node.metadata[score_type]
                rank = node.node.metadata.get(f"{score_type.split('_')[0]}_rank", 'N/A')
                scores.append(f"*{score_type.replace('_', ' ').title()}*: {score:.3f}({rank})")
        
        if node.score is not None:
            scores.append(f"*Cross Encoder score*: {node.score:.3f}")
        
        if scores:
            status.markdown("**Scores**: " + ". ".join(scores))
            
        # Display metadata
        status.markdown(f"**Source**: *{node.node.metadata.get('source', 'N/A')}*")
        status.markdown(f"**Document**: *{node.node.metadata.get('filename', 'N/A')}*")