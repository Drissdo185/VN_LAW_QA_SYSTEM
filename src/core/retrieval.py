from typing import List, Any, Optional
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
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
        weaviate_client,
        embed_model,
        post_processing_pipeline,
        collection_name,
        similarity_top_k,
        dense_weight,
        query_mode
    ):
        """Initialize the RetrievalManager."""
        self.weaviate_client = weaviate_client
        self.embed_model = embed_model
        self.post_processing_pipeline = post_processing_pipeline
        self.collection_name = collection_name
        self.similarity_top_k = similarity_top_k
        self.dense_weight = dense_weight
        self.query_mode = query_mode
        
        # Initialize vector store and index
        self.vector_store = WeaviateVectorStore(
            weaviate_client=self.weaviate_client,
            index_name=self.collection_name
        )
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=self.embed_model
        )

    def perform_hybrid_search(
        self,
        query: str,
        st_container: Optional[Any] = None
    ) -> List[NodeWithScore]:
        """Perform hybrid search combining dense and sparse retrieval."""
        try:
            if st_container:
                status = st_container.status("**Retrieving relevant passages...**")

            # Prepare query bundle with Vietnamese tokenization
            # query_bundle = QueryBundle(
            #     query_str=query,
            #     custom_embedding_strs=[ViTokenizer.tokenize(query.lower())]
            # )

            # Initialize retriever with hybrid search parameters
            retriever = self.index.as_retriever(
                vector_store_query_mode = self.query_mode,
                similarity_top_k=self.similarity_top_k,
                alpha=self.dense_weight
            )

            # Perform retrieval
            results = retriever.retrieve(ViTokenizer.tokenize(query.lower()))
            
            # Apply post-processing
            processed_results = self.post_processing_pipeline.process(
                results,
                ViTokenizer.tokenize(query.lower())
            )

            # Update UI if streamlit container is provided
            if st_container and status:
                self._update_retrieval_status(status, processed_results, results)

            return processed_results

        except Exception as e:
            logger.error("Error in hybrid search: %s", str(e))
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
            start_part = parent_text[:node.node.start_char_idx]
            highlight_part = parent_text[node.node.start_char_idx:node.node.end_char_idx].replace('\n', ' ')
            end_part = parent_text[node.node.end_char_idx:]
            return f"{start_part}:green[{highlight_part}]{end_part}"
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
            