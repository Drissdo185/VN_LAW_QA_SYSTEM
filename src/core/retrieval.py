from typing import List, Any, Optional
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.llms.openai import OpenAI
from pyvi import ViTokenizer
from logging import getLogger
from core.types import ReasoningResult

logger = getLogger(__name__)

class RetrievalManager:
    """Handles search and retrieval operations with autonomous reasoning."""
    
    DEFAULT_SYSTEM_PROMPT = PromptTemplate(
        template=(
            "Dựa trên tài liệu đã trích xuất, hãy phân tích và trả lời câu hỏi.\n"
            "Câu hỏi: {question}\n"
            "Tài liệu: {context}\n"
            "Hãy suy nghĩ từng bước:\n"
            "1. Phân tích xem thông tin có đủ và liên quan không?\n"
            "2. Nếu chưa đủ, hãy đưa ra truy vấn mới để tìm thêm thông tin\n"
            "3. Nếu đã đủ, đưa ra câu trả lời cuối cùng\n\n"
            "Hãy trả lời theo định dạng sau:\n"
            "Phân tích: <phân tích thông tin hiện có>\n"
            "Quyết định: [Cần thêm thông tin/Đã đủ thông tin]\n"
            "Truy vấn tiếp theo: <truy vấn mới> (nếu cần)\n"
            "Câu trả lời cuối cùng: <câu trả lời> (nếu đã đủ thông tin)\n"
        )
    )
    
    def __init__(
        self,
        weaviate_client,
        embed_model,
        post_processing_pipeline,
        collection_name: "ND168",
        similarity_top_k: int = 5,
        dense_weight: float = 0.5,
        query_mode: str = "hybrid",
        max_reasoning_steps: int = 3,
        system_prompt: Optional[PromptTemplate] = None
    ):
        """Initialize the RetrievalManager."""
        try:
            
            self.max_reasoning_steps = max_reasoning_steps
            self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
            self.post_processing_pipeline = post_processing_pipeline
            
            # Initialize vector store and index
            self.vector_store = WeaviateVectorStore(
                weaviate_client=weaviate_client,
                index_name=collection_name
            )
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=embed_model
            )
            
            # Initialize retriever
            self.retriever = self.index.as_retriever(
                vector_store_query_mode=query_mode,
                similarity_top_k=similarity_top_k,
                alpha=dense_weight
            )
        except Exception as e:
            logger.error(f"Error initializing RetrievalManager: {e}")
            raise

    async def perform_autonomous_reasoning(
        self,
        query: str,
        st_container: Optional[Any] = None
    ) -> ReasoningResult:
        """Perform autonomous reasoning with iterative retrieval."""
        try:
            if st_container:
                status = st_container.status("**Starting autonomous reasoning...**")
            
            current_query = query
            step_count = 0
            last_result = None
            
            while step_count < self.max_reasoning_steps:
                # Retrieve and process documents
                retrieved_docs = self.retriever.retrieve(
                    ViTokenizer.tokenize(current_query.lower())
                )
                docs_text = "\n".join([doc.text for doc in retrieved_docs])
                
                # Generate reasoning
                response = await self.llm.acomplete(
                    self.system_prompt.format(
                        question=current_query,
                        context=docs_text
                    )
                )
                
                result = self._parse_reasoning_response(response.text.strip())
                last_result = result
                
                if st_container and status:
                    self._update_reasoning_status(status, step_count, result)
                
                if result.decision == "Đã đủ thông tin":
                    return result
                
                current_query = result.next_query
                step_count += 1
            
            logger.warning(
                f"Reached maximum reasoning steps ({self.max_reasoning_steps}) "
                "without definitive answer"
            )
            return last_result

        except Exception as e:
            logger.error(f"Error in autonomous reasoning: {e}")
            raise

    def _parse_reasoning_response(self, response: str) -> ReasoningResult:
        """Parse the LLM response into a structured format."""
        try:
            result = ReasoningResult(
                analysis="",
                decision="",
                next_query=None,
                final_answer=None
            )
            
            current_section = None
            for line in response.split('\n'):
                if not line.strip():
                    continue
                    
                if ":" not in line:
                    if current_section == "analysis":
                        result.analysis += " " + line.strip()
                    elif current_section == "final_answer":
                        result.final_answer += " " + line.strip()
                    continue
                    
                key, value = [x.strip() for x in line.split(":", 1)]
                if key == "Phân tích":
                    current_section = "analysis"
                    result.analysis = value
                elif key == "Quyết định":
                    current_section = "decision"
                    result.decision = value
                elif key == "Truy vấn tiếp theo":
                    current_section = "next_query"
                    result.next_query = value
                elif key == "Câu trả lời cuối cùng":
                    current_section = "final_answer"
                    result.final_answer = value
            
            return result
        except Exception as e:
            logger.error(f"Error parsing reasoning response: {e}")
            raise

    def perform_hybrid_search(
    self,
    query: str,
    st_container: Optional[Any] = None
    ) -> List[NodeWithScore]:
        """Perform hybrid search combining dense and sparse retrieval."""
        try:
            # Query is already processed at this point
            results = self.retriever.retrieve(query)
            
            # Apply post-processing if available
            if self.post_processing_pipeline:
                results = self.post_processing_pipeline.process(
                    results,
                    query
                )
            
            # Update UI if container provided
            if st_container:
                self._display_search_results(st_container, results)
            
            return results
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise

    def _update_reasoning_status(
        self,
        status: Any,
        step_count: int,
        result: ReasoningResult
    ):
        """Update UI with reasoning progress."""
        try:
            status.markdown(f"**Step {step_count + 1}**")
            if result.analysis:
                status.markdown(f"*Analysis:* {result.analysis}")
            if result.decision:
                status.markdown(f"*Decision:* {result.decision}")
            if result.next_query:
                status.markdown(f"*Next Query:* {result.next_query}")
            if result.final_answer:
                status.markdown(f"*Final Answer:* {result.final_answer}")
                
            status.markdown("---")
        except Exception as e:
            logger.error(f"Error updating reasoning status: {e}")

    @staticmethod
    def _display_search_results(status: Any, results: List[NodeWithScore]):
        """Display search results in the UI."""
        try:
            for idx, node in enumerate(results, 1):
                status.markdown("---")
                status.markdown(f"**Relevant Passage {idx}**")
                
                # Display content
                content = node.node.get_content()
                status.markdown(content)
                
                # Display score if available
                if node.score is not None:
                    status.markdown(f"*Relevance Score:* {node.score:.3f}")
                
                # Display metadata if available
                if hasattr(node.node, 'metadata') and node.node.metadata:
                    metadata_str = "\n".join(
                        f"- {key}: {value}"
                        for key, value in node.node.metadata.items()
                        if key not in ['dense_score', 'bm25_score', 'retrieval_score']
                    )
                    if metadata_str:
                        status.markdown("*Metadata:*")
                        status.markdown(metadata_str)
                
                # Display scores from metadata
                scores = []
                for score_type in ['dense_score', 'bm25_score', 'retrieval_score']:
                    if score_type in node.node.metadata:
                        score = node.node.metadata[score_type]
                        rank = node.node.metadata.get(
                            f"{score_type.split('_')[0]}_rank",
                            'N/A'
                        )
                        scores.append(
                            f"*{score_type.replace('_', ' ').title()}*: "
                            f"{score:.3f} (Rank: {rank})"
                        )
                
                if scores:
                    status.markdown("*Component Scores:*")
                    status.markdown("\n".join(scores))
        except Exception as e:
            logger.error(f"Error displaying search results: {e}")

    def get_retriever(self):
        """Get the configured retriever instance."""
        return self.retriever

    def get_embedding_model(self):
        """Get the configured embedding model."""
        return self.embed_model

    def get_llm(self):
        """Get the configured LLM instance."""
        return self.llm