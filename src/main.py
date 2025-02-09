import os
import logging
from typing import List, Dict, Optional
import streamlit as st
import weaviate
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from weaviate.classes.init import Auth
from pyvi import ViTokenizer
from core.retrieval import RetrievalManager
from core.types import SearchResult, QuestionType
from core.question_handler import QuestionHandler
from core.postprocessing import PostProcessingPipeline
from config.domain_config import DOMAIN_CONFIGS, DomainStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DomainManager:
    """Manages domain-specific retrieval systems."""
    
    def __init__(self):
        self.domain_clients: Dict[str, RetrievalManager] = {}
        self.embed_model = self._initialize_embeddings()
        self.llm = self._initialize_llm()
        Settings.llm = self.llm
        
    def _initialize_embeddings(self) -> HuggingFaceEmbedding:
        """Initialize embedding model."""
        try:
            return HuggingFaceEmbedding(
                model_name="dangvantuan/vietnamese-document-embedding",
                max_length=256,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise

    def _initialize_llm(self) -> OpenAI:
        """Initialize language model."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return OpenAI(model="gpt-4o-mini", temperature=0.1)
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise

    def get_retrieval_manager(self, domain: str) -> RetrievalManager:
        """Get or create retrieval manager for specified domain."""
        if domain not in DOMAIN_CONFIGS:
            raise ValueError(f"Unsupported domain: {domain}")
            
        if not DOMAIN_CONFIGS[domain].is_available:
            raise ValueError(f"Domain {domain} is not yet available")
            
        if domain not in self.domain_clients:
            try:
                config = DOMAIN_CONFIGS[domain]
                
                # Initialize Weaviate client
                weaviate_client = weaviate.connect_to_weaviate_cloud(
                    cluster_url="https://v3dtdzg0skwvxinygzckra.c0.asia-southeast1.gcp.weaviate.cloud",
                    auth_credentials=Auth.api_key("Wn0Zd8LOYfOwAyPBgxtnsoIpgLGEOTM7iHb0")
                )
                # Initialize post-processing pipeline
                processors: List[BaseNodePostprocessor] = [
                    SimilarityPostprocessor(similarity_cutoff=0.5)
                ]
                pipeline = PostProcessingPipeline(processors=processors)
                
                # Create retrieval manager
                self.domain_clients[domain] = RetrievalManager(
                    weaviate_client=weaviate_client,
                    embed_model=self.embed_model,
                    post_processing_pipeline=pipeline,
                    collection_name=config.collection_name,
                    similarity_top_k=config.similarity_top_k,
                    dense_weight=config.dense_weight,
                    query_mode="hybrid"
                )
                
            except Exception as e:
                logger.error(f"Error creating retrieval manager for {domain}: {e}")
                raise
                
        return self.domain_clients[domain]

def generate_response(
    question: str,
    retrieved_docs: List,
    llm: OpenAI,
    max_retries: int = 3
) -> str:
    """Generate response using retrieved documents."""
    try:
        context = "\n".join([doc.text for doc in retrieved_docs if doc.text])
        
        prompt = f"""Dựa trên các tài liệu pháp lý sau đây, vui lòng trả lời câu hỏi.
Cung cấp câu trả lời rõ ràng, súc tích và trích dẫn các điều khoản cụ thể khi có liên quan.
Nếu không tìm thấy thông tin trong các tài liệu, hãy thông báo và đưa ra hướng dẫn chung.

Câu hỏi: {question}

Tài liệu pháp lý:
{context}

Trả lời:"""

        for attempt in range(max_retries):
            try:
                response = llm.complete(prompt)
                return response.text
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to generate response after {max_retries} attempts: {e}")
                    return "Xin lỗi, đã có lỗi xảy ra khi tạo câu trả lời. Vui lòng thử lại."
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                continue
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Xin lỗi, đã có lỗi xảy ra khi xử lý câu trả lời. Vui lòng thử lại."

def display_chat_interface(chat_history: List[ChatMessage]):
    """Display chat interface with message history."""
    for message in chat_history:
        with st.chat_message(message.role.value):
            st.write(message.content)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[ChatMessage] = []
    
    if "domain_manager" not in st.session_state:
        st.session_state.domain_manager = DomainManager()

def setup_sidebar():
    """Setup and configure sidebar elements."""
    st.sidebar.header("Cài Đặt Hệ Thống")
    st.sidebar.subheader("Lĩnh vực tra cứu")
    
    col1, col2 = st.sidebar.columns([3, 1])
    
    with col1:
        selected_domain = st.selectbox(
            "Lĩnh vực",
            list(DOMAIN_CONFIGS.keys()),
            key="domain_selector"
        )
    
    with col2:
        if DOMAIN_CONFIGS[selected_domain].status == DomainStatus.ACTIVE:
            st.success("✓ Đang hoạt động")
        else:
            st.error("⚠️ Chưa hỗ trợ")
    
    st.sidebar.write(DOMAIN_CONFIGS[selected_domain].description)
    
    if not DOMAIN_CONFIGS[selected_domain].is_available:
        st.sidebar.warning(
            "🚧 Lĩnh vực này đang được phát triển và chưa sẵn sàng để tìm kiếm. "
            "Vui lòng chọn 'Giao thông'."
        )
        st.stop()
    
    return selected_domain

def process_user_input(
    user_input: str,
    selected_domain: str,
    chat_history: List[ChatMessage],
    domain_manager: DomainManager
) -> None:
    """Process user input and generate response."""
    try:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Add to chat history
        chat_history.append(
            ChatMessage(role=MessageRole.USER, content=user_input)
        )
        
        # Create search container
        search_container = st.container()
        
        # Get retrieval manager
        retrieval_manager = domain_manager.get_retrieval_manager(selected_domain)
        
        # Process question with ViTokenizer
        processed_question = ViTokenizer.tokenize(user_input.lower())
        
        # Perform search
        results = retrieval_manager.perform_hybrid_search(
            query=processed_question,
            st_container=search_container
        )
        
        # Create search result
        search_result = SearchResult(
            documents=results,
            combined_score=sum(node.score or 0 for node in results) / len(results) if results else 0,
            question_type=QuestionType.LEGAL if results else QuestionType.STANDALONE,
            raw_results=results
        )
        
        # Generate response
        if results:
            response = generate_response(user_input, results, domain_manager.llm)
        else:
            response = (
                f"Không tìm thấy tài liệu pháp lý liên quan đến {selected_domain} "
                "để trả lời câu hỏi của bạn. Vui lòng diễn đạt lại hoặc cung cấp thêm ngữ cảnh."
            )
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)
        
        # Add to chat history
        chat_history.append(
            ChatMessage(role=MessageRole.ASSISTANT, content=response)
        )
        
    except Exception as e:
        logger.error(f"Error processing user input: {e}", exc_info=True)
        st.error(f"Lỗi khi xử lý câu hỏi: {str(e)}")

def main():
    """Main application function."""
    # Configure page
    st.set_page_config(
        page_title="Tìm kiếm tài liệu pháp lý",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Hệ thống tìm kiếm văn bản pháp lý")
    
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar and get selected domain
    selected_domain = setup_sidebar()
    
    # Display chat interface
    st.subheader("Giao diện chat")
    display_chat_interface(st.session_state.chat_history)
    
    # Handle user input
    if user_input := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
        process_user_input(
            user_input,
            selected_domain,
            st.session_state.chat_history,
            st.session_state.domain_manager
        )

if __name__ == "__main__":
    main()