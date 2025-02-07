import streamlit as st
import weaviate
from typing import List, Dict
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from core.retrieval import RetrievalManager
from core.types import SearchResult, QuestionType
from core.question_handler import QuestionHandler
from core.postprocessing import PostProcessingPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from weaviate.classes.init import Auth
from config.domain_config import DOMAIN_CONFIGS
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
AVAILABLE_DOMAINS = {"Giao thông"}

class DomainManager:
    def __init__(self):
        self.domain_clients: Dict[str, RetrievalManager] = {}
        self.llm = None
        self.embed_model = None
        self.initialize_common_components()
        
    def initialize_common_components(self):
        """Khởi tạo các thành phần dùng chung."""
        MODEL_NAME = "dangvantuan/vietnamese-embedding"
        self.embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME, max_length=256)
        
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        Settings.llm = self.llm
        
    def get_retrieval_manager(self, domain: str) -> RetrievalManager:
        """Lấy hoặc tạo retrieval manager cho lĩnh vực được chỉ định."""
        if domain not in AVAILABLE_DOMAINS:
            raise ValueError(f"Lĩnh vực {domain} hiện chưa được hỗ trợ")
            
        if domain not in self.domain_clients:
            config = DOMAIN_CONFIGS[domain]
            
            
            weaviate_client = weaviate.connect_to_wcs(
                cluster_url=os.getenv(config.cluster_url),
                auth_credentials=Auth.api_key(os.getenv(config.api_key))
            )
            
            
            processors: List[BaseNodePostprocessor] = [
                SimilarityPostprocessor(similarity_cutoff=0.5)
            ]
            post_processing_pipeline = PostProcessingPipeline(processors=processors)
            
            
            self.domain_clients[domain] = RetrievalManager(
                weaviate_client=weaviate_client,
                embed_model=self.embed_model,
                post_processing_pipeline=post_processing_pipeline,
                collection_name=config.collection_name,
                similarity_top_k=10,
                dense_weight=0.3,
                query_mode="hybrid"
            )
            
        return self.domain_clients[domain]

def generate_response(question: str, retrieved_docs: List, llm: OpenAI) -> str:
    context = "\n".join([doc.text for doc in retrieved_docs if doc.text])
    
    prompt = f"""Dựa trên các tài liệu pháp lý sau đây, vui lòng trả lời câu hỏi.
Cung cấp câu trả lời rõ ràng, súc tích và trích dẫn các điều khoản cụ thể khi có liên quan.
Nếu không tìm thấy thông tin trong các tài liệu, hãy thông báo và đưa ra hướng dẫn chung.
    
    Câu hỏi: {question}
    
    Tài liệu pháp lý:
    {context}
    
    Trả lời:"""
    
    try:
        response = llm.complete(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Lỗi khi tạo câu trả lời: {e}")
        return "Xin lỗi, đã có lỗi xảy ra khi tạo câu trả lời. Vui lòng thử lại."

def main():
    st.set_page_config(
        page_title="Tìm kiếm tài liệu pháp lý",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Hệ thống tìm kiếm văn bản pháp lý")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[ChatMessage] = []
    
    if "domain_manager" not in st.session_state:
        st.session_state.domain_manager = DomainManager()
    
    with st.sidebar:
        st.header("Cài Đặt Hệ Thống")
        st.subheader("Lĩnh vực tra cứu")
        
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_domain = st.selectbox(
                "Lĩnh vực",
                list(DOMAIN_CONFIGS.keys()),
                key="domain_selector"
            )
        
        with col2:
            if selected_domain not in AVAILABLE_DOMAINS:
                st.error("⚠️ Chưa hỗ trợ")
            else:
                st.success("✓ Đang hoạt động")
        
        
        st.write(DOMAIN_CONFIGS[selected_domain].description)
        
        if selected_domain not in AVAILABLE_DOMAINS:
            st.warning("🚧 Lĩnh vực này đang được phát triển và chưa sẵn sàng để tìm kiếm. Vui lòng chọn 'Giao thông'.")
            st.stop()
    
    st.subheader("Giao diện chat")
    
    
    for message in st.session_state.chat_history:
        with st.chat_message(message.role.value):
            st.write(message.content)
    
    user_input = st.chat_input("Nhập câu hỏi của bạn tại đây...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        st.session_state.chat_history.append(
            ChatMessage(role=MessageRole.USER, content=user_input)
        )
        
        try:
            search_container = st.container()
            
            
            retrieval_manager = st.session_state.domain_manager.get_retrieval_manager(selected_domain)
            
            processed_question = QuestionHandler.process_standalone_question(
                user_input,
                st.session_state.chat_history
            )
            
            results = retrieval_manager.perform_hybrid_search(
                processed_question,
                search_container
            )
            
            search_result = SearchResult(
                documents=results,
                combined_score=sum(node.score or 0 for node in results) / len(results) if results else 0,
                question_type=QuestionType.LEGAL if results else QuestionType.STANDALONE,
                raw_results=results
            )
            
            if results:
                response = generate_response(user_input, results, st.session_state.domain_manager.llm)
            else:
                response = f"Không tìm thấy tài liệu pháp lý liên quan đến {selected_domain} để trả lời câu hỏi của bạn. Vui lòng diễn đạt lại hoặc cung cấp thêm ngữ cảnh."
            
            with st.chat_message("assistant"):
                st.write(response)
            
            st.session_state.chat_history.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=response)
            )
            
        except Exception as e:
            st.error(f"Lỗi khi xử lý câu hỏi: {e}")
            logger.error(f"Lỗi khi xử lý câu hỏi: {e}", exc_info=True)

if __name__ == "__main__":
    main()