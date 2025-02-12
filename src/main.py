import streamlit as st
import logging
from typing import List
import asyncio
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config.config import ModelConfig, RetrievalConfig, load_and_validate_configs
from config.domain_router import DomainRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DomainManager:
    def __init__(self):
        self.llm = None
        self.embed_model = None
        self.domain_router = None
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize common components and domain router"""
        try:
            # Initialize embedding model
            self.embed_model = HuggingFaceEmbedding(
                model_name="dangvantuan/vietnamese-document-embedding",
                max_length=256,
                trust_remote_code=True
            )
            
            # Load and validate configs
            model_config, retrieval_config = load_and_validate_configs()
            
            # Initialize LLM
            self.llm = OpenAI(model=model_config.llm_model, temperature=0.1)
            Settings.llm = self.llm
            
            # Initialize domain router
            self.domain_router = DomainRouter(
                llm=self.llm,
                model_config=model_config,
                retrieval_config=retrieval_config
            )
            
            logger.info("Successfully initialized all components")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}", exc_info=True)
            raise
    
    async def process_question(self, question: str) -> str:
        """Process user question through domain routing and search pipeline"""
        try:
            if not self.domain_router:
                raise ValueError("Domain router not initialized")
                
            # Get results and domain classification
            results, domain = await self.domain_router.route_and_search(question)
            
            if not results or not domain:
                return (
                    "Xin lỗi, tôi không thể tìm thấy thông tin phù hợp để trả lời câu hỏi của bạn. "
                    "Vui lòng thử diễn đạt lại câu hỏi hoặc cung cấp thêm chi tiết."
                )
            
            # Generate response from results
            response = await self.generate_response(question, results, domain)
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            return "Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại sau."

    async def generate_response(self, question: str, results: List, domain: str) -> str:
        """Generate response based on retrieved documents"""
        context = "\n\n".join([doc.text for doc in results if doc.text])
        
        prompt = f"""Dựa trên các tài liệu pháp lý về {domain} sau đây, vui lòng trả lời câu hỏi.
Cung cấp câu trả lời rõ ràng, súc tích và trích dẫn các điều khoản cụ thể khi có liên quan.

Câu hỏi: {question}

Tài liệu pháp lý:
{context}

Trả lời:"""
        
        try:
            response = await self.llm.acomplete(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Xin lỗi, đã có lỗi xảy ra khi tạo câu trả lời. Vui lòng thử lại."

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[ChatMessage] = []
    
    if "domain_manager" not in st.session_state:
        st.session_state.domain_manager = DomainManager()

def display_chat_history():
    """Display chat history"""
    for message in st.session_state.chat_history:
        with st.chat_message(message.role.value):
            st.write(message.content)

async def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Hệ thống Tra cứu Pháp lý",
        page_icon="⚖️",
        layout="wide"
    )
    
    # Page header
    st.title("⚖️ Hệ thống Tra cứu Văn bản Pháp lý")
    st.write("Hỗ trợ tra cứu về: Giao thông 🚗 | Chứng khoán 📈 | Lao động 👷")
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    user_input = st.chat_input("Nhập câu hỏi của bạn...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Add user message to chat history
        st.session_state.chat_history.append(
            ChatMessage(role=MessageRole.USER, content=user_input)
        )
        
        try:
            # Process question
            with st.spinner("Đang xử lý câu hỏi..."):
                response = await st.session_state.domain_manager.process_question(user_input)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=response)
            )
            
        except Exception as e:
            st.error("❌ Đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại.")
            logger.error(f"Error in main loop: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())