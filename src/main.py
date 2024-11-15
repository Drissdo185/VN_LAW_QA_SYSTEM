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

# Define available domains
AVAILABLE_DOMAINS = {"Giao thông"}

class DomainManager:
    def __init__(self):
        self.domain_clients: Dict[str, RetrievalManager] = {}
        self.llm = None
        self.embed_model = None
        self.initialize_common_components()
        
    def initialize_common_components(self):
        """Initialize components shared across domains."""
        MODEL_NAME = "dangvantuan/vietnamese-embedding"
        self.embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME, max_length=256)
        
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        Settings.llm = self.llm
        
    def get_retrieval_manager(self, domain: str) -> RetrievalManager:
        """Get or create retrieval manager for specified domain."""
        if domain not in AVAILABLE_DOMAINS:
            raise ValueError(f"Domain {domain} is not available yet")
            
        if domain not in self.domain_clients:
            config = DOMAIN_CONFIGS[domain]
            
            # Initialize Weaviate client for domain
            weaviate_client = weaviate.connect_to_wcs(
                cluster_url=os.getenv(config.cluster_url),
                auth_credentials=Auth.api_key(os.getenv(config.api_key))
            )
            
            # Initialize post-processing pipeline
            processors: List[BaseNodePostprocessor] = [
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ]
            post_processing_pipeline = PostProcessingPipeline(processors=processors)
            
            # Create retrieval manager
            self.domain_clients[domain] = RetrievalManager(
                weaviate_client=weaviate_client,
                embed_model=self.embed_model,
                post_processing_pipeline=post_processing_pipeline,
                collection_name=config.collection_name,
                similarity_top_k=100,
                dense_weight=0.2,
                query_mode="hybrid"
            )
            
        return self.domain_clients[domain]

def generate_response(question: str, retrieved_docs: List, llm: OpenAI) -> str:
    context = "\n".join([doc.text for doc in retrieved_docs if doc.text])
    
    prompt = f"""Based on the following legal documents, please answer the question. 
    Provide a clear, concise response and cite specific articles or sections when relevant.
    If the information is not found in the documents, acknowledge that and provide general guidance.
    
    Question: {question}
    
    Legal Documents:
    {context}
    
    Response:"""
    
    try:
        response = llm.complete(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I apologize, but I encountered an error generating the response. Please try again."

def main():
    st.set_page_config(
        page_title="Legal Document Search",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Legal Document Search System")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[ChatMessage] = []
    
    if "domain_manager" not in st.session_state:
        st.session_state.domain_manager = DomainManager()
    
    with st.sidebar:
        st.header("Cài Đặt Hệ Thống")
        st.subheader("Lĩnh vực tra cứu")
        
        # Create columns for domain selection and status
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_domain = st.selectbox(
                "Lĩnh vực",
                list(DOMAIN_CONFIGS.keys()),
                key="domain_selector"
            )
        
        with col2:
            if selected_domain not in AVAILABLE_DOMAINS:
                st.error("⚠️ Not Available")
            else:
                st.success("✓ Available")
        
        # Display domain description and availability message
        st.write(DOMAIN_CONFIGS[selected_domain].description)
        
        if selected_domain not in AVAILABLE_DOMAINS:
            st.warning("🚧 This domain is currently under development and not available for search. Please select 'Giao thông' for now.")
            st.stop()  # Stop execution here if domain is not available
    
    st.subheader("Chat Interface")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message.role.value):
            st.write(message.content)
    
    user_input = st.chat_input("Enter your question here...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        st.session_state.chat_history.append(
            ChatMessage(role=MessageRole.USER, content=user_input)
        )
        
        try:
            search_container = st.container()
            
            # Get retrieval manager for selected domain
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
                response = f"I couldn't find any relevant legal documents about {selected_domain} to answer your question. Could you please rephrase or provide more context?"
            
            with st.chat_message("assistant"):
                st.write(response)
            
            st.session_state.chat_history.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=response)
            )
            
        except Exception as e:
            st.error(f"Error processing question: {e}")
            logger.error(f"Error processing question: {e}", exc_info=True)

if __name__ == "__main__":
    main()