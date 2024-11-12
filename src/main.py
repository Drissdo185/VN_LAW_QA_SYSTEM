import streamlit as st
import weaviate
from typing import List
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.postprocessor import SimilarityPostprocessor
from core.retrieval import RetrievalManager
from core.types import SearchResult, QuestionType
from core.question_handler import QuestionHandler
from core.postprocessing import PostProcessingPipeline
from core.formatter import ResultFormatter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from weaviate.classes.init import Auth
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_components():
    """Initialize all required components."""
    # Initialize Weaviate client
    cluster_url = ""
    api_key = ""
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=cluster_url,
        auth_credentials=Auth.api_key(api_key)
    )
    
    MODEL_NAME = "qducnguyen/vietnamese-bi-encoder"
    embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME,
                                   max_length=256)
    
    # Initialize post-processing pipeline
    processors: List[BaseNodePostprocessor] = [
        SimilarityPostprocessor(similarity_cutoff=0.7)
    ]
    post_processing_pipeline = PostProcessingPipeline(processors=processors)
    
    # Initialize retrieval manager
    collection_name = "Vn_law"
    retrieval_manager = RetrievalManager(
        weaviate_client=weaviate_client,
        embed_model=embed_model,
        post_processing_pipeline=post_processing_pipeline,
        collection_name=collection_name,
        similarity_top_k=100,
        dense_weight=0.2
    )
    
    return retrieval_manager

def main():
    st.set_page_config(
        page_title="Legal Document Search",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Legal Document Search System")
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[ChatMessage] = []
    
    # Initialize components
    try:
        retrieval_manager = initialize_components()
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        show_raw_results = st.checkbox("Show Raw Results", value=False)
        clear_history = st.button("Clear Chat History")
        
        if clear_history:
            st.session_state.chat_history = []
            st.experimental_rerun()
    
    # Main chat interface
    st.subheader("Chat Interface")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message.role.value):
            st.write(message.content)
    
    # User input
    user_input = st.chat_input("Enter your question here...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Add user message to history
        st.session_state.chat_history.append(
            ChatMessage(role=MessageRole.USER, content=user_input)
        )
        
        # Process question
        processed_question = QuestionHandler.process_standalone_question(
            user_input,
            st.session_state.chat_history
        )
        
        try:
            # Create container for search results
            search_container = st.container()
            
            # Perform search
            results = retrieval_manager.perform_hybrid_search(
                processed_question,
                search_container
            )
            
            # Create search result object
            search_result = SearchResult(
                documents=results,
                combined_score=sum(node.score or 0 for node in results) / len(results) if results else 0,
                question_type=QuestionType.LEGAL if results else QuestionType.STANDALONE,
                raw_results=results if show_raw_results else None
            )
            
            # Format results
            formatted_results = ResultFormatter.format_search_results(
                search_result,
                include_raw_results=show_raw_results
            )
            
            # Display assistant response
            with st.chat_message("assistant"):
                if formatted_results["question_type"] == QuestionType.LEGAL.value:
                    response = (
                        "Based on the retrieved documents, here are the relevant passages. "
                        f"(Confidence Score: {formatted_results['confidence_score']:.2f})"
                    )
                else:
                    response = QuestionHandler.get_legal_message()
                
                st.write(response)
            
            # Add assistant message to history
            st.session_state.chat_history.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=response)
            )
            
            # Display raw results if enabled
            if show_raw_results:
                with st.expander("Raw Results"):
                    st.json(formatted_results)
                    
        except Exception as e:
            st.error(f"Error processing question: {e}")
            logger.error(f"Error processing question: {e}", exc_info=True)

if __name__ == "__main__":
    main()