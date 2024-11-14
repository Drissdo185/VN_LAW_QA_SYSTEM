import streamlit as st
import weaviate
from typing import List
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
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["OPENAI_API_KEY"] = ""

def initialize_components():
    """Initialize all required components."""
    # Initialize Weaviate client
    cluster_url = ""
    api_key = ""
    weaviate_client = weaviate.connect_to_wcs(
        cluster_url=cluster_url,
        auth_credentials=Auth.api_key(api_key)
    )
    
    MODEL_NAME = "dangvantuan/vietnamese-embedding"
    embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME,
                                   max_length=256)
    
    # Initialize LLM
    llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    Settings.llm = llm
    
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
        dense_weight=0.2,
        query_mode="hybrid"
    )
    
    return retrieval_manager, llm

def generate_response(question: str, retrieved_docs: List, llm: OpenAI) -> str:
    """Generate a response using GPT-4 based on retrieved documents."""
    # Prepare context from retrieved documents
    context = "\n".join([doc.text for doc in retrieved_docs if doc.text])
    
    # Prepare prompt
    prompt = f"""Based on the following legal documents, please answer the question. 
    Provide a clear, concise response and cite specific articles or sections when relevant.
    If the information is not found in the documents, acknowledge that and provide general guidance.
    
    Question: {question}
    
    Legal Documents:
    {context}
    
    Response:"""
    
    try:
        # Generate response
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
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[ChatMessage] = []
    
    # Initialize components
    try:
        retrieval_manager, llm = initialize_components()
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
        
        try:
            # Create container for search results
            search_container = st.container()
            
            # Process question
            processed_question = QuestionHandler.process_standalone_question(
                user_input,
                st.session_state.chat_history
            )
            
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
            
            # Generate response using GPT-4
            if results:
                response = generate_response(user_input, results, llm)
            else:
                response = "I couldn't find any relevant legal documents to answer your question. Could you please rephrase or provide more context?"
            
            # Display assistant response
            with st.chat_message("assistant"):
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