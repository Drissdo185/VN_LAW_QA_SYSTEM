import streamlit as st
import asyncio
from llama_index.llms.openai import OpenAI
import logging

from config.config import load_and_validate_configs, get_domain_descriptions
from config.domain_router import DomainRouter
from reasoning.auto_rag import AutoRAG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Legal Document Search",
    page_icon="⚖️",
    layout="wide"
)

def initialize_system():
    """Initialize the RAG system components"""
    try:
        # Load configurations
        model_config, retrieval_config = load_and_validate_configs()
        
        # Initialize LLM
        llm = OpenAI(
            model=model_config.llm_model,
            api_key=model_config.llm_api_key
        )
        
        # Initialize domain router
        router = DomainRouter(
            llm=llm,
            model_config=model_config,
            retrieval_config=retrieval_config
        )
        
        return router, model_config
        
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        return None, None

def display_system_info():
    """Display available domains and their descriptions"""
    st.sidebar.header("Available Domains")
    
    domains = get_domain_descriptions()
    for domain, description in domains.items():
        st.sidebar.markdown(f"**{domain}**: {description}")

def main():
    st.title("Legal Document Search System")
    
    # Initialize session state
    if 'router' not in st.session_state:
        st.session_state.router, st.session_state.model_config = initialize_system()
    
    # Display system information
    display_system_info()
    
    # Main input area
    st.write("Enter your legal question below:")
    question = st.text_area("Question", height=100)
    
    if st.button("Search"):
        if not question:
            st.warning("Please enter a question.")
            return
            
        if not st.session_state.router:
            st.error("System not properly initialized.")
            return
            
        # Show spinner during processing
        with st.spinner("Processing your question..."):
            try:
                # Get search results
                results, domain = asyncio.run(st.session_state.router.route_and_search(question))
                
                if not results or not domain:
                    st.error("Could not process the question. Please try again.")
                    return
                
                # Display domain and search results
                st.subheader("Search Results")
                st.write(f"Domain: {domain}")
                
                # Get retriever from the router for the specific domain
                retriever = st.session_state.router.domain_pipelines[domain].retriever
                
                # Initialize AutoRAG with the domain-specific retriever
                auto_rag = AutoRAG(
                    model_config=st.session_state.model_config,
                    retriever=retriever
                )
                
                # Get detailed answer using AutoRAG
                rag_response = asyncio.run(auto_rag.get_answer(question))
                
                # Display RAG analysis
                with st.expander("Analysis", expanded=True):
                    st.write("**Analysis:**", rag_response["analysis"])
                    st.write("**Decision:**", rag_response["decision"])
                    if rag_response["next_query"]:
                        st.write("**Suggested follow-up query:**", rag_response["next_query"])
                    if rag_response["final_answer"]:
                        st.write("**Final Answer:**", rag_response["final_answer"])
                    st.write("**Token Usage:**", rag_response["token_usage"])
                
                # Display retrieved documents
                st.subheader("Retrieved Documents")
                for i, result in enumerate(results, 1):
                    with st.expander(f"Document {i} (Score: {result.score:.3f})"):
                        st.write(result.text)
                        if hasattr(result.node, 'metadata'):
                            st.write("Metadata:", result.node.metadata)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"Error processing question: {e}", exc_info=True)

if __name__ == "__main__":
    main()