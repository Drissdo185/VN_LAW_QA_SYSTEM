import streamlit as st
import asyncio
import time
from typing import Dict
import os
from contextlib import contextmanager
import logging
import sys
import nest_asyncio

# Apply nest_asyncio before any other asyncio operations
nest_asyncio.apply()

# Initialize logging
from log.logging_config import setup_logging

setup_logging(
    level=logging.INFO,
    log_format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Import the rest of the dependencies
try:
    from rag.question_process.traffic_synonyms import TrafficSynonymExpander
    from retrieval.search_pipline import SearchPipeline
    from retrieval.retriever import DocumentRetriever
    from retrieval.vector_store import VectorStoreManager
    from reasoning.auto_rag import AutoRAG
    from reasoning.enhanced_autorag import EnhancedAutoRAG
    from web_handle.web_search import WebSearchIntegrator, WebEnabledAutoRAG
    from config.config import (
        ModelConfig, 
        RetrievalConfig, 
        WeaviateConfig, 
        WebSearchConfig,
        LLMProvider,
        VLLMConfig
    )
    from utils import measure_performance
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    st.error(f"Error importing modules: {str(e)}")
    sys.exit(1)

# Initialize session state
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = None
if 'show_web_search_prompt' not in st.session_state:
    st.session_state.show_web_search_prompt = False
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'initial_response' not in st.session_state:
    st.session_state.initial_response = None
if 'use_simplified_query' not in st.session_state:
    st.session_state.use_simplified_query = True  # Default to using simplified queries

@st.cache_resource
def init_configs():
    """Initialize configuration objects"""
    logger.info("Initializing configurations")
    try:
        weaviate_config = WeaviateConfig(
            url=os.getenv("WEAVIATE_TRAFFIC_URL"),
            api_key=os.getenv("WEAVIATE_TRAFFIC_KEY"),
            collection="ND168"  # Explicitly set the collection
        )
        model_config = ModelConfig()
        retrieval_config = RetrievalConfig()
        web_search_config = WebSearchConfig()
        logger.info("Configurations initialized successfully")
        return weaviate_config, model_config, retrieval_config, web_search_config
    except Exception as e:
        logger.error(f"Error initializing configurations: {str(e)}")
        raise

@st.cache_resource
def init_components(configs: tuple, llm_provider: LLMProvider = LLMProvider.OPENAI):
    """
    Initialize all RAG components with the specified configurations.
    
    Args:
        configs: Tuple of configuration objects (weaviate_config, model_config, etc.)
        llm_provider: LLM provider to use
        
    Returns:
        Dictionary containing all initialized components
    """
    logger.info(f"Initializing components with LLM provider: {llm_provider.value}")
    weaviate_config, base_model_config, retrieval_config, web_search_config = configs
    
    # Create a copy of model_config with the selected LLM provider
    model_config = ModelConfig(
        device=base_model_config.device,
        embedding_model=base_model_config.embedding_model,
        cross_encoder_model=base_model_config.cross_encoder_model,
        chunk_size=base_model_config.chunk_size,
        chunk_overlap=base_model_config.chunk_overlap,
        llm_provider=llm_provider,
        openai_model=base_model_config.openai_model,
        openai_api_key=base_model_config.openai_api_key,
        vllm_config=base_model_config.vllm_config
    )
    
    try:
        # Initialize vector store
        logger.info("Initializing vector store")
        vector_store_manager = VectorStoreManager(
            weaviate_config=weaviate_config,
            model_config=model_config
        )
        vector_store_manager.initialize()
        
        # Initialize retriever
        logger.info("Initializing document retriever")
        retriever = DocumentRetriever(
            index=vector_store_manager.get_index(),
            config=retrieval_config
        )
        
        # Initialize search pipeline
        logger.info("Initializing search pipeline")
        search_pipeline = SearchPipeline(
            retriever=retriever,
            model_config=model_config,
            retrieval_config=retrieval_config
        )
        
        # Initialize standard AutoRAG
        logger.info("Initializing AutoRAG")
        auto_rag = AutoRAG(
            model_config=model_config,
            retriever=retriever,
            search_pipeline=search_pipeline  # Always provide search_pipeline
        )
        
        # Initialize enhanced AutoRAG
        logger.info("Initializing EnhancedAutoRAG")
        enhanced_auto_rag = EnhancedAutoRAG(
            model_config=model_config,
            retriever=retriever,
            search_pipeline=search_pipeline  # Always provide search_pipeline
        )
        
        # Initialize web search if enabled
        web_search = None
        if web_search_config.web_search_enabled:
            logger.info("Initializing web search integration")
            web_search = WebSearchIntegrator(
                google_api_key=web_search_config.google_api_key,
                google_cse_id=web_search_config.google_cse_id,
                model_config=model_config,
                retrieval_config=retrieval_config
            )
        else:
            logger.info("Web search is disabled")
        
        logger.info("All components initialized successfully")
        
        # Return all components in a dictionary
        return {
            'vector_store_manager': vector_store_manager,
            'retriever': retriever,
            'search_pipeline': search_pipeline,
            'auto_rag': auto_rag,
            'enhanced_auto_rag': enhanced_auto_rag,
            'web_search': web_search
        }
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

class AppComponents:
    """
    Container for all application components, handling initialization and cleanup.
    """
    def __init__(self, configs: tuple, llm_provider: LLMProvider = LLMProvider.OPENAI):
        logger.info(f"Setting up application components with LLM provider: {llm_provider.value}")
        self.configs = configs
        
        try:
            # Initialize all components using the helper function
            components = init_components(configs, llm_provider)
            
            # Assign components to class attributes for easy access
            self.vector_store_manager = components['vector_store_manager']
            self.retriever = components['retriever']
            self.search_pipeline = components['search_pipeline']
            self.auto_rag = components['auto_rag']
            self.enhanced_auto_rag = components['enhanced_auto_rag']
            self.web_search = components['web_search']
            
            logger.info("AppComponents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AppComponents: {str(e)}")
            raise
    
    def get_active_rag(self, use_simplified=True):
        """Get active RAG component based on whether to use query simplification"""
        if use_simplified:
            logger.info("Using enhanced AutoRAG with query simplification")
            return self.enhanced_auto_rag
        else:
            logger.info("Using standard AutoRAG")
            return self.auto_rag
    
    def get_web_enabled_rag(self, use_simplified=True, fallback_threshold: float = 0.5):
        """Create web-enabled RAG wrapper on demand"""
        logger.info("Creating web-enabled RAG wrapper")
        if hasattr(self, 'web_search') and self.web_search:
            base_rag = self.get_active_rag(use_simplified)
            return WebEnabledAutoRAG(
                auto_rag=base_rag,
                web_search=self.web_search,
                fallback_threshold=fallback_threshold
            )
        logger.warning("Web search not available, returning standard AutoRAG")
        return self.get_active_rag(use_simplified)

    def cleanup(self):
        """Clean up resources - only call this when the application is shutting down"""
        logger.info("Cleaning up components")
        if hasattr(self, 'vector_store_manager'):
            self.vector_store_manager.cleanup()

# Create a session-based component holder that persists across reruns
@st.cache_resource
def get_app_components(configs, selected_provider):
    """Create and cache AppComponents to persist across Streamlit reruns"""
    logger.info("Creating or retrieving cached AppComponents")
    return AppComponents(configs, selected_provider)

# Fixed event loop handler
def run_async(coroutine):
    """Run an async function safely in Streamlit"""
    try:
        # First try getting the current event loop
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If there is no event loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run the coroutine in the event loop
    return loop.run_until_complete(coroutine)

@measure_performance
async def process_question(auto_rag, question: str):
    """Process a question using the RAG system"""
    logger.info(f"Processing question: {question}")
    return await auto_rag.get_answer(question)

def display_token_usage(token_usage: Dict[str, int]):
    """Display token usage information"""
    with st.expander("Token Usage", expanded=False):
        st.write(f"**Input Tokens:** {token_usage['input_tokens']}")
        st.write(f"**Output Tokens:** {token_usage['output_tokens']}")
        st.write(f"**Total Tokens:** {token_usage['total_tokens']}")
        logger.debug(f"Token usage - Total: {token_usage['total_tokens']}")

def display_performance_metrics():
    """Display performance metrics"""
    if st.session_state.processing_time:
        with st.expander("Performance Metrics", expanded=False):
            st.write(f"**Processing Time:** {st.session_state.processing_time:.2f} seconds")
            logger.debug(f"Processing time: {st.session_state.processing_time:.2f} seconds")

def display_llm_info(response: Dict):
    """Display information about which LLM was used"""
    if 'llm_provider' in response:
        provider = response['llm_provider']
        if provider == LLMProvider.OPENAI:
            provider_name = "OpenAI GPT-4o mini"
        elif provider == LLMProvider.VLLM:
            provider_name = "Qwen2.5-14B (vLLM)"
        st.info(f"LLM Provider: {provider_name}")
        logger.info(f"Response generated using: {provider_name}")

def display_source_info(response: Dict):
    """Display information about the response source"""
    if 'source' in response:
        source_type = "🌐 Web search" if response['source'] == 'web_search' else "📚 Knowledge base"
        st.info(f"Source: {source_type}")
        logger.info(f"Response source: {source_type}")

def display_query_info(response: Dict):
    """Display information about query processing pipeline"""
    if 'query_info' in response:
        with st.expander("Query Processing Pipeline", expanded=False):
            query_info = response['query_info']
            
            # Original query
            st.write("**Original Query:**")
            st.write(query_info.get('original_query', 'N/A'))
            
            # Expanded query (with synonyms)
            st.write("**Expanded Query (Synonyms):**")
            st.write(query_info.get('expanded_query', 'N/A'))
            
            # Standardized query
            st.write("**Standardized Query:**")
            st.write(query_info.get('standardized_query', 'N/A'))
            
            # Show detected violations
            st.write("**Detected Violations:**")
            violations = query_info.get('metadata', {}).get('violations', [])
            if violations:
                for violation in violations:
                    st.write(f"- {violation}")
            else:
                st.write("No specific violations detected")
                
            # Show vehicle type
            st.write("**Vehicle Type:**")
            vehicle_type = query_info.get('metadata', {}).get('vehicle_type', 'Not specified')
            st.write(vehicle_type)
            
            # Show penalty types
            st.write("**Penalty Types:**")
            penalty_types = query_info.get('metadata', {}).get('penalty_types', [])
            if penalty_types:
                for penalty in penalty_types:
                    st.write(f"- {penalty}")
            else:
                st.write("No specific penalty types detected")
            
            # Log the complete transformation
            logger.info(f"Query transformation: '{query_info.get('original_query')}' → " +
                      f"'{query_info.get('expanded_query')}' → " +
                      f"'{query_info.get('standardized_query')}'")

def display_results(response: Dict):
    """Display the analysis results"""
    logger.info("Displaying results")
    display_llm_info(response)
    display_source_info(response)
    display_token_usage(response["token_usage"])
    display_performance_metrics()
    
    # Display query processing pipeline if available
    if 'query_info' in response:
        display_query_info(response)
    
    st.subheader("📊 Analysis Results")
    
    if response.get("analysis"):
        st.markdown("**Analysis:**")
        st.info(response["analysis"])
    
    if response.get("decision"):
        st.markdown("**Decision:**")
        st.success(response["decision"])
    
    if response.get("final_answer"):
        st.markdown("**Final Answer:**")
        st.markdown(response["final_answer"], unsafe_allow_html=False)
        
    # Show note if present (e.g., for best-effort answers)
    if response.get("note"):
        st.warning(response["note"])

def needs_web_search(response: Dict) -> bool:
    """Check if web search might be helpful"""
    needs_search = (
        response.get("decision", "").lower() == "không tìm thấy đủ thông tin" or
        not response.get("final_answer")
    )
    logger.info(f"Needs web search: {needs_search}")
    return needs_search

def main():
    """Main application function"""
    try:
        logger.info("Starting application")
        st.set_page_config(page_title="QA System for Vietnamese Traffic Law", layout="wide")
        st.title("📖 Question and Answering System for Vietnamese Traffic Law")

        # Sidebar for configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            try:
                configs = init_configs()
                
                # LLM provider selection (removed Ollama)
                llm_provider_options = {
                    "OpenAI GPT-4o mini": LLMProvider.OPENAI,
                    "Qwen2.5-14B (vLLM)": LLMProvider.VLLM
                }
                selected_provider_name = st.selectbox(
                    "Select LLM Provider",
                    options=list(llm_provider_options.keys()),
                    index=0
                )
                selected_provider = llm_provider_options[selected_provider_name]
                
                # Query standardization toggle
                st.session_state.use_simplified_query = st.toggle(
                    "Use Query Standardization",
                    value=st.session_state.use_simplified_query,
                    help="Enable LLM-based query standardization to extract key legal concepts"
                )
                
                logger.info(f"Selected LLM provider: {selected_provider.value}")
                logger.info(f"Query simplification enabled: {st.session_state.use_simplified_query}")
                
                # Initialize components with selected provider and updated config
                # Use the cached version to maintain connection state
                components = get_app_components(configs, selected_provider)
                
            except Exception as e:
                error_msg = f"Error initializing components: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                return

        # Main Input Section
        st.subheader("Ask a Question")
        question = st.text_input("Enter your question:")
        search_button = st.button("💡 Get Answer", use_container_width=True)
        
        # Process initial search
        if search_button and question:
            logger.info(f"Processing new question: {question}")
            # Reset states for new question
            if question != st.session_state.current_question:
                st.session_state.show_web_search_prompt = False
                st.session_state.initial_response = None
            
            st.session_state.current_question = question
            
            try:
                start_time = time.time()
                progress_bar = st.progress(0)
                
                with st.spinner("🔍 Searching knowledge base..."):
                    # Get the appropriate RAG model based on simplification toggle
                    rag_model = components.get_active_rag(st.session_state.use_simplified_query)
                    
                    # Run async function safely
                    response = run_async(process_question(rag_model, question))
                    
                    st.session_state.processing_time = time.time() - start_time
                    logger.info(f"Question processed in {st.session_state.processing_time:.2f} seconds")
                    
                    if "error" in response:
                        logger.error(f"Error in response: {response['error']}")
                        st.error(response["error"])
                        return
                    
                    progress_bar.progress(100)
                    
                    # Store initial response
                    st.session_state.initial_response = response
                    
                    # Check if web search might help
                    if needs_web_search(response):
                        st.session_state.show_web_search_prompt = True
                        logger.info("Web search prompt triggered")
                    
                    # Display initial results
                    display_results(response)
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                return
            # Connection stays open, no cleanup call here
        
        # Show web search prompt if needed
        if st.session_state.show_web_search_prompt:
            logger.info("Displaying web search prompt")
            st.warning("No satisfactory answer found in the knowledge base.")
            web_search_cols = st.columns([2, 1])
            with web_search_cols[0]:
                st.info("Would you like to search the web for more information?")
            with web_search_cols[1]:
                if st.button("🔍 Search Web", use_container_width=True):
                    logger.info("Web search initiated")
                    try:
                        start_time = time.time()
                        progress_bar = st.progress(0)
                        
                        with st.spinner("🌐 Searching web..."):
                            # Get web-enabled RAG instance with appropriate simplification setting
                            web_rag = components.get_web_enabled_rag(st.session_state.use_simplified_query)
                            
                            # Run async function safely
                            response = run_async(process_question(web_rag, st.session_state.current_question))
                            
                            st.session_state.processing_time = time.time() - start_time
                            logger.info(f"Web search completed in {st.session_state.processing_time:.2f} seconds")
                            
                            if "error" in response:
                                logger.error(f"Error in web search response: {response['error']}")
                                st.error(response["error"])
                                return
                            
                            progress_bar.progress(100)
                            
                            # Clear web search prompt
                            st.session_state.show_web_search_prompt = False
                            
                            # Display web search results
                            display_results(response)
                            
                    except Exception as e:
                        error_msg = f"Error during web search: {str(e)}"
                        logger.error(error_msg)
                        st.error(error_msg)
                    # Connection stays open, no cleanup call here
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()