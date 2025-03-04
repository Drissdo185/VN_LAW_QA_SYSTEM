import streamlit as st
import asyncio
import time
from typing import Dict
import os
from contextlib import contextmanager
import logging
from log.logging_config import setup_logging

# Initialize logging first, before any other operations
setup_logging(
    level=logging.INFO,
    log_format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
    log_file='app.log'
)
logger = logging.getLogger(__name__)

from retrieval.search_pipline import SearchPipeline
from retrieval.retriever import DocumentRetriever
from retrieval.vector_store import VectorStoreManager
from reasoning.auto_rag import AutoRAG
from web_handle.web_search import WebSearchIntegrator, WebEnabledAutoRAG
from config.config import (
    ModelConfig, 
    RetrievalConfig, 
    WeaviateConfig, 
    WebSearchConfig,
    Domain,
    LLMProvider,
    VLLMConfig
)
from utils import measure_performance, RAGException

if 'processing_time' not in st.session_state:
    st.session_state.processing_time = None
if 'show_web_search_prompt' not in st.session_state:
    st.session_state.show_web_search_prompt = False
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'initial_response' not in st.session_state:
    st.session_state.initial_response = None

@st.cache_resource
def init_configs():
    """Initialize configuration objects"""
    logger.info("Initializing configurations")
    try:
        weaviate_config = WeaviateConfig(
            url=os.getenv("WEAVIATE_TRAFFIC_URL"),
            api_key=os.getenv("WEAVIATE_TRAFFIC_KEY")
        )
        model_config = ModelConfig()
        retrieval_config = RetrievalConfig()
        web_search_config = WebSearchConfig()
        logger.info("Configurations initialized successfully")
        return weaviate_config, model_config, retrieval_config, web_search_config
    except Exception as e:
        logger.error(f"Error initializing configurations: {str(e)}")
        raise

class DomainComponents:
    def __init__(self, domain: Domain, configs: tuple, llm_provider: LLMProvider = LLMProvider.OPENAI):
        logger.info(f"Initializing components for domain: {domain.value} with LLM provider: {llm_provider.value}")
        self.domain = domain
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
            self.vector_store_manager = VectorStoreManager(
                weaviate_config=weaviate_config,
                model_config=model_config,
                domain=domain
            )
            self.vector_store_manager.initialize()
            
            self.retriever = DocumentRetriever(
                index=self.vector_store_manager.get_index(),
                config=retrieval_config
            )
            
            self.search_pipeline = SearchPipeline(
                retriever=self.retriever,
                model_config=model_config,
                retrieval_config=retrieval_config,
                domain=self.domain
            )
            
            self.auto_rag = AutoRAG(
                model_config=model_config,
                retriever=self.retriever,
                current_domain=domain
            )
            
            # Initialize web search if enabled
            if web_search_config.web_search_enabled:
                logger.info("Initializing web search integration")
                self.web_search = WebSearchIntegrator(
                    google_api_key=web_search_config.google_api_key,
                    google_cse_id=web_search_config.google_cse_id,
                    model_config=model_config,
                    retrieval_config=retrieval_config,
                    domain=domain
                )
            else:
                logger.info("Web search is disabled")
                self.web_search = None
                
            logger.info(f"Components initialized successfully for domain: {domain.value}")
            
        except Exception as e:
            logger.error(f"Error initializing domain components: {str(e)}")
            raise
    
    def get_web_enabled_rag(self, fallback_threshold: float = 0.5):
        """Create web-enabled RAG wrapper on demand"""
        logger.info("Creating web-enabled RAG wrapper")
        if hasattr(self, 'web_search') and self.web_search:
            return WebEnabledAutoRAG(
                auto_rag=self.auto_rag,
                web_search=self.web_search,
                fallback_threshold=fallback_threshold
            )
        logger.warning("Web search not available, returning standard AutoRAG")
        return self.auto_rag

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up domain components")
        if hasattr(self, 'vector_store_manager'):
            self.vector_store_manager.cleanup()

@st.cache_resource
def init_domain_components(configs: tuple, llm_provider: LLMProvider) -> Dict[Domain, DomainComponents]:
    """Initialize components for all domains with specified LLM provider"""
    logger.info(f"Initializing components for all domains with LLM provider: {llm_provider.value}")
    return {domain: DomainComponents(domain, configs, llm_provider) for domain in Domain}

@contextmanager
def get_event_loop():
    """Context manager for event loop handling"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.close()

@measure_performance
async def process_question(auto_rag: AutoRAG, question: str):
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
        provider_name = "OpenAI GPT-4o mini" if provider == LLMProvider.OPENAI else "Qwen2.5-14B (vLLM)"
        st.info(f"LLM Provider: {provider_name}")
        logger.info(f"Response generated using: {provider_name}")

def display_source_info(response: Dict):
    """Display information about the response source"""
    if 'source' in response:
        source_type = "🌐 Web search" if response['source'] == 'web_search' else "📚 Knowledge base"
        st.info(f"Source: {source_type}")
        logger.info(f"Response source: {source_type}")

def display_results(response: Dict):
    """Display the analysis results"""
    logger.info("Displaying results")
    display_llm_info(response)
    display_source_info(response)
    display_token_usage(response["token_usage"])
    display_performance_metrics()
    
    st.subheader("📊 Analysis Results")
    
    if response.get("analysis"):
        st.markdown("**Analysis:**")
        st.info(response["analysis"])
    
    if response.get("decision"):
        st.markdown("**Decision:**")
        st.success(response["decision"])
    
    if response.get("final_answer"):
        st.markdown("**Final Answer:**")
        st.write(response["final_answer"])

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
    logger.info("Starting application")
    st.set_page_config(page_title="QA System for Vietnamese Law", layout="wide")
    st.title("📖 Question and Answering System for Vietnamese Law")

    # Sidebar for domain selection and LLM provider
    with st.sidebar:
        st.header("⚙️ Configuration")
        try:
            configs = init_configs()
            
            # LLM provider selection
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
            
            # Initialize components with selected provider
            domain_components = init_domain_components(configs, selected_provider)
            
            # Domain selection
            selected_domain = st.selectbox(
                "Select Domain",
                options=[domain.value for domain in Domain],
                format_func=lambda x: x.title()
            )
            current_domain = Domain(selected_domain)
            components = domain_components[current_domain]
            
            logger.info(f"Selected domain: {current_domain.value}, LLM provider: {selected_provider.value}")
            
            # Display vLLM configuration if selected
            if selected_provider == LLMProvider.VLLM:
                with st.expander("vLLM Configuration"):
                    st.text_input("API URL", configs[1].vllm_config.api_url, key="vllm_url")
                    st.text_input("Model Name", configs[1].vllm_config.model_name, key="vllm_model")
                    st.slider("Temperature", 0.0, 1.0, configs[1].vllm_config.temperature, key="vllm_temp")
                    st.slider("Top P", 0.0, 1.0, configs[1].vllm_config.top_p, key="vllm_top_p")
            
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
                with get_event_loop() as loop:
                    response = loop.run_until_complete(
                        process_question(components.auto_rag, question)
                    )
                
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
        finally:
            components.cleanup()
    
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
                        # Get web-enabled RAG instance
                        web_rag = components.get_web_enabled_rag()
                        
                        with get_event_loop() as loop:
                            response = loop.run_until_complete(
                                process_question(web_rag, st.session_state.current_question)
                            )
                        
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
                finally:
                    components.cleanup()
    
if __name__ == "__main__":
    main()