import streamlit as st
import asyncio
import time
from typing import Dict
import os
from contextlib import contextmanager

from retrieval.search_pipline import SearchPipeline
from retrieval.retriever import DocumentRetriever
from retrieval.vector_store import VectorStoreManager
from reasoning.auto_rag import AutoRAG
from config.config import ModelConfig, RetrievalConfig, WeaviateConfig, Domain
from utils import measure_performance, RAGException

if 'processing_time' not in st.session_state:
    st.session_state.processing_time = None

@st.cache_resource
def init_configs():
    weaviate_config = WeaviateConfig(
        url=os.getenv("WEAVIATE_TRAFFIC_URL"),
        api_key=os.getenv("WEAVIATE_TRAFFIC_KEY")
    )
    model_config = ModelConfig()
    retrieval_config = RetrievalConfig()
    return weaviate_config, model_config, retrieval_config

class DomainComponents:
    def __init__(self, domain: Domain, configs: tuple):
        self.domain = domain
        weaviate_config, model_config, retrieval_config = configs
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
    
    def cleanup(self):
        if hasattr(self, 'vector_store_manager'):
            self.vector_store_manager.cleanup()

@st.cache_resource
def init_domain_components(configs: tuple) -> Dict[Domain, DomainComponents]:
    return {domain: DomainComponents(domain, configs) for domain in Domain}

@contextmanager
def get_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.close()

@measure_performance
async def process_question(auto_rag: AutoRAG, question: str):
    return await auto_rag.get_answer(question)

def display_token_usage(token_usage: Dict[str, int]):
    with st.expander("Token Usage", expanded=False):
        st.write(f"**Input Tokens:** {token_usage['input_tokens']}")
        st.write(f"**Output Tokens:** {token_usage['output_tokens']}")
        st.write(f"**Total Tokens:** {token_usage['total_tokens']}")

def display_performance_metrics():
    if st.session_state.processing_time:
        with st.expander("Performance Metrics", expanded=False):
            st.write(f"**Processing Time:** {st.session_state.processing_time:.2f} seconds")

def main():
    st.set_page_config(page_title="QA System for Vietnamese Law", layout="wide")
    st.title("📖 Question and Answering System for Vietnamese Law")

    # Sidebar for domain selection
    with st.sidebar:
        st.header("⚙️ Configuration")
        try:
            configs = init_configs()
            domain_components = init_domain_components(configs)
            selected_domain = st.selectbox(
                "Select Domain",
                options=[domain.value for domain in Domain],
                format_func=lambda x: x.title()
            )
            current_domain = Domain(selected_domain)
            components = domain_components[current_domain]
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            return

    # Main Input Section
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question:")
    
    if st.button("💡 Get Answer", use_container_width=True) and question:
        try:
            start_time = time.time()
            progress_bar = st.progress(0)
            
            with st.spinner("🔍 Processing..."):
                with get_event_loop() as loop:
                    response = loop.run_until_complete(
                        process_question(components.auto_rag, question)
                    )
                
                st.session_state.processing_time = time.time() - start_time
                
                if "error" in response:
                    st.error(response["error"])
                    return
                
                progress_bar.progress(100)
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
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            components.cleanup()
    
if __name__ == "__main__":
    main()
