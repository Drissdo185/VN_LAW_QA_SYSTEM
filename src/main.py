import streamlit as st
import asyncio
from typing import List
import os
from dataclasses import dataclass

# Import your existing classes
from retrieval.search_pipline import SearchPipeline
from retrieval.retriever import DocumentRetriever
from retrieval.vector_store import VectorStoreManager
from reasoning.auto_rag import AutoRAG
from config.config import ModelConfig, RetrievalConfig, WeaviateConfig

# Initialize configurations
@st.cache_resource
def init_configs():
    weaviate_config = WeaviateConfig(
        url=os.getenv("WEAVIATE_TRAFFIC_URL"),
        api_key=os.getenv("WEAVIATE_TRAFFIC_KEY"),
        collection="ND168"
    )
    
    model_config = ModelConfig()
    retrieval_config = RetrievalConfig()
    
    return weaviate_config, model_config, retrieval_config

# Initialize components
@st.cache_resource
def init_components(weaviate_config, model_config, retrieval_config):
    # Initialize vector store
    vector_store_manager = VectorStoreManager(
        weaviate_config=weaviate_config,
        model_config=model_config
    )
    vector_store_manager.initialize()
    
    # Initialize retriever
    retriever = DocumentRetriever(
        index=vector_store_manager.get_index(),
        config=retrieval_config
    )
    
    # Initialize search pipeline
    search_pipeline = SearchPipeline(
        retriever=retriever,
        model_config=model_config,
        retrieval_config=retrieval_config
    )
    
    # Initialize AutoRAG
    auto_rag = AutoRAG(
        model_config=model_config,
        retriever=retriever
    )
    
    return search_pipeline, auto_rag

async def process_question(auto_rag, question):
    """Async function to process the question using AutoRAG"""
    return await auto_rag.get_answer(question)

def display_token_usage(token_usage):
    """Display token usage information in a formatted way"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Input Tokens", token_usage["input_tokens"])
    with col2:
        st.metric("Output Tokens", token_usage["output_tokens"])
    with col3:
        st.metric("Total Tokens", token_usage["total_tokens"])

def main():
    st.title("RAG System Demo")
    
    # Initialize configurations and components
    try:
        weaviate_config, model_config, retrieval_config = init_configs()
        search_pipeline, auto_rag = init_components(
            weaviate_config, 
            model_config, 
            retrieval_config
        )
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return
    
    # Input section
    st.header("Ask a Question")
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if not question:
            st.warning("Please enter a question")
            return
            
        try:
            # Show search results
            st.subheader("Search Results")
            with st.spinner("Searching..."):
                search_results = search_pipeline.search(question)
                
                for i, result in enumerate(search_results, 1):
                    with st.expander(f"Result {i} (Score: {result.score:.3f})"):
                        st.write(result.text)
            
            # Get and show RAG response
            st.subheader("RAG Response")
            with st.spinner("Generating response..."):
                # Create event loop and run async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(process_question(auto_rag, question))
                loop.close()
                
                # Display token usage
                st.subheader("Token Usage")
                display_token_usage(response["token_usage"])
                
                # Display analysis
                st.markdown("**Analysis:**")
                st.write(response["analysis"])
                
                # Display decision
                st.markdown("**Decision:**")
                st.write(response["decision"])
                
                # Display next query if exists
                if response["next_query"]:
                    st.markdown("**Next Query:**")
                    st.write(response["next_query"])
                
                # Display final answer if exists
                if response["final_answer"]:
                    st.markdown("**Final Answer:**")
                    st.write(response["final_answer"])
                    
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

if __name__ == "__main__":
    main()