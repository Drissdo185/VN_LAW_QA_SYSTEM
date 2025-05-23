import streamlit as st
import json
import time
import tiktoken
from auto_rag import AutoRAG

st.set_page_config(
    page_title="Trợ Lý Luật Giao Thông Việt Nam",
    page_icon="🚦",
    layout="wide"
)

def count_tokens(text, model="gpt-4o-mini"):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def initialize_session_state():
    """Initialize session state variables"""
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

def setup_rag_system():
    """Configure and initialize the RAG system"""
    st.sidebar.title("Cấu Hình Hệ Thống")
    
    
    
    llm_provider = st.sidebar.radio("LLM Provider", ["openai", "vllm"], index=0)

    if llm_provider == "openai":
        model_name = st.sidebar.selectbox("OpenAI Model", ["gpt-4o-mini"], index=0)
        vllm_api_url = ""
        vllm_model_name = ""
        vllm_max_tokens = 4096
    
    else:
        model_name = ""
        vllm_api_url = st.sidebar.text_input("VLLM API URL", value="http://192.168.100.125:8000/v1/completions")
        vllm_model_name = st.sidebar.selectbox(
        "VLLM Model Name", 
        ["Qwen/Qwen2.5-14B-Instruct-AWQ", "Qwen/Qwen2.5-3B-Instruct"],
        index=0
    )
        vllm_max_tokens = st.sidebar.number_input("Max Tokens", value=4096)
        
    
    
    if st.sidebar.button("Khởi Tạo Hệ Thống"):
        with st.spinner("Đang khởi tạo hệ thống RAG..."):
            try:
                st.session_state.rag = AutoRAG(
                    weaviate_host="192.168.100.125",
                    weaviate_port=8080,
                    weaviate_grpc_port=50051,
                    index_name="ND168",
                    embed_model_name="bkai-foundation-models/vietnamese-bi-encoder",
                    embed_cache_folder="/home/drissdo/.cache/huggingface/hub",
                    model_name="gpt-4o-mini",
                    temperature=0.2,
                    top_k=10,
                    alpha=0.5,
                    max_iterations=3,
                    llm_provider="openai",
                    vllm_api_url=vllm_api_url,
                    vllm_model_name=vllm_model_name,
                    vllm_max_tokens=vllm_max_tokens
                )
                st.sidebar.success("Hệ thống đã được khởi tạo thành công!")
            except Exception as e:
                st.sidebar.error(f"Lỗi khi khởi tạo: {str(e)}")
    
    st.sidebar.divider()
    st.sidebar.subheader("Thống kê token")
    st.sidebar.metric("Tổng token đã sử dụng", st.session_state.total_tokens)
    
    if st.sidebar.button("Xóa lịch sử"):
        st.session_state.history = []
        st.session_state.total_tokens = 0

def process_query(question):
    """Process a user query with the RAG system"""
    if st.session_state.rag is None:
        st.error("Vui lòng khởi tạo hệ thống trước khi sử dụng!")
        return None
    
    input_tokens = count_tokens(question)
    st.session_state.total_tokens += input_tokens
    
    start_time = time.time()
    result = st.session_state.rag.process(question)
    processing_time = time.time() - start_time
    
    answer = result["answer"]
    output_tokens = count_tokens(answer)
    st.session_state.total_tokens += output_tokens
    
    query_record = {
        "question": question,
        "answer": answer,
        "processing_time": processing_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "details": result
    }
    
    st.session_state.history.append(query_record)
    return query_record

def display_results(query_record):
    """Display the results of a processed query"""
    if query_record:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Câu trả lời")
            st.markdown(query_record["answer"])
            
            with st.expander("Chi tiết về câu hỏi"):
                st.write("**Câu hỏi gốc:**", query_record["question"])
                st.write("**Câu hỏi đã xử lý:**", query_record["details"]["processed_question"])
                st.write("**Loại câu hỏi:**", query_record["details"].get("question_type", "N/A"))
                
                if "replacements" in query_record["details"]:
                    st.write("**Thay thế thuật ngữ:**")
                    for original, legal in query_record["details"]["replacements"].items():
                        st.write(f"- '{original}' → '{legal}'")
        
        with col2:
            st.markdown("### Thống kê")
            st.metric("Thời gian xử lý", f"{query_record['processing_time']:.2f}s")
            st.metric("Token đầu vào", query_record["input_tokens"])
            st.metric("Token đầu ra", query_record["output_tokens"])
            st.metric("Tổng token", query_record["total_tokens"])
            
            if "iterations" in query_record["details"]:
                st.write(f"**Số lần lặp:** {query_record['details']['iterations']}")
        
        # Query information section
        st.markdown("### Thông tin truy vấn")
        if "formatted_query" in query_record["details"]:
            st.write("**Truy vấn định dạng:**", query_record["details"]["formatted_query"])
        
        # Display iterations
        if "query_history" in query_record["details"]:
            st.markdown("#### Quá trình lặp")
            tabs = st.tabs([f"Lần lặp {i+1}" for i in range(len(query_record["details"]["query_history"]))])
            
            for i, (tab, query) in enumerate(zip(tabs, query_record["details"]["query_history"])):
                with tab:
                    st.write("**Truy vấn:**", query)
        
        # Display extracted information
        if "context" in query_record["details"] and query_record["details"]["context"]:
            st.markdown("#### Thông tin đã trích xuất")
            st.text_area("Nội dung", query_record["details"]["context"], height=300)

def display_history():
    """Display the history of processed queries"""
    st.subheader("Lịch sử câu hỏi")
    if not st.session_state.history:
        st.info("Chưa có câu hỏi nào được đặt")
        return
    
    for i, query in enumerate(reversed(st.session_state.history)):
        with st.expander(f"{len(st.session_state.history) - i}. {query['question']}"):
            st.write("**Câu trả lời:**")
            st.write(query["answer"])
            st.write(f"**Thời gian xử lý:** {query['processing_time']:.2f}s")
            st.write(f"**Token sử dụng:** {query['total_tokens']}")

def main():
    """Main application function"""
    initialize_session_state()
    
    st.title("Trợ Lý Luật Giao Thông Việt Nam 🚦")
    
    setup_rag_system()
    
    st.divider()
    
    question = st.text_area("Nhập câu hỏi của bạn về luật giao thông:", height=100)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("Gửi câu hỏi")
    
    if submit_button and question and not st.session_state.is_processing:
        st.session_state.is_processing = True
        
        with st.spinner("Đang xử lý câu hỏi của bạn..."):
            query_record = process_query(question)
        
        st.session_state.is_processing = False
        display_results(query_record)
    
    st.divider()
    display_history()

if __name__ == "__main__":
    main()