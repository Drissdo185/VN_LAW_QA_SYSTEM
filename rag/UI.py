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
    
    weaviate_host = st.sidebar.text_input("Weaviate Host", value="192.168.100.125")
    weaviate_port = st.sidebar.text_input("Weaviate Port", value="8080")
    weaviate_grpc_port = st.sidebar.number_input("Weaviate gRPC Port", value=50051)
    
    index_name = st.sidebar.text_input("Index Name", value="ND168")
    embed_model_name = st.sidebar.text_input("Embedding Model", value="dangvantuan/vietnamese-document-embedding")
    embed_cache_folder = st.sidebar.text_input("Cache Folder", value="/home/drissdo/.cache/huggingface/hub")
    
    model_name = st.sidebar.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    
    top_k = st.sidebar.slider("Top K Results", min_value=1, max_value=20, value=10)
    alpha = st.sidebar.slider("Alpha (Hybrid Search)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    
    max_iterations = st.sidebar.slider("Max Iterations", min_value=1, max_value=5, value=3)
    
    if st.sidebar.button("Khởi Tạo Hệ Thống"):
        with st.spinner("Đang khởi tạo hệ thống RAG..."):
            try:
                st.session_state.rag = AutoRAG(
                    weaviate_host=weaviate_host,
                    weaviate_port=int(weaviate_port),
                    weaviate_grpc_port=weaviate_grpc_port,
                    index_name=index_name,
                    embed_model_name=embed_model_name,
                    embed_cache_folder=embed_cache_folder,
                    model_name=model_name,
                    temperature=temperature,
                    top_k=top_k,
                    alpha=alpha,
                    max_iterations=max_iterations
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