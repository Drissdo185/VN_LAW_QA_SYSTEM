import streamlit as st
import weaviate

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.types import PydanticProgramMode

from CustomClass.CustomWeaviateVectorStore import CustomWeaviateVectorStore
from CustomClass.SentenceTransformerRerank import SentenceTransformerRerank
from CustomClass.DummySentenceTransformerRerank import DummySentenceTransformerRerank
from CustomClass.CustomRetrieverQueryEngine import CustomRetrieverQueryEngine
from CustomClass.CustomCondenseQuestionChatEngine import CustomCondenseQuestionChatEngine
from CustomClass.RemoveDuplicatedParentPostProcesser import RemoveDuplicatedParentPostProcesser


from utils import HybridRetriever
from utils import messages_to_prompt, completion_to_prompt
from utils import get_qa_prompt, get_standalone_question_prompt
from utils import StreamlitChatMessageHistory


st.set_page_config(page_title="Trợ lý tư vấn pháp luật",
                   page_icon="img/bkai_logo.png",
                   )

st.markdown(
    r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.write(
    f"""
    <div style="display: flex; align-items: center; margin-left: 0;">
        <h1 style="display: inline-block;">Trợ lý tư vấn pháp luật</h1>
        <sup style="margin-left:2px;font-size:small; color: green;">beta</sup>
    </div>
    """,
    unsafe_allow_html=True,
    )

foot = f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 30%;
        right: 0;
        width: 50%;
        padding: 0px 0px;
        text-align: center;
    ">
        <p>Enhanced by BKAI.FM</p>
    </div>
    """

st.markdown(foot, unsafe_allow_html=True)

@st.cache_resource
def configure_retriever():    
    WEAVIATE_URL = "http://localhost:9090"
    DATA_COLLECTION = "Law07032024"
    DEVICE = "cuda"
    TOP_K = 100
    RETRIEVAL_TOP_K = 10
    

    embed_model = HuggingFaceEmbedding(model_name="qducnguyen/vietnamese-bi-encoder", 
                                        max_length=256,
                                        device=DEVICE)

    client = weaviate.Client(WEAVIATE_URL)
    vector_store = CustomWeaviateVectorStore(weaviate_client=client, index_name=DATA_COLLECTION)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    hybrid_retriever = HybridRetriever(index, 
                                       top_k=TOP_K, 
                                       retrieval_top_k=RETRIEVAL_TOP_K)

    return hybrid_retriever

@st.cache_resource
def get_llm():
    llm = LlamaCPP(
        model_path="../ggml-vistral-7B-chat-q8.gguf",
        temperature=0,
        max_new_tokens=1024,
        context_window=8000,
        # generate_kwargs={"repeat_penalty": 1.05},
        model_kwargs={"rope_freq_base": 0,
                    "rope_freq_scale": 0,
                    "n_gpu_layers": 96},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=False)

    return  llm

@st.cache_resource
def get_query_engine():

    DEVICE="cuda"
    FINAL_TOP_RETRIEVAL=3
    reranker = DummySentenceTransformerRerank(top_n=FINAL_TOP_RETRIEVAL, 
                                        model="bkai-foundation-models/vietnamese-cross-encoder",
                                        device=DEVICE,
                                        keep_retrieval_score=True)


    hybrid_retriever = configure_retriever()    
    llm = get_llm()
    qa_prompt = get_qa_prompt()

    query_engine = CustomRetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker, 
                             RemoveDuplicatedParentPostProcesser(), 
                             MetadataReplacementPostProcessor(target_metadata_key="parent_text")],
        llm=llm,
        text_qa_template=qa_prompt,
        verbose=True
    )

    return query_engine

llm = get_llm()
query_engine = get_query_engine()
condense_question_prompt = get_standalone_question_prompt()
msgs = StreamlitChatMessageHistory()

TOKEN_LIMIT_MEMORY = 3000
memory = ChatMemoryBuffer.from_defaults(token_limit=TOKEN_LIMIT_MEMORY, 
                                        chat_history=msgs.messages)

chat_engine = CustomCondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    condense_question_prompt=condense_question_prompt,
    verbose=True,
    memory=memory,
    llm=llm
)

## Rewrite chat history
avatars = {MessageRole.USER: "user", MessageRole.ASSISTANT: "assistant"}
st.chat_message("assistant").write("Hãy hỏi tôi về pháp luật Việt Nam!")
for msg in msgs.messages:
    st.chat_message(avatars[msg.role]).write(msg.content)

## Do it
if user_query := st.chat_input("Vui lòng nhập câu hỏi về pháp luật của bạn ..."): 
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        response = chat_engine.stream_chat(user_query, st_container=st.container())    
        response_str = ""
        response_container = st.empty()
        for token in response.response_gen:
            response_str += token
            response_container.write(response_str)
