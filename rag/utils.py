# Custom fusion 
from typing import List, Optional, Sequence

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.prompts.base import PromptTemplate

class HybridRetriever(BaseRetriever):
    def __init__(self, index, top_k, retrieval_top_k):
        self.top_k = top_k
        self.retrieval_top_k = retrieval_top_k
        self.dense_retrieval = index.as_retriever(similarity_top_k=self.top_k,
                                                  vector_store_query_mode="default")
        
        self.bm25_retrieval = index.as_retriever(similarity_top_k=self.top_k,
                                                 vector_store_query_mode="sparse")
        super().__init__()

    def _retrieve(self, query, **kwargs):
        """
        query should be segmented and refined
        """
        vector_nodes = self.dense_retrieval.retrieve(query, **kwargs)
        bm25_nodes = self.bm25_retrieval.retrieve(query, **kwargs)

        # Reciprocal Rerank Fusion Retriever
        # Dictionary: node_id --> NodeWithScore + metadata 
        # Sort values based on score values of dictionary 
        all_nodes_dict = {}
        for idx, n in enumerate(vector_nodes):
            n.metadata["dense_rank"] = float(idx + 1)
            n.metadata["dense_score"] = n.score
            # 1 / rank
            n.score = 1  / (n.metadata["dense_rank"]+60)
            # dictionary 
            all_nodes_dict[n.node_id] = n
            n.metadata["bm25_rank"] = 9999 # ~ inf
            n.metadata["bm25_score"] = 0.0

        for idx, n in enumerate(bm25_nodes):
            node_id = n.node_id
            if node_id in all_nodes_dict:
                refer_node = all_nodes_dict[node_id]
            else:
                all_nodes_dict[node_id] = n
                refer_node = n
                refer_node.metadata["dense_rank"] = 9999 # ~ inf
                refer_node.metadata["dense_score"] = 0.0
                refer_node.score = 0.0

            refer_node.metadata["bm25_rank"] = float(idx + 1)
            refer_node.metadata["bm25_score"] = n.score

            # 1 / rank
            refer_node.score = refer_node.score + 1 / (refer_node.metadata["bm25_rank"]+60)

        ## Return list of sorted nodes based on score 
        all_nodes = sorted(all_nodes_dict.values(), key = lambda x: -x.score)
        
        return [x for x in all_nodes if x.metadata["bm25_score"] > 8 and x.metadata["dense_score"] > 0.35][:self.retrieval_top_k]



BOS, EOS = "<s>", "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\nCâu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực.Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."

def messages_to_prompt(
    messages: Sequence[ChatMessage], system_prompt: Optional[str] = None
) -> str:
    string_messages: List[str] = []
    if messages[0].role == MessageRole.SYSTEM:
        # pull out the system message (if it exists in messages)
        system_message_str = messages[0].content or ""
        messages = messages[1:]
    else:
        system_message_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    system_message_str = f"{B_SYS} {system_message_str.strip()} {E_SYS}"

    for i in range(0, len(messages), 2):
        # first message should always be a user
        user_message = messages[i]
        assert user_message.role == MessageRole.USER

        if i == 0:
            # make sure system prompt is included at the start
            str_message = f"{BOS} {B_INST} {system_message_str} "
        else:
            # end previous user-assistant interaction
            string_messages[-1] += f" {EOS}"
            # no need to include system prompt
            str_message = f"{BOS} {B_INST} "

        # include user message content
        str_message += f"{user_message.content} {E_INST}"

        if len(messages) > (i + 1):
            # if assistant message exists, add to str_message
            assistant_message = messages[i + 1]
            assert assistant_message.role == MessageRole.ASSISTANT
            str_message += f" {assistant_message.content}"

        string_messages.append(str_message)

    return "".join(string_messages)


def completion_to_prompt(completion: str, system_prompt: Optional[str] = None) -> str:
    system_prompt_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    return (
        f"{BOS} {B_INST} {B_SYS} {system_prompt_str.strip()} {E_SYS} "
        f"{completion.strip()} {E_INST}"
    )

def get_qa_prompt():
    # DEFAULT_TEXT_QA_PROMPT_TMPL ="""[INST] Chỉ dựa vào ngữ cảnh được cung cấp để trả lời câu hỏi ở phía dưới. Nếu bạn không biết câu trả lời, chỉ cần nói là bạn không biết câu trả lời dựa vào nội dung ngữ cảnh, đừng cố gắng bịa câu ra câu trả lời.
    # 

    # Ngữ cảnh: 

    # {context_str}

    # Câu hỏi: {query_str}

    # Trả lời:
    # [/INST]
    # """
    
# Your response should be strictly based on the context provided below. Remember it! If the information available does not allow you to formulate an answer, state clearly that you are unable to provide an answer based on the given context. Avoid speculating or creating an answer without direct support from the provided context. Ensure your answer is concluded effectively and coherently.    

    DEFAULT_TEXT_QA_PROMPT_TMPL ="""
You are a highly specialized legal assistant designed to provide information and insights solely on legal matters. Your purpose is to assist users by providing accurate, reliable, and contextually relevant legal information.

When answering questions, it is crucial that you strictly adhere to the context provided below. Your responses should be directly related to and based upon the information given in the user's query. Extrapolation or deviation from the provided context is not permitted. 

You only answer questions related to legal matters. You are not to engage with or provide opinions on non-legal matters. If a question falls outside the scope of legal domains, you must politely decline to answer, stating that the query is outside your area of expertise. 

Please end your answer propertly. Do not repeate sentences.

Context: 
{context_str}

Question: {query_str}

Useful Answer:
"""

    qa_prompt = PromptTemplate(
        DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
    )

    return qa_prompt

def get_standalone_question_prompt():

    # DEFAULT_TEMPLATE = """\
    # [INST] 
    # Dựa vào cuộc hội thoại (giữa Human và Assistant) và tin nhắn tiếp theo từ Human, hãy viết cái tin nhắn đó sang một câu hỏi độc lập và bao gồm tất cả các ngữ cảnh.

    # Hội thoại:

    # {chat_history}

    # Tin nhắn tiếp theo: {question}
    # Câu hỏi độc lập:
    # [/INST]
    # """

#     DEFAULT_TEMPLATE = """\
# [INST]
# Given a conversation (between Human and Assistant) and a follow up message from Human, \
# rewrite the message to be a standalone question that captures all relevant context \
# from the conversation.

# Let me share a couple examples that will be important. 

# If this is the second question onwards, you should properly rephrase the question like this:

# ```
# Chat History:

# Human: Dạo này Đức thế nào?

# Assistant: Đức bị chấn thương và không thể đi học.

# Follow Up Message: Anh ý bị chấn thương ở đâu?
# Standalone Question:
# Đức bị chấn thương ở đâu?
# ```

# Now, with those examples, here is the actual chat history and input question.

# Chat History:

# {chat_history}

# Follow Up Message:
# {question}

# Standalone Question:
# [/INST]
# """
    
    DEFAULT_TEMPLATE = """
Given a Chat History (between Human and Assistant) and a Follow Up Message from Human, \
rewrite the message to be a Standalone Question that captures all relevant Chat History context \
then determine if the Standalone Question is related to legal matter.

Chat History:

{chat_history}

Follow Up Message: {question}
"""

    condense_question_prompt = PromptTemplate(DEFAULT_TEMPLATE)

    return condense_question_prompt


class StreamlitChatMessageHistory():
    """
    Chat message history that stores messages in Streamlit session state.

    Args:
        key: The key to use in Streamlit session state for storing messages.
    """

    def __init__(self, key: str = "llamaindex_messages"):
        try:
            import streamlit as st
        except ImportError as e:
            raise ImportError(
                "Unable to import streamlit, please run `pip install streamlit`."
            ) from e

        if key not in st.session_state:
            st.session_state[key] = []
        self._messages = st.session_state[key]

    @property
    def messages(self) -> List[ChatMessage]:  # type: ignore
        """Retrieve the current list of messages"""
        return self._messages

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the session memory"""
        self._messages.append(message)

    def clear(self) -> None:
        """Clear session memory"""
        self._messages.clear()


class StreamlitChatMessageHistoryTest():
    """
    Chat message history that stores messages in Streamlit session state.

    Args:
        key: The key to use in Streamlit session state for storing messages.
    """

    def __init__(self, session_state, key: str = "llamaindex_messages"):
        if key not in session_state:
            session_state[key] = []
        self._messages = session_state[key]

    @property
    def messages(self) -> List[ChatMessage]:  # type: ignore
        """Retrieve the current list of messages"""
        return self._messages

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the session memory"""
        self._messages.append(message)

    def clear(self) -> None:
        """Clear session memory"""
        self._messages.clear()