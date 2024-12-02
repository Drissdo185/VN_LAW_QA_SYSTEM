from typing import List, Optional
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from pyvi import ViTokenizer

class QuestionHandler:
    """Handles question preprocessing and context management."""
    
    @staticmethod
    def process_standalone_question(
        question: str,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> str:
        """Process question to generate standalone version."""
        processed_question = ViTokenizer.tokenize(question.lower())
        
        if not chat_history:
            return processed_question
            
        context = ""
        for msg in chat_history:
            if msg.role == MessageRole.USER:
                context += f"User: {msg.content}\n"
            else:
                context += f"Assistant: {msg.content}\n"
                
        return f"{context}\nCurrent question: {processed_question}"

    @staticmethod
    def get_legal_message() -> str:
        """Get standard message for non-legal questions."""
        return (
            "Xin lỗi, tôi là trợ lý về pháp luật. "
            "Tôi không tìm thấy dữ liệu liên quan tới câu hỏi của bạn. "
            "Vui lòng hỏi các vấn đề liên quan tới pháp luật để tôi có thể trợ giúp bạn."
        )