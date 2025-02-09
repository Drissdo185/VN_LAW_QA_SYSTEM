from typing import List, Optional
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from pyvi import ViTokenizer
from logging import getLogger

logger = getLogger(__name__)

class QuestionHandler:
    """Handles question preprocessing and context management."""
    
    @staticmethod
    def process_question(
        question: str,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> str:
        """Process question with chat history context."""
        try:
            processed_question = ViTokenizer.tokenize(question.lower())
            
            if not chat_history:
                return processed_question
                
            context = "\n".join(
                f"{'User' if msg.role == MessageRole.USER else 'Assistant'}: {msg.content}"
                for msg in chat_history
            )
            
            return f"{context}\nCurrent question: {processed_question}"
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise

    @staticmethod
    def get_legal_message() -> str:
        """Get standard message for non-legal questions."""
        return (
            "Xin lỗi, tôi là trợ lý về pháp luật. "
            "Tôi không tìm thấy dữ liệu liên quan tới câu hỏi của bạn. "
            "Vui lòng hỏi các vấn đề liên quan tới pháp luật để tôi có thể trợ giúp bạn."
        )