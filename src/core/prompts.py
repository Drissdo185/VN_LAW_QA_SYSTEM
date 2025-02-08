from llama_index.core import PromptTemplate

SYSTEM_PROMPT = PromptTemplate(
    template=(
                        "Hãy trả lời câu hỏi bằng cách trích xuất thông tin từ tài liệu. "
                        "Phân tích và trích xuất thông tin hữu ích từ mỗi tài liệu được cung cấp. "
                        "Nếu thông tin không đầy đủ hoặc không liên quan, "
                        "hãy tinh chỉnh truy vấn và tìm kiếm lại cho đến khi có thể trả lời câu hỏi.\n\n"
                        "Câu hỏi: {question}\n"
                        "Tài liệu đã trích xuất: {documents}\n"
                        "Hãy suy nghĩ từng bước:\n"
                        "1. Phân tích xem thông tin trích xuất có liên quan không\n"
                        "2. Quyết định xem có cần thêm thông tin không\n"
                        "3. Hoặc tinh chỉnh truy vấn hoặc đưa ra câu trả lời cuối cùng\n"
                        "4. Nếu không thể tìm thấy thông tin phù hợp sau nhiều lần thử, hãy thông báo rằng không thể tìm thấy câu trả lời\n"
                    )
)

QUERY_GENERATION_PROMPT = PromptTemplate(
     template=(
                        "Dựa trên ngữ cảnh và câu hỏi sau đây, hãy tạo một truy vấn tìm kiếm được tinh chỉnh.\n"
                        "Ngữ cảnh: {context}\n"
                        "Câu hỏi: {question}\n"
                        "Các truy vấn trước đây: {previous_queries}\n"
                        "Tạo một truy vấn cụ thể và tập trung để tìm thông tin còn thiếu."
                    )
)

