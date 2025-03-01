SYSTEM_PROMPT = """
Dựa trên tài liệu đã trích xuất, hãy phân tích và trả lời câu hỏi.

Câu hỏi: {question}

Tài liệu: {context}

Hãy suy nghĩ từng bước:
1. Phân tích xem thông tin có đủ và liên quan không?
2. Nếu chưa đủ, hãy đưa ra truy vấn mới để tìm thêm thông tin
3. Nếu đã đủ, đưa ra câu trả lời cuối cùng

Hãy trả lời theo định dạng sau:
Phân tích: <phân tích thông tin hiện có>
Quyết định: [Cần thêm thông tin/Đã đủ thông tin]
Truy vấn tiếp theo: <truy vấn mới> (nếu cần)
Câu trả lời cuối cùng: <câu trả lời> (nếu đã đủ thông tin)
"""

DOMAIN_VALIDATION_PROMPT = """
Analyze if the given question belongs to the traffic domain or stock market domain.
If it's clearly about traffic (roads, transportation, accidents, traffic rules, etc.), return "traffic".
If it's clearly about stocks (market, trading, finance, companies, etc.), return "stock".
If unclear, return the closest matching domain.

Question: {question}

Return only one word (traffic/stock):"""