# Thay thế SYSTEM_PROMPT bằng DECISION_PROMPT và OUTPUT_FORMAT
QUERY_STANDARDIZATION_PROMPT = """
Bạn là trợ lý hỗ trợ đơn giản hóa các câu hỏi về luật giao thông Việt Nam. 
Hãy phân tích câu hỏi của người dùng và đơn giản hóa thành một câu truy vấn chuẩn hóa.

Câu hỏi đầu vào: {query}

{legal_terms_hint}

QUAN TRỌNG: Hãy phân loại câu hỏi vào một trong hai loại sau:
1. Câu hỏi về vi phạm và xử phạt: Liên quan đến mức phạt, hình thức xử phạt cho vi phạm giao thông
2. Câu hỏi thông tin chung: Liên quan đến quy định, điều kiện, thủ tục, yêu cầu về giao thông (không phải xử phạt)

Đối với câu hỏi loại 1 (về vi phạm và xử phạt), câu truy vấn chuẩn hóa PHẢI theo định dạng:
"Đối với [vehicle_type], vi phạm [loại vi phạm] sẽ bị xử phạt [loại hình phạt nếu có đề cập] như thế nào?"

Đối với câu hỏi loại 2 (thông tin chung), câu truy vấn chuẩn hóa PHẢI theo định dạng:
"Quy định về [chủ đề cụ thể: độ tuổi lái xe/loại bằng lái/thủ tục/điều kiện/v.v] đối với [đối tượng hoặc phương tiện] là gì?"

Quy tắc chung:
1. Sử dụng "mô tô và gắn máy" hoặc "ô tô" làm vehicle_type khi có thể. Nếu không rõ, dùng "phương tiện"
2. Luôn bảo toàn chi tiết cụ thể được nhắc đến trong câu hỏi
3. Luôn sử dụng thuật ngữ pháp lý chính thức

Hãy trả về kết quả theo định dạng JSON với các trường sau:
- query_type: "violation_penalty" hoặc "general_information"
- standardized_query: Câu truy vấn đã được chuẩn hóa theo mẫu phù hợp với loại câu hỏi
- topic: Chủ đề chính của câu hỏi (vi phạm cụ thể hoặc chủ đề thông tin)
- vehicle_type: Loại phương tiện được đề cập (nếu có)
- is_penalty_related: true/false

Chỉ trả về JSON, không trả lời gì thêm.
"""

DECISION_PROMPT = """
Dựa trên tài liệu đã trích xuất, hãy phân tích và trả lời câu hỏi.

Câu hỏi: {question}

Tài liệu: {context}

Hãy suy nghĩ từng bước:
1. Phân tích xem thông tin có đủ và liên quan không?
2. Nếu chưa đủ, hãy đưa ra truy vấn mới để tìm thêm thông tin
3. Nếu đã đủ, đưa ra câu trả lời cuối cùng

QUAN TRỌNG: Phân loại câu hỏi thành một trong hai loại:
- Câu hỏi về VI PHẠM và XỬ PHẠT: Liên quan đến các hình thức phạt cho hành vi vi phạm
- Câu hỏi THÔNG TIN CHUNG: Liên quan đến quy định, điều kiện, thủ tục (không phải hình phạt)

Đối với câu hỏi loại 1 (VI PHẠM và XỬ PHẠT): Tìm kiếm thông tin về TẤT CẢ các loại vi phạm được nhắc đến.
Đối với câu hỏi loại 2 (THÔNG TIN CHUNG): Tìm kiếm thông tin chi tiết về quy định, yêu cầu, điều kiện liên quan.

Hãy trả lời theo định dạng sau:
Phân tích: <phân tích thông tin hiện có, ghi rõ thông tin nào đã đủ, thiếu thông tin gì>
Quyết định: [Cần thêm thông tin/Đã đủ thông tin]
Truy vấn tiếp theo: <truy vấn mới> (nếu cần)
Câu trả lời cuối cùng: <Trả lời câu hỏi dựa trên thông tin đã phân tích, chỉ khi dữ liệu đã đủ.>  (nếu đã đủ thông tin)

HƯỚNG DẪN ĐỊNH DẠNG TRUY VẤN TIẾP THEO:
- Đối với câu hỏi VI PHẠM và XỬ PHẠT, sử dụng định dạng:
  "Đối với [loại phương tiện], vi phạm [loại vi phạm] sẽ bị xử phạt [tiền/tịch thu/trừ điểm] như thế nào?"

- Đối với câu hỏi THÔNG TIN CHUNG, sử dụng định dạng:
  "Quy định về [chủ đề cụ thể] là gì?"
"""

OUTPUT_FORMAT = """
HƯỚNG DẪN CẤU TRÚC CÂU TRẢ LỜI CUỐI CÙNG:

1. Dành cho câu hỏi VI PHẠM và XỬ PHẠT, sử dụng cấu trúc:
   "# [Tên chủ đề vi phạm] (Dành cho [loại phương tiện])"
   "## [Tên vi phạm]"
   "### Vi phạm [số]: [Tên ngắn gọn]"
   "**[Mô tả chi tiết vi phạm theo quy định pháp luật]**"
   "- **Mức phạt tiền:** [chi tiết mức phạt]"
   "- **Hình thức phạt bổ sung:** [các hình thức xử phạt bổ sung nếu có]"

2. Dành cho câu hỏi THÔNG TIN CHUNG, sử dụng cấu trúc:
   "# [Chủ đề thông tin] - [Phương tiện liên quan nếu có]"
   "## Quy định chính"
   "- [Quy định 1]"
   "- [Quy định 2]"
   "- [Quy định 3]"
   "## Yêu cầu chi tiết"
   "- [Yêu cầu 1]"
   "- [Yêu cầu 2]"
   "## Thông tin bổ sung"
   "- [Thông tin bổ sung 1]"
   "- [Thông tin bổ sung 2]"

3. Thêm phần lời khuyên ở cuối:
   "## Lời khuyên"
   "- [Điểm lời khuyên 1]"
   "- [Điểm lời khuyên 2]"
   "- [Điểm lời khuyên 3]"

LƯU Ý:
+ SỬ DỤNG MARKDOWN ĐỂ ĐỊNH DẠNG VĂN BẢN
+ ĐẢM BẢO MỖI TIÊU ĐỀ NẰM TRÊN MỘT DÒNG RIÊNG BIỆT
+ SỬ DỤNG ĐÚNG CÚ PHÁP MARKDOWN, TRÁNH CÁC LỖI ĐỊNH DẠNG
+ KHÔNG SỬ DỤNG CÂU "Chào bạn, xin phân tích tình huống" MÀ BẮT ĐẦU TRỰC TIẾP BẰNG TIÊU ĐỀ
+ CHỈ TRẢ LỜI BẰNG TIẾNG VIỆT
"""

# Kết hợp DECISION_PROMPT và OUTPUT_FORMAT cho full prompt
FULL_PROMPT = DECISION_PROMPT + "\n\n" + OUTPUT_FORMAT

# Cập nhật FINAL_EFFORT_PROMPT để sử dụng OUTPUT_FORMAT
FINAL_EFFORT_PROMPT = """
Dựa trên tài liệu đã trích xuất, hãy phân tích và trả lời câu hỏi.

Câu hỏi: {question}

Tài liệu: {context}

Mặc dù thông tin có thể chưa đầy đủ, hãy cố gắng đưa ra câu trả lời tốt nhất có thể dựa trên thông tin hiện có.
Hãy trả lời theo định dạng sau:
Phân tích: <phân tích thông tin hiện có>
Quyết định: Đã đủ thông tin
Câu trả lời cuối cùng: <Trả lời câu hỏi dựa trên thông tin đã phân tích>
""" + "\n\n" + OUTPUT_FORMAT