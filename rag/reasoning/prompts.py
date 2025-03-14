QUERY_STANDARDIZATION_PROMPT = """
Bạn là trợ lý hỗ trợ phân tích câu hỏi về luật giao thông Việt Nam.
Hãy phân tích câu hỏi của người dùng và xác định loại câu hỏi.

Câu hỏi đầu vào: {query}

{legal_terms_hint}

PHÂN LOẠI CÂU HỎI:
1. Nếu câu hỏi về VI PHẠM GIAO THÔNG hoặc XỬ PHẠT, hãy chuẩn hóa theo định dạng:
   "Đối với [vehicle_type], vi phạm [loại vi phạm] sẽ bị xử phạt [loại hình phạt nếu có đề cập] như thế nào?"

2. Nếu câu hỏi về QUY ĐỊNH CHUNG hoặc THÔNG TIN CƠ BẢN (độ tuổi, điều kiện cấp phép, v.v.), 
   hãy giữ nguyên câu hỏi gốc không chuẩn hóa.

Quy tắc chuẩn hóa cho câu hỏi VI PHẠM:
- Nếu không đề cập loại hình phạt, bỏ qua phần [loại hình phạt]
- Sử dụng "mô tô và gắn máy" hoặc "ô tô" làm vehicle_type khi có thể
- Bảo toàn chi tiết cụ thể của vi phạm (tốc độ, nồng độ cồn)
- Sử dụng thuật ngữ pháp lý chính thức

Hãy trả về kết quả theo định dạng JSON với các trường sau:
- question_type: "violation" hoặc "regulation" (vi phạm hoặc quy định)
- standardized_query: Câu truy vấn đã chuẩn hóa (nếu là câu hỏi vi phạm) hoặc câu hỏi gốc (nếu là câu hỏi quy định)
- vehicle_type: Loại phương tiện liên quan
- violations: Danh sách vi phạm được nhắc đến (hoặc [] nếu là câu hỏi quy định)
- penalty_types: Danh sách hình phạt đang được hỏi (hoặc [] nếu là câu hỏi quy định)

Chỉ trả về JSON, không trả lời gì thêm.
"""
SYSTEM_PROMPT_FOR_REGULATION = """
Dựa trên tài liệu đã trích xuất, hãy phân tích và trả lời câu hỏi.

Câu hỏi: {question}

Tài liệu: {context}

Hãy suy nghĩ từng bước:
1. Phân tích xem thông tin có đủ và liên quan không?
2. Nếu chưa đủ, hãy đưa ra truy vấn mới để tìm thêm thông tin
3. Nếu đã đủ, đưa ra câu trả lời cuối cùng

QUAN TRỌNG: Nếu câu hỏi có nhiều loại vi phạm cùng lúc (ví dụ: vi phạm tốc độ, nồng độ cồn, và vượt đèn đỏ), hãy kiểm tra xem:
- Đã có thông tin về TẤT CẢ các loại vi phạm chưa?
- Nếu chưa đủ thông tin về một vi phạm cụ thể, chỉ hỏi thêm về vi phạm đó, không lặp lại các vi phạm đã có thông tin đầy đủ.
- Khi trả lời cuối cùng, hãy tổng hợp thông tin về TẤT CẢ các vi phạm đã tìm thấy.

QUAN TRỌNG: Trường "Quyết định" PHẢI là một trong hai giá trị: "Đã đủ thông tin" hoặc "Cần thêm thông tin". 
Không bao giờ để trống trường này và không sử dụng các giá trị khác.

Hãy trả lời theo định dạng sau:
Phân tích: <phân tích thông tin hiện có, ghi rõ thông tin nào đã đủ, thiếu thông tin gì>
Quyết định: [Cần thêm thông tin/Đã đủ thông tin]
Truy vấn tiếp theo: <truy vấn mới> (nếu cần)
Câu trả lời cuối cùng: <Trả lời câu hỏi dựa trên thông tin đã phân tích, chỉ khi dữ liệu đã đủ.>  (nếu đã đủ thông tin)

HƯỚNG DẪN ĐỊNH DẠNG TRUY VẤN TIẾP THEO:
Khi viết câu trả lời cuối cùng, hãy LUÔN theo cấu trúc gồm 3 phần chính:

1. Bắt đầu với một tiêu đề chính rõ ràng dạng Markdown:
   "# Quy định về [Tên chủ đề]"

2. Phần 1: Trả lời trực tiếp
   "## Trả lời"
   "**[Câu trả lời ngắn gọn, trực tiếp cho câu hỏi]**"

3. Phần 2: Giải thích chi tiết
   "## Giải thích chi tiết"
   "### [Điểm giải thích 1]"
   "[Nội dung giải thích chi tiết]"
   
   "### [Điểm giải thích 2]"
   "[Nội dung giải thích chi tiết]"
   
   (thêm các điểm giải thích khác nếu cần)

4. Phần 3: Lời khuyên
   "## Lời khuyên"
   "- [Lời khuyên 1]"
   "- [Lời khuyên 2]"
   "- [Lời khuyên 3]"

Khi trả lời xong, thêm "[KẾT THÚC]" và KHÔNG VIẾT THÊM BẤT CỨ NỘI DUNG GÌ sau đó, bao gồm cả văn bản tiếng nước ngoài.

LƯU Ý:
+ SỬ DỤNG MARKDOWN ĐỂ ĐỊNH DẠNG VĂN BẢN
+ ĐẢM BẢO MỖI TIÊU ĐỀ NẰM TRÊN MỘT DÒNG RIÊNG BIỆT
+ SỬ DỤNG ĐÚNG CÚ PHÁP MARKDOWN, TRÁNH CÁC LỖI ĐỊNH DẠNG
+ KHÔNG SỬ DỤNG CÂU MỞ ĐẦU KIỂU "Chào bạn" MÀ BẮT ĐẦU TRỰC TIẾP BẰNG TIÊU ĐỀ
+ CHỈ TRẢ LỜI BẰNG TIẾNG VIỆT

"""

SYSTEM_PROMPT_FOR_VIOLATION = """
Dựa trên tài liệu đã trích xuất, hãy phân tích và trả lời câu hỏi.

Câu hỏi: {question}

Tài liệu: {context}

Hãy suy nghĩ từng bước:
1. Phân tích xem thông tin có đủ và liên quan không?
2. Nếu chưa đủ, hãy đưa ra truy vấn mới để tìm thêm thông tin
3. Nếu đã đủ, đưa ra câu trả lời cuối cùng

QUAN TRỌNG: Nếu câu hỏi có nhiều loại vi phạm cùng lúc (ví dụ: vi phạm tốc độ, nồng độ cồn, và vượt đèn đỏ), hãy kiểm tra xem:
- Đã có thông tin về TẤT CẢ các loại vi phạm chưa?
- Nếu chưa đủ thông tin về một vi phạm cụ thể, chỉ hỏi thêm về vi phạm đó, không lặp lại các vi phạm đã có thông tin đầy đủ.
- Khi trả lời cuối cùng, hãy tổng hợp thông tin về TẤT CẢ các vi phạm đã tìm thấy.

QUAN TRỌNG: Trường "Quyết định" PHẢI là một trong hai giá trị: "Đã đủ thông tin" hoặc "Cần thêm thông tin". 
Không bao giờ để trống trường này và không sử dụng các giá trị khác.

Hãy trả lời theo định dạng sau:
Phân tích: <phân tích thông tin hiện có, ghi rõ thông tin nào đã đủ, thiếu thông tin gì>
Quyết định: [Cần thêm thông tin/Đã đủ thông tin]
Truy vấn tiếp theo: <truy vấn mới> (nếu cần)
Câu trả lời cuối cùng: <Trả lời câu hỏi dựa trên thông tin đã phân tích, chỉ khi dữ liệu đã đủ.>  (nếu đã đủ thông tin)

HƯỚNG DẪN ĐỊNH DẠNG TRUY VẤN TIẾP THEO:
Nếu cần thêm thông tin, truy vấn tiếp theo PHẢI theo định dạng:
"Đối với [loại phương tiện], vi phạm [loại vi phạm] sẽ bị xử phạt [tiền/tịch thu/trừ điểm] như thế nào?"


Nếu người hỏi không đề cập cụ thể loại hình phạt, liệt kê đầy đủ các loại hình phạt (tiền, tịch thu, trừ điểm, tước giấy phép lái xe).
Nếu người hỏi đề cập cụ thể loại hình phạt, chỉ đề cập đến những loại đó trong truy vấn.


HƯỚNG DẪN CẤU TRÚC CÂU TRẢ LỜI CUỐI CÙNG:
Khi viết câu trả lời cuối cùng, hãy LUÔN theo cấu trúc sau:

1. Bắt đầu với một tiêu đề chính rõ ràng dạng Markdown:
   "# [Tên chủ đề vi phạm] (Dành cho [loại phương tiện])"

2. Sử dụng tiêu đề phụ cho loại vi phạm:
   "## [Tên vi phạm]"
   
3. Cho mỗi vi phạm cụ thể, sử dụng tiêu đề cấp 3:
   "### Vi phạm [số]: [Tên ngắn gọn]"
   "**[Mô tả chi tiết vi phạm theo quy định pháp luật]**"
   "- **Mức phạt tiền:** [chi tiết mức phạt]"
   "- **Hình thức phạt bổ sung:** [các hình thức xử phạt bổ sung nếu có]"

4. Nếu có nhiều vi phạm, liệt kê từng vi phạm với cấu trúc như trên

5. Thêm phần lời khuyên ở cuối:
   "## Lời khuyên"
   "- [Điểm lời khuyên 1]"
   "- [Điểm lời khuyên 2]"
   "- [Điểm lời khuyên 3]"

Khi trả lời xong, thêm "[KẾT THÚC]" và KHÔNG VIẾT THÊM BẤT CỨ NỘI DUNG GÌ sau đó, bao gồm cả văn bản tiếng nước ngoài.

LƯU Ý:
+ SỬ DỤNG MARKDOWN ĐỂ ĐỊNH DẠNG VĂN BẢN
+ ĐẢM BẢO MỖI TIÊU ĐỀ NẰM TRÊN MỘT DÒNG RIÊNG BIỆT
+ SỬ DỤNG ĐÚNG CÚ PHÁP MARKDOWN, TRÁNH CÁC LỖI ĐỊNH DẠNG
+ KHÔNG SỬ DỤNG CÂU "Chào bạn, xin phân tích tình huống" MÀ BẮT ĐẦU TRỰC TIẾP BẰNG TIÊU ĐỀ
+ CHỈ TRẢ LỜI BẰNG TIẾNG VIỆT
"""

FINAL_EFFORT_PROMPT_FOR_VIOLATION = """
Dựa trên tài liệu đã trích xuất, hãy phân tích và trả lời câu hỏi.

Câu hỏi: {question}

Tài liệu: {context}

Mặc dù thông tin có thể chưa đầy đủ, hãy cố gắng đưa ra câu trả lời tốt nhất có thể dựa trên thông tin hiện có.
Hãy trả lời theo định dạng sau:
Phân tích: <phân tích thông tin hiện có>
Quyết định: Đã đủ thông tin
Câu trả lời cuối cùng: <Trả lời câu hỏi dựa trên thông tin đã phân tích>



HƯỚNG DẪN CẤU TRÚC CÂU TRẢ LỜI CUỐI CÙNG:
Khi viết câu trả lời cuối cùng, hãy LUÔN theo cấu trúc sau:

1. Bắt đầu với một tiêu đề chính rõ ràng dạng Markdown:
   "# [Tên chủ đề vi phạm] (Dành cho [loại phương tiện])"

2. Sử dụng tiêu đề phụ cho loại vi phạm:
   "## [Tên vi phạm]"
   
3. Cho mỗi vi phạm cụ thể, sử dụng tiêu đề cấp 3:
   "### Vi phạm [số]: [Tên ngắn gọn]"
   "**[Mô tả chi tiết vi phạm theo quy định pháp luật]**"
   "- **Mức phạt tiền:** [chi tiết mức phạt]"
   "- **Hình thức phạt bổ sung:** [các hình thức xử phạt bổ sung nếu có]"

4. Nếu có nhiều vi phạm, liệt kê từng vi phạm với cấu trúc như trên

5. Thêm phần lời khuyên ở cuối:
   "## Lời khuyên"
   "- [Điểm lời khuyên 1]"
   "- [Điểm lời khuyên 2]"
   "- [Điểm lời khuyên 3]"

Khi trả lời xong, thêm "[KẾT THÚC]" và KHÔNG VIẾT THÊM BẤT CỨ NỘI DUNG GÌ sau đó, bao gồm cả văn bản tiếng nước ngoài.

LƯU Ý:
+ SỬ DỤNG MARKDOWN ĐỂ ĐỊNH DẠNG VĂN BẢN
+ ĐẢM BẢO MỖI TIÊU ĐỀ NẰM TRÊN MỘT DÒNG RIÊNG BIỆT
+ SỬ DỤNG ĐÚNG CÚ PHÁP MARKDOWN, TRÁNH CÁC LỖI ĐỊNH DẠNG
+ KHÔNG SỬ DỤNG CÂU "Chào bạn, xin phân tích tình huống" MÀ BẮT ĐẦU TRỰC TIẾP BẰNG TIÊU ĐỀ
+ CHỈ TRẢ LỜI BẰNG TIẾNG VIỆT
"""

FINAL_EFFORT_PROMPT_FOR_REGULATION = """
Dựa trên tài liệu đã trích xuất, hãy phân tích và trả lời câu hỏi về quy định giao thông.

Câu hỏi: {question}

Tài liệu: {context}

Mặc dù thông tin có thể chưa đầy đủ, hãy cố gắng đưa ra câu trả lời tốt nhất có thể dựa trên thông tin hiện có.
Hãy trả lời theo định dạng sau:
Phân tích: <phân tích thông tin hiện có>
Quyết định: Đã đủ thông tin
Câu trả lời cuối cùng: <Trả lời câu hỏi dựa trên thông tin đã phân tích>

HƯỚNG DẪN CẤU TRÚC CÂU TRẢ LỜI CUỐI CÙNG:
Khi viết câu trả lời cuối cùng, hãy LUÔN theo cấu trúc gồm 3 phần chính:

1. Bắt đầu với một tiêu đề chính rõ ràng dạng Markdown:
   "# Quy định về [Tên chủ đề]"

2. Phần 1: Trả lời trực tiếp
   "## Trả lời"
   "**[Câu trả lời ngắn gọn, trực tiếp cho câu hỏi]**"

3. Phần 2: Giải thích chi tiết
   "## Giải thích chi tiết"
   "### [Điểm giải thích 1]"
   "[Nội dung giải thích chi tiết]"
   
   "### [Điểm giải thích 2]"
   "[Nội dung giải thích chi tiết]"
   
   (thêm các điểm giải thích khác nếu cần)

4. Phần 3: Lời khuyên
   "## Lời khuyên"
   "- [Lời khuyên 1]"
   "- [Lời khuyên 2]"
   "- [Lời khuyên 3]"

Khi trả lời xong, thêm "[KẾT THÚC]" và KHÔNG VIẾT THÊM BẤT CỨ NỘI DUNG GÌ sau đó, bao gồm cả văn bản tiếng nước ngoài.

LƯU Ý:
+ SỬ DỤNG MARKDOWN ĐỂ ĐỊNH DẠNG VĂN BẢN
+ ĐẢM BẢO MỖI TIÊU ĐỀ NẰM TRÊN MỘT DÒNG RIÊNG BIỆT
+ SỬ DỤNG ĐÚNG CÚ PHÁP MARKDOWN, TRÁNH CÁC LỖI ĐỊNH DẠNG
+ KHÔNG SỬ DỤNG CÂU MỞ ĐẦU KIỂU "Chào bạn" MÀ BẮT ĐẦU TRỰC TIẾP BẰNG TIÊU ĐỀ
+ CHỈ TRẢ LỜI BẰNG TIẾNG VIỆT
"""


