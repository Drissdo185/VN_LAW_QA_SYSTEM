from typing import Dict, List, Set
import re

class TrafficSynonyms:
    def __init__(self):
        self._setup_bidirectional_map()
        
    def _setup_bidirectional_map(self):
        self.legal_to_colloquial = {
            "không mang theo giấy đăng ký xe": [
                "không mang cà vẹt xe",
                "quên mang cà vẹt xe",
                "quên cà vẹt xe"
            ],
            
            "không chứng nhận đăng ký xe": [
                "không có cà vẹt",
                "chưa có cà vẹt xe",
                "chưa đăng ký giấy tờ",
                "không đăng ký xe",
                "sử dụng xe không có giấy tờ hợp lệ"
            ],
            
            "không chấp hành hiệu lệnh của đèn tín hiệu giao thông": [
                "vượt đèn đỏ",
                "không chấp hành hiệu lệnh đèn giao thông",
                "vượt đèn tín hiệu"
            ],
            
            "trong máu hoặc hơi thở có nồng độ cồn": [
                "say xỉn"
                "nồng độ cồn",
                "lái xe sau khi uống rượu bia",
                "điều khiển xe khi đã sử dụng rượu bia",
                "sử dụng rượu bia khi lái xe",
                "có cồn trong máu",
                "có cồn trong hơi thở",
                "đi nhậu",
                "uống rượu"
            ],
            
            "nồng độ cồn vượt quá 80 miligam/100 mililít máu hoặc vượt quá 0,4 miligam/1 lít khí thở": [
                "1 thùng bia",
                "uống 1 thùng bia"
            ],
            
            "không có giấy phép lái xe": [
                "không có gplx",
                "không có bằng lái",
                "giấy phép lái xe đã bị trừ hết điểm",
                "không có bằng",
                "chưa đủ tuổi để có giấy phép lái xe"
            ],
            
            "không mang giấy phép lái xe": [
                "không mang gplx",
                "không mang bằng lái"
            ],
            
            "vượt quá tốc độ quy định": [
                "chạy quá tốc độ",
                "vi phạm tốc độ",
                "điều khiển xe chạy quá tốc độ",
                "lái xe quá tốc độ cho phép",
                "chạy nhanh hơn tốc độ quy định",
                "vượt tốc độ tối đa",
                "phóng nhanh",
                "chạy nhanh quá mức"
            ],
            
            "điều khiển xe chạy bằng một bánh": [
                "bốc đầu"
            ],
            
            "vỉa hè": [
                "lề đường"
            ],
            
            "đỗ":[
                "đậu"
            ],
            
            "đỗ xe không đúng nơi quy định": [
                "đỗ xe sai quy định",
                "đậu xe sai quy định",
                "dừng xe sai quy định",
                "đậu xe không đúng nơi quy định",
                "đỗ xe trái phép",
                "đậu xe trái phép",
                "vi phạm quy định về đỗ xe",
                "đỗ xe nơi cấm đỗ",
                "dừng xe nơi cấm dừng"
            ],
            
            "không sử dụng mũ bảo hiểm": [
                "không đội mũ bảo hiểm",
                "thiếu mũ bảo hiểm",
                "vi phạm quy định về mũ bảo hiểm",
                "không đội nón bảo hiểm",
                "không đội mũ"
            ],
            
            "không chấp hành biển báo": [
                "vi phạm biển báo",
                "không tuân thủ biển báo",
                "bỏ qua biển báo",
                "không chấp hành biển hiệu",
                "không tuân thủ hiệu lệnh",
                "không tuân thủ biển hiệu đường bộ"
            ],
            
            "không đi đúng phần đường": [
                "đi không đúng làn đường",
                "vi phạm làn đường",
                "đi sai làn",
                "lấn làn",
                "chạy sai làn đường",
                "không tuân thủ làn đường"
            ],
            
            "gây tai nạn giao thông": [
                "gây tai nạn",
                "làm xảy ra tai nạn",
                "va chạm giao thông",
                "gây va chạm",
                "gây ra tai nạn",
                "để xảy ra tai nạn"
            ],
            
            "vượt xe không đúng quy định": [
                "vượt ẩu",
                "vượt xe sai quy định",
                "vượt xe không an toàn",
                "lấn làn vượt xe",
                "vượt xe nguy hiểm",
                "vượt xe trái phép"
            ],
            
            "điều khiển xe sau khi sử dụng ma túy": [
                "sử dụng ma túy",
                "sử dụng chất kích thích",
                "sử dụng chất cấm",
                "lái xe khi đã sử dụng chất ma túy",
                "chất ma túy trong cơ thể",
                "dương tính với ma túy"
            ]
        }
        
        self.colloquial_to_legal = {}
        for legal_term, colloquial_terms in self.legal_to_colloquial.items():
            for term in colloquial_terms:
                self.colloquial_to_legal[term.lower()] = legal_term
    
    def change_to_specific_term(self, text: str) -> Dict[str, str]:
        replancements = {}
        modified_text = text
        
        colloquial_terms = sorted(self.colloquial_to_legal.keys(), key=len, reverse=True)
        
        for colloquial_term in colloquial_terms:
            pattern = r'\b' + re.escape(colloquial_term) + r'\b'
            if re.search(pattern, modified_text.lower()):
                legal_term = self.colloquial_to_legal[colloquial_term]
                
                matches = re.finditer(pattern, modified_text.lower())
                offset = 0
                
                for match in matches:
                    start, end = match.span()
                    start += offset
                    end += offset
                    original_term = modified_text[start:end]
                    replancements[original_term] = legal_term
                    modified_text = modified_text[:start] + legal_term + modified_text[end:]
                    offset += len(legal_term) - len(original_term)
                    
        return{
            "correct_term": modified_text,
            'replacements': replancements
        }