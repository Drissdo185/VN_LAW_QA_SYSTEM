from typing import Dict, List, Set
import re
import logging

logger = logging.getLogger(__name__)

class TrafficSynonymExpander:
    
    def __init__(self):
       
        self._setup_bidirectional_mapping()
    
    def _setup_bidirectional_mapping(self):
      
        self.legal_to_colloquial = {
           
            "không mang theo giấy đăng ký xe": [
                "không mang cà vẹt xe",
                "quên mang cà vẹt xe",
                "quên cà vẹt xe"
            ],
            
            "không chứng nhận đăng ký xe": [
                "không có cà vẹt xe",
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
            
            
            "vi phạm nồng độ cồn": [
                "nồng độ cồn",
                "lái xe sau khi uống rượu bia",
                "điều khiển xe khi đã sử dụng rượu bia",
                "sử dụng rượu bia khi lái xe",
                "có cồn trong máu",
                "có cồn trong hơi thở",
                "đi nhậu"
            ],
            
            "nồng độ cồn vượt quá 80 miligam/100 mililít máu hoặc vượt quá 0,4 miligam/1 lít khí thở": [
                "1 thùng bia",
                "uống 1 thùng bia"
            ],
             
            "không có giấy phép lái xe": [
                "không có gplx",
                "không có bằng lái",
                "giấy phép lái xe đã bị trừ hết điểm"
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
            
            self.colloquial_to_legal[legal_term] = legal_term
            
            
            for term in colloquial_terms:
                self.colloquial_to_legal[term] = legal_term
        
        logger.info(f"Initialized with {len(self.legal_to_colloquial)} legal terms and {len(self.colloquial_to_legal)} total terms")
    
    def expand_query(self, query: str) -> str:
        
        expanded_query = query.lower()
        
       
        replacements = []
        
        
        sorted_terms = sorted(self.colloquial_to_legal.keys(), key=len, reverse=True)
        
        for colloquial_term in sorted_terms:
            
            pattern = r'\b' + re.escape(colloquial_term) + r'\b'
            
            
            if re.search(pattern, expanded_query):
                
                legal_term = self.colloquial_to_legal[colloquial_term]
                
                
                if colloquial_term != legal_term:
                    
                    expanded_query = re.sub(pattern, legal_term, expanded_query)
                    replacements.append(f"'{colloquial_term}' → '{legal_term}'")
                    logger.info(f"Replaced '{colloquial_term}' with '{legal_term}'")
        
        
        if replacements:
            logger.info(f"Made {len(replacements)} replacements: {', '.join(replacements)}")
        else:
            logger.info("No synonyms expanded in query")
        
        return expanded_query
    

