from typing import Dict, List, Set

class TrafficSynonymExpander:
    
    def __init__(self):
       
        self.traffic_synonyms = {
           
            "vượt đèn đỏ": [
                "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
                "không chấp hành hiệu lệnh đèn giao thông",
                "vượt đèn tín hiệu",
            ],
            
           
            "nồng độ cồn": [
                "vi phạm nồng độ cồn",
                "lái xe sau khi uống rượu bia",
                "điều khiển xe khi đã sử dụng rượu bia",
                "sử dụng rượu bia khi lái xe",
                "có cồn trong máu",
                "có cồn trong hơi thở",
            ],
            
            "uống 1 thùng bia":[
                "nồng độ cồn vượt quá 80 miligam/100 mililít máu hoặc vượt quá 0,4 miligam/1 lít khí thở"
            ],
            
            "không có giấy phép lái xe": [
                "không có gplx",
                "không mang bằng lái",
                ],
                
            
           
            "chạy quá tốc độ": [
                "vượt quá tốc độ quy định",
                "vi phạm tốc độ",
                "điều khiển xe chạy quá tốc độ",
                "lái xe quá tốc độ cho phép",
                "chạy nhanh hơn tốc độ quy định",
                "vượt tốc độ tối đa",
                "phóng nhanh",
                "chạy nhanh quá mức"
            ],
            
           
            "không mang giấy phép lái xe": [
                "không mang gplx",
                "không mang bằng lái",
            ],
            
            "không có bằng lái":[
                "không có gplx", 
                "giấy phép lái xe đã bị trừ hết điểm"],
            
            "không mang bằng lái":[
                "không mang gplx",
                "không mang bằng lái"],
            
            "giấy phép lái xe đã bị trừ hết điểm":["không có gplx"],
            
            "không mang cà vẹt xe":[
                "không mang theo giấy đăng ký xe"
            ],
            
            
            "không có cà vẹt xe":[
                "không có theo giấy đăng ký xe"
            ],
           
            "đỗ xe sai quy định": [
                "đậu xe sai quy định",
                "dừng xe sai quy định",
                "đỗ xe không đúng nơi quy định",
                "đậu xe không đúng nơi quy định",
                "đỗ xe trái phép",
                "đậu xe trái phép",
                "vi phạm quy định về đỗ xe",
                "đỗ xe nơi cấm đỗ",
                "dừng xe nơi cấm dừng"
            ],
           
           
            "không đội mũ bảo hiểm": [
                "không sử dụng mũ bảo hiểm",
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
            
          
            "đi không đúng làn đường": [
                "vi phạm làn đường",
                "không đi đúng phần đường",
                "đi sai làn",
                "lấn làn",
                "chạy sai làn đường",
                "không tuân thủ làn đường"
            ],
            
           
            "gây tai nạn": [
                "gây tai nạn giao thông",
                "làm xảy ra tai nạn",
                "va chạm giao thông",
                "gây va chạm",
                "gây ra tai nạn",
                "để xảy ra tai nạn"
            ],
            
            
            "vượt ẩu": [
                "vượt không đúng quy định",
                "vượt xe sai quy định",
                "vượt xe không an toàn",
                "lấn làn vượt xe",
                "vượt xe nguy hiểm",
                "vượt xe trái phép"
            ],
            
            
            "sử dụng ma túy": [
                "sử dụng chất kích thích",
                "sử dụng chất cấm",
                "điều khiển xe sau khi sử dụng ma túy",
                "lái xe khi đã sử dụng chất ma túy",
                "chất ma túy trong cơ thể",
                "dương tính với ma túy"
            ]
        }
    
    def expand_query(self, query: str) -> str:
    
        expanded_query = query
        query_lower = query.lower()
        
       
        for main_term, synonyms in self.traffic_synonyms.items():
           
            if main_term in query_lower or any(syn in query_lower for syn in synonyms):
               
                expanded_query = query_lower.replace(main_term, " ".join(synonyms[:3]))
        
        return expanded_query
    
    def get_legal_terms(self, query: str) -> List[str]:
        legal_terms = []
        query_lower = query.lower()
        
        for main_term, synonyms in self.traffic_synonyms.items():
            all_terms = [main_term] + synonyms
            for term in all_terms:
                if term in query_lower and main_term not in legal_terms:
                    legal_terms.append(main_term)
                    break
        
        return legal_terms