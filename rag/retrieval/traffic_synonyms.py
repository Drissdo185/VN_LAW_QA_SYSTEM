from typing import Dict, List, Set

class TrafficSynonymExpander:
    """
    Mở rộng các truy vấn liên quan đến giao thông với danh sách từ đồng nghĩa
    và cụm từ liên quan trong luật giao thông Việt Nam.
    """
    
    def __init__(self):
       
        self.traffic_synonyms = {
           
            "vượt đèn đỏ": [
                "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
                "không chấp hành hiệu lệnh đèn giao thông",
                "vượt đèn tín hiệu",
                "vượt đèn tín hiệu giao thông",
                "vượt đèn",
                "không tuân thủ đèn tín hiệu giao thông",
                "bỏ qua tín hiệu đèn đỏ",
                "đi khi đèn đỏ",
                "vượt qua đèn đỏ",
                "chạy đèn đỏ"
            ],
            
           
            "nồng độ cồn": [
                "vi phạm nồng độ cồn",
                "lái xe sau khi uống rượu bia",
                "điều khiển xe khi đã sử dụng rượu bia",
                "sử dụng rượu bia khi lái xe",
                "có cồn trong máu",
                "có cồn trong hơi thở",
                "uống rượu bia khi lái xe",
                "nồng độ cồn trong máu",
                "nồng độ cồn trong hơi thở",
                "uống bia",
                "uống rượu",
                "uống cồn",
                "say xỉn",
                "say rượu",
                "say bia",
                "có men",
                "có men bia",
                "men rượu",
                "đã uống bia",
                "đã uống rượu",
                "uống vài lon",
                "uống vài chai",
                "uống một thùng",
                "nhậu",
                "đi nhậu",
                "nhậu xong",
                "bia rượu",
                "bia",
                "rượu"
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
            
           
            "không có giấy phép lái xe": [
                "không mang giấy phép lái xe",
                "không có bằng lái",
                "không mang bằng lái",
                "không có gplx",
                "không mang gplx",
                "thiếu giấy phép lái xe",
                "không xuất trình được giấy phép lái xe",
                "vi phạm về giấy phép lái xe",
                "chưa có giấy phép lái xe"
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
        """
        Mở rộng truy vấn với các từ đồng nghĩa liên quan đến luật giao thông.
        
        Args:
            query: Truy vấn gốc của người dùng
            
        Returns:
            Truy vấn đã được mở rộng với các từ đồng nghĩa
        """
        expanded_query = query
        query_lower = query.lower()
        
        # Kiểm tra từng nhóm từ đồng nghĩa
        for main_term, synonyms in self.traffic_synonyms.items():
            # Nếu thuật ngữ chính hoặc một đồng nghĩa có trong truy vấn
            if main_term in query_lower or any(syn in query_lower for syn in synonyms):
                # Đã có một thuật ngữ trong truy vấn, thêm thuật ngữ chính vào
                if main_term not in query_lower:
                    expanded_query += f" {main_term}"
                
                # Thêm các đồng nghĩa quan trọng (hạn chế để không làm truy vấn quá rối)
                for important_syn in synonyms[:2]:  # Chỉ thêm 2 đồng nghĩa quan trọng nhất
                    if important_syn not in query_lower:
                        expanded_query += f" {important_syn}"
        
        return expanded_query
    
    def get_legal_terms(self, query: str) -> List[str]:
        """
        Xác định các thuật ngữ pháp lý có trong truy vấn
        
        Args:
            query: Truy vấn của người dùng
            
        Returns:
            Danh sách các thuật ngữ pháp lý được tìm thấy
        """
        legal_terms = []
        query_lower = query.lower()
        
        for main_term, synonyms in self.traffic_synonyms.items():
            all_terms = [main_term] + synonyms
            for term in all_terms:
                if term in query_lower and main_term not in legal_terms:
                    legal_terms.append(main_term)
                    break
        
        return legal_terms