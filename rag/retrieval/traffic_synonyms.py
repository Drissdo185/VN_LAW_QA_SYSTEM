from typing import Dict, List, Set
import re
import logging

logger = logging.getLogger(__name__)

class TrafficSynonymExpander:
    
    def __init__(self):
        # Dictionary of traffic-related terms and their synonyms
        self.traffic_synonyms = {
            # Driving violations
            "vượt đèn đỏ": [
                "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
                "không chấp hành hiệu lệnh đèn giao thông",
                "vượt đèn tín hiệu",
            ],
            
            # Alcohol-related violations
            "nồng độ cồn": [
                "vi phạm nồng độ cồn",
                "lái xe sau khi uống rượu bia",
                "điều khiển xe khi đã sử dụng rượu bia",
                "sử dụng rượu bia khi lái xe",
                "có cồn trong máu",
                "có cồn trong hơi thở",
            ],
            
            # Added specific alcohol amount term
            "1 thùng bia": [
                "nồng độ cồn vượt quá 80 miligam/100 mililít máu hoặc vượt quá 0,4 miligam/1 lít khí thở"
            ],
            
            "uống 1 thùng bia": [
                "nồng độ cồn vượt quá 80 miligam/100 mililít máu hoặc vượt quá 0,4 miligam/1 lít khí thở"
            ],
            
            # License-related violations
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
            
            # Vehicle registration violations
            "không mang cà vẹt xe":[
                "không mang theo giấy đăng ký xe"
            ],
            
            "không có cà vẹt xe":[
                "không có giấy đăng ký xe"
            ],
            
            "chưa đăng ký giấy tờ": [
                "không có giấy đăng ký xe",
                "không đăng ký xe",
                "sử dụng xe không có giấy tờ hợp lệ"
            ],
            
            # Parking violations
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
           
            # Safety equipment violations
            "không đội mũ bảo hiểm": [
                "không sử dụng mũ bảo hiểm",
                "thiếu mũ bảo hiểm",
                "vi phạm quy định về mũ bảo hiểm",
                "không đội nón bảo hiểm",
                "không đội mũ"
            ],
            
            # Road sign violations
            "không chấp hành biển báo": [
                "vi phạm biển báo",
                "không tuân thủ biển báo",
                "bỏ qua biển báo",
                "không chấp hành biển hiệu",
                "không tuân thủ hiệu lệnh",
                "không tuân thủ biển hiệu đường bộ"
            ],
            
            # Lane violations
            "đi không đúng làn đường": [
                "vi phạm làn đường",
                "không đi đúng phần đường",
                "đi sai làn",
                "lấn làn",
                "chạy sai làn đường",
                "không tuân thủ làn đường"
            ],
            
            # Accident-related
            "gây tai nạn": [
                "gây tai nạn giao thông",
                "làm xảy ra tai nạn",
                "va chạm giao thông",
                "gây va chạm",
                "gây ra tai nạn",
                "để xảy ra tai nạn"
            ],
            
            # Dangerous overtaking
            "vượt ẩu": [
                "vượt không đúng quy định",
                "vượt xe sai quy định",
                "vượt xe không an toàn",
                "lấn làn vượt xe",
                "vượt xe nguy hiểm",
                "vượt xe trái phép"
            ],
            
            # Drug-related violations
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
        Expand query with traffic-related synonyms
        
        This improved version:
        1. Uses regex to find terms even within larger phrases
        2. Handles partial matches better
        3. Applies multiple replacements if needed
        
        Args:
            query: Original user query
            
        Returns:
            Expanded query with traffic terms replaced by their standard legal forms
        """
        expanded_query = query.lower()
        
        # Create a list to track replacements so we can log them
        replacements = []
        
        # Sort terms by length (descending) to ensure longer phrases are matched first
        sorted_terms = sorted(self.traffic_synonyms.keys(), key=len, reverse=True)
        
        for main_term in sorted_terms:
            # Create pattern that matches the term as a whole word or phrase
            pattern = r'\b' + re.escape(main_term) + r'\b'
            
            # If this term is found in the query
            if re.search(pattern, expanded_query):
                # Get the synonyms for this term
                synonyms = self.traffic_synonyms[main_term]
                if synonyms:
                    # Replace with first synonym (most official form)
                    replacement = synonyms[0]
                    original_text = main_term
                    expanded_query = re.sub(pattern, replacement, expanded_query)
                    replacements.append(f"'{original_text}' → '{replacement}'")
                    logger.info(f"Replaced '{original_text}' with '{replacement}'")
        
        # Log all replacements made
        if replacements:
            logger.info(f"Made {len(replacements)} replacements: {', '.join(replacements)}")
        else:
            logger.info("No synonyms expanded in query")
        
        return expanded_query
    
    def get_legal_terms(self, query: str) -> List[str]:
        """
        Extract legal terms mentioned in the query.
        
        Args:
            query: The user query
            
        Returns:
            List of legal terms found in the query
        """
        legal_terms = []
        query_lower = query.lower()
        
        for main_term, synonyms in self.traffic_synonyms.items():
            all_terms = [main_term] + synonyms
            for term in all_terms:
                # Use regex for better word boundary matching
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, query_lower) and main_term not in legal_terms:
                    legal_terms.append(main_term)
                    break
        
        if legal_terms:
            logger.info(f"Found legal terms in query: {', '.join(legal_terms)}")
        else:
            logger.info("No legal terms found in query")
            
        return legal_terms