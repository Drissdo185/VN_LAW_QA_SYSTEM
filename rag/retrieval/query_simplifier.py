import logging
from typing import Dict, Any

from llama_index.llms.openai import OpenAI
from config.config import ModelConfig, LLMProvider
from llm.vllm_client import VLLMClient
from llm.ollama_client import OllamaClient
from retrieval.traffic_synonyms import TrafficSynonymExpander
import json
import re

logger = logging.getLogger(__name__)

class QuerySimplifier:
    """
    A component that standardizes user queries before retrieval to improve relevance.
    It removes noise and extracts key legal concepts related to traffic laws.
    """
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.llm = self._setup_llm()
        self.synonym_expander = TrafficSynonymExpander()
        
    def _setup_llm(self):
        """Set up the LLM based on the provider configuration"""
        if self.model_config.llm_provider == LLMProvider.OPENAI:
            return OpenAI(
                model=self.model_config.openai_model,
                api_key=self.model_config.openai_api_key
            )
        elif self.model_config.llm_provider == LLMProvider.VLLM:
            
                client = VLLMClient.from_config(self.model_config.vllm_config)
        
                return client
        else:
            raise ValueError(f"Unsupported LLM provider: {self.model_config.llm_provider}")
    

    async def simplify_query(self, original_query: str) -> Dict[str, Any]:
        """
        Standardize a user query by extracting relevant legal concepts and removing noise.
        
        Args:
            original_query: The original user query, potentially with noise
            
        Returns:
            Dictionary containing standardized query and extraction metadata
        """
        logger.info(f"Standardizing query: {original_query}")
        
        
        legal_terms = self.synonym_expander.get_legal_terms(original_query)
        logger.info(f"Identified legal terms: {legal_terms}")
        
        
        legal_terms_hint = ""
        if legal_terms:
            legal_terms_hint = f"""
            Các thuật ngữ pháp lý được nhận diện trong câu hỏi:
            {', '.join(legal_terms)}
            """
        
        
        prompt = f"""
        Bạn là trợ lý hỗ trợ đơn giản hóa các câu hỏi về luật giao thông Việt Nam. 
        Hãy phân tích câu hỏi của người dùng và đơn giản hóa thành một câu truy vấn chuẩn hóa,
        tập trung vào các yếu tố pháp lý liên quan đến vi phạm giao thông.
        
        Câu hỏi gốc: {original_query}
        
        {legal_terms_hint}
        
        QUAN TRỌNG: Câu truy vấn chuẩn hóa PHẢI theo đúng định dạng sau:
        "Đối với [vehicle_type], vi phạm [loại vi phạm] sẽ bị xử phạt [loại hình phạt nếu có đề cập] như thế nào?"
        
        Khi nói đến "vượt đèn đỏ", hãy dùng thuật ngữ pháp lý: "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"
        
        
        Quy tắc:
        1. Nếu người dùng không đề cập cụ thể loại hình phạt, bỏ qua phần [loại hình phạt] trong câu truy vấn
        2. Nếu người dùng đề cập cụ thể (như tiền phạt, trừ điểm), đưa vào câu truy vấn
        3. Sử dụng "xe máy" hoặc "ô tô" làm vehicle_type khi có thể. Nếu không rõ, dùng "phương tiện"
        4. Luôn bảo toàn chi tiết cụ thể của vi phạm (ví dụ: tốc độ, nồng độ cồn)
        5. Luôn sử dụng thuật ngữ pháp lý chính thức cho các vi phạm
        
        Hãy trả về kết quả theo định dạng JSON với các trường sau:
        - standardized_query: Câu truy vấn đã được chuẩn hóa theo mẫu trên
        - violations: Danh sách các loại vi phạm được nhắc đến (nồng độ cồn, không mang giấy tờ, v.v)
        - vehicle_type: Loại phương tiện (ô tô, xe máy, v.v)
        - penalty_types: Loại hình phạt đang được hỏi (tiền phạt, trừ điểm, tước giấy phép lái xe, v.v)
        
        Chỉ trả về JSON, không trả lời gì thêm.
        """
        
        try:
            response = await self.llm.acomplete(prompt)
            simplified_result = self._parse_simplifier_response(response.text)
            simplified_result = self._parse_simplifier_response(response.text)
            
          
            standardized_query = simplified_result.get('standardized_query')
            if not standardized_query:
               
                vehicle_type = simplified_result.get('vehicle_type', 'phương tiện')
                violations_str = ', '.join(simplified_result.get('violations', ['vi phạm giao thông']))
                standardized_query = f"Đối với {vehicle_type}, vi phạm {violations_str} sẽ bị xử phạt như thế nào?"
                simplified_result['standardized_query'] = standardized_query
            
            logger.info(f"Successfully standardized query: '{standardized_query}'")
            logger.info(f"Detected violations: {simplified_result.get('violations')}")
            logger.info(f"Vehicle type: {simplified_result.get('vehicle_type')}")
            logger.info(f"Penalty types: {simplified_result.get('penalty_types')}")
            
            return {
                "original_query": original_query,
                "standardized_query": standardized_query,
                "metadata": {
                    "violations": simplified_result.get("violations", []),
                    "vehicle_type": simplified_result.get("vehicle_type"),
                    "penalty_types": simplified_result.get("penalty_types", [])
                }
            }
            
            
        except Exception as e:
            logger.error(f"Error standardizing query: {str(e)}")
            return {
                "original_query": original_query,
                "standardized_query": original_query,
                "metadata": {
                    "violations": [],
                    "vehicle_type": None,
                    "penalty_types": [],
                    "error": str(e)
                }
            }
    
    def _parse_simplifier_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the JSON response from the LLM.
        If parsing fails, extract data using simpler methods.
        """
        
        try:
            json_pattern = r'```json\s*([\s\S]*?)\s*```|{\s*"[\s\S]*?}'
            json_match = re.search(json_pattern, response_text)
            
            if json_match:
                json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
                
                json_str = json_str.replace('```json', '').replace('```', '')
                return json.loads(json_str)
            
           
            standardized_query = None
            match = re.search(r'standardized_query[": ]+(.*?)[\n",}]', response_text)
            if match:
                standardized_query = match.group(1).strip()
            
            violations = []
            violations_match = re.search(r'violations[": ]+\[(.*?)\]', response_text, re.DOTALL)
            if violations_match:
                violations_str = violations_match.group(1)
                violations = [v.strip(' "\'') for v in re.findall(r'"([^"]*)"|\S+', violations_str)]
            
            vehicle_type = None
            vehicle_match = re.search(r'vehicle_type[": ]+(.*?)[\n",}]', response_text)
            if vehicle_match:
                vehicle_type = vehicle_match.group(1).strip(' "\'')
            
            penalty_types = []
            penalty_match = re.search(r'penalty_types[": ]+\[(.*?)\]', response_text, re.DOTALL)
            if penalty_match:
                penalty_str = penalty_match.group(1)
                penalty_types = [p.strip(' "\'') for p in re.findall(r'"([^"]*)"|\S+', penalty_str)]
            
            return {
                "standardized_query": standardized_query or response_text.strip(),
                "violations": violations,
                "vehicle_type": vehicle_type,
                "penalty_types": penalty_types
            }
        except Exception as e:
            logger.error(f"Error parsing simplifier response: {str(e)}")
            logger.debug(f"Original response: {response_text}")
            return {
                "standardized_query": response_text.strip(),
                "violations": [],
                "vehicle_type": None,
                "penalty_types": []
            }