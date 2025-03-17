from typing import Dict
from traffic_synonyms import TrafficSynonymExpander
from reasoning.prompts import QUERY_STANDARDIZATION_VIOLATION
import openai

class QuestionHandle:
    
    def __init__(self):
        self.traffic_synonums = TrafficSynonymExpander()
        
        
    def question_type(self, question):
            
        penalty_keywords = [
            'xử phạt', 'bị phạt', 'bị gì', 'xử lý', 'nộp phạt', 
            'phạt tiền', 'phạt bao nhiêu', 'mức phạt', 'tước giấy phép',
            'trừ điểm', 'phạt như thế nào', 'bị phạt gì', 'hình thức xử phạt',
            'xử phạt hành chính', 'tiền phạt', 'phạt hành chính', 'bị tịch thu',
            'thu giữ', 'tạm giữ', 'hình phạt'
        ]
        
        question_lower = question.lower()
        
        for keyword in penalty_keywords:
            if keyword in question_lower:
                return 'violation_type'
        
        return 'general_information'
    
    def process_question(self, question):
        
        convert_to_standard_legal_forms = self.traffic_synonums.expand_query(question)
        
        q_type = self.question_type(question)
        
        if q_type == "violation_type":
            standardzation_prompt = QUERY_STANDARDIZATION_VIOLATION.format(question = convert_to_standard_legal_forms)
            llm_reponse = self.call_llm(standardzation_prompt)
    
    
    def call_llm(self, prompt: str) -> str:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs valid JSON as requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=128
            )
            
            content = response.choices[0].message.content
            
            if content.startswith("```") and content.endswith("```"):
                content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
            
            if content.startswith("```json"):
                content = content.replace("```json", "", 1)
                
            if content.endswith("```"):
                content = content[:-3]
                
            content = content.strip()
            
            return content
        except Exception as e:
            return f"Exception during OpenAI query: {str(e)}"