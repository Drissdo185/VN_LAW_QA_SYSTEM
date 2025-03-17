from typing import Dict
from traffic_synonyms import TrafficSynonyms

class QuestionProcess:
    def __init__(self):
        self.traffic_synonyms = TrafficSynonyms()
    
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
        term_conversion = self.traffic_synonyms.change_to_specific_term(question)
        
        processed_question = term_conversion["correct_term"]
        replacements = term_conversion["replacements"]
        
        question_type = self.question_type(processed_question)
        
        return {
            "original_question": question,
            "processed_question": processed_question,
            "replacements": replacements,
            "question_type": question_type
        }