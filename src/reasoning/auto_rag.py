from typing import Optional, Dict, Any
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate

from config.config import ModelConfig
from retrieval.retriever import DocumentRetriever
from reasoning.prompts import SYSTEM_PROMPT

class AutoRAG:
    
    def __init__(
        self,
        model_config: ModelConfig,
        retriever: DocumentRetriever
    ):
        self.model_config = model_config
        self.retriever = retriever
        self.llm = self._self_llm()
        self.prompt_template = PromptTemplate(
            template=SYSTEM_PROMPT
        )
    
    
    def _self_llm(self) -> OpenAI:
        """Setup LLM with configured parameters"""
        return OpenAI(
            model=self.model_config.llm_model,
            api_key=self.model_config.llm_api_key
        )
    
    async def get_answer(self, question: str) -> Dict[str, Any]:
        """
        Get answer for a question using Auto RAG
        
        Args:
            question: User question
            
        Returns:
            Dictionary containing:
                - analysis: Analysis of available information
                - decision: Whether more information is needed
                - next_query: Follow-up query if needed
                - final_answer: Final answer if sufficient information
        """
        # Initial retrieval
        retrieved_docs = self.retriever.retrieve(question)
        context = self.retriever.get_formatted_context(retrieved_docs)
        
        # Generate prompt
        prompt = self.system_prompt.format(
            question=question,
            context=context
        )
        
        # Get LLM response
        response = await self.llm.acomplete(prompt)
        parsed_response = self._parse_response(response.text)
        
        return parsed_response
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse structured response from LLM"""
        lines = response.strip().split("\n")
        parsed = {
            "analysis": "",
            "decision": "",
            "next_query": None,
            "final_answer": None
        }
        
        current_section = None
        for line in lines:
            if line.startswith("Phân tích:"):
                current_section = "analysis"
                parsed["analysis"] = line.replace("Phân tích:", "").strip()
            elif line.startswith("Quyết định:"):
                current_section = "decision"
                parsed["decision"] = line.replace("Quyết định:", "").strip()
            elif line.startswith("Truy vấn tiếp theo:"):
                current_section = "next_query"
                parsed["next_query"] = line.replace("Truy vấn tiếp theo:", "").strip()
            elif line.startswith("Câu trả lời cuối cùng:"):
                current_section = "final_answer"
                parsed["final_answer"] = line.replace("Câu trả lời cuối cùng:", "").strip()
            elif current_section and line.strip():
                parsed[current_section] += " " + line.strip()
                
        return parsed
        