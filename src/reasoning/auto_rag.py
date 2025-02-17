from typing import Optional, Dict, Any, List
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore
import tiktoken

from config.config import ModelConfig, Domain
from retrieval.retriever import DocumentRetriever
from reasoning.prompts import SYSTEM_PROMPT, DOMAIN_VALIDATION_PROMPT

class AutoRAG:
    def __init__(
        self,
        model_config: ModelConfig,
        retriever: DocumentRetriever,
        current_domain: Domain,
        max_iterations: int = 2
    ):
        self.model_config = model_config
        self.retriever = retriever
        self.current_domain = current_domain
        self.max_iterations = max_iterations
        self.llm = self._setup_llm()
        self.prompt_template = PromptTemplate(template=SYSTEM_PROMPT)
        self.domain_prompt = PromptTemplate(template=DOMAIN_VALIDATION_PROMPT)
        self.tokenizer = tiktoken.encoding_for_model(model_config.llm_model)
        
    async def validate_domain(self, question: str) -> bool:
        """Validate if question matches current domain"""
        prompt = self.domain_prompt.format(question=question)
        response = await self.llm.acomplete(prompt)
        detected_domain = response.text.strip().lower()
        return detected_domain == self.current_domain.value
    
    async def get_answer(self, question: str) -> Dict[str, Any]:
        """Get answer for a question using Auto RAG with domain validation and iterations"""
        # Validate domain
        is_valid_domain = await self.validate_domain(question)
        if not is_valid_domain:
            return {
                "error": f"Question appears to be about {self.current_domain.value} domain. "
                        f"Please switch to the appropriate domain or rephrase your question.",
                "token_usage": {
                    "input_tokens": self._count_tokens(question),
                    "output_tokens": 0,
                    "total_tokens": self._count_tokens(question)
                }
            }
        
        # Initialize tracking variables
        iteration = 0
        accumulated_context = []
        total_input_tokens = 0
        total_output_tokens = 0
        search_history = []
        
        current_query = question
        while iteration < self.max_iterations:
            try:
                # Get documents for current query
                retrieved_docs = self.retriever.retrieve(current_query)
                
                # Skip iteration if no documents found
                if not retrieved_docs:
                    search_history.append({
                        "iteration": iteration + 1,
                        "query": current_query,
                        "num_docs": 0,
                        "error": "No documents found"
                    })
                    break
                
                accumulated_context.extend(retrieved_docs)
                
                # Format context from all retrieved documents
                context = self.retriever.get_formatted_context(accumulated_context)
                
                # Generate prompt and get response
                prompt = self.prompt_template.format(
                    question=question,  # Always use original question
                    context=context
                )
                
                input_tokens = self._count_tokens(prompt)
                response = await self.llm.acomplete(prompt)
                output_tokens = self._count_tokens(response.text)
                
                # Update token counts
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                # Parse response
                parsed_response = self._parse_response(response.text)
                
                # Track search iteration
                search_history.append({
                    "iteration": iteration + 1,
                    "query": current_query,
                    "num_docs": len(retrieved_docs),
                    "response": parsed_response
                })
                
                # Check if we have enough information
                if parsed_response["decision"].strip().lower() == "đã đủ thông tin":
                    # We have enough information, return final response
                    parsed_response["search_history"] = search_history
                    parsed_response["token_usage"] = {
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "total_tokens": total_input_tokens + total_output_tokens
                    }
                    return parsed_response
                
                # If we need more information and have a next query
                if parsed_response["next_query"]:
                    current_query = parsed_response["next_query"]
                    iteration += 1
                else:
                    # No next query provided but needs more info - break loop
                    break
                    
            except Exception as e:
                search_history.append({
                    "iteration": iteration + 1,
                    "query": current_query,
                    "error": str(e)
                })
                # Continue to next iteration if possible
                iteration += 1
                continue
        
        # If we exit the loop without finding enough information
        final_response = {
            "analysis": "Sau nhiều lần tìm kiếm, hệ thống vẫn chưa tìm thấy đủ thông tin.",
            "decision": "Không tìm thấy đủ thông tin",
            "final_answer": "Xin lỗi, tôi không tìm thấy đủ thông tin để trả lời câu hỏi của bạn.",
            "search_history": search_history,
            "token_usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens
            }
        }
        
        return final_response
    
    def _setup_llm(self) -> OpenAI:
        return OpenAI(
            model=self.model_config.llm_model,
            api_key=self.model_config.llm_api_key
        )
    
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
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