from typing import Optional, Dict, Any, List
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore
import tiktoken

from config.config import ModelConfig
from retrieval.retriever import DocumentRetriever
from reasoning.prompts import SYSTEM_PROMPT

class AutoRAG:
    
    def __init__(
        self,
        model_config: ModelConfig,
        retriever: DocumentRetriever,
        max_iterations: int = 3  # Maximum number of search iterations
    ):
        self.model_config = model_config
        self.retriever = retriever
        self.llm = self._setup_llm()
        self.prompt_template = PromptTemplate(
            template=SYSTEM_PROMPT
        )
        self.tokenizer = tiktoken.encoding_for_model(model_config.llm_model)
        self.max_iterations = max_iterations
        
    def _setup_llm(self) -> OpenAI:
        """Setup LLM with configured parameters"""
        return OpenAI(
            model=self.model_config.llm_model,
            api_key=self.model_config.llm_api_key
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count number of tokens in text"""
        return len(self.tokenizer.encode(text))
    
    async def get_answer(self, question: str) -> Dict[str, Any]:
        """
        Get answer for a question using Auto RAG (single iteration)
        
        Args:
            question: User question
            
        Returns:
            Dictionary containing:
                - analysis: Analysis of available information
                - decision: Whether more information is needed
                - next_query: Follow-up query if needed
                - final_answer: Final answer if sufficient information
                - token_usage: Dictionary with input and output token counts
        """
        # Initial retrieval
        retrieved_docs = self.retriever.retrieve(question)
        context = self.retriever.get_formatted_context(retrieved_docs)
        
        # Generate prompt
        prompt = self.prompt_template.format(
            question=question,
            context=context
        )
        
        # Count input tokens
        input_tokens = self._count_tokens(prompt)
        
        # Get LLM response
        response = await self.llm.acomplete(prompt)
        
        # Count output tokens
        output_tokens = self._count_tokens(response.text)
        
        # Parse response
        parsed_response = self._parse_response(response.text)
        
        # Add token usage information
        parsed_response["token_usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
        
        return parsed_response
    
    async def get_answer_iterative(self, question: str) -> Dict[str, Any]:
        """
        Get answer using iterative search process when more information is needed
        
        Args:
            question: Initial user question
            
        Returns:
            Dictionary containing:
                - all_analyses: List of all analyses performed
                - all_contexts: List of all contexts used
                - iterations: Number of iterations performed
                - final_answer: Final answer found
                - token_usage: Cumulative token usage across all iterations
        """
        iteration = 0
        current_query = question
        all_analyses = []
        all_contexts = []
        cumulative_tokens = {"input_tokens": 0, "output_tokens": 0}
        
        while iteration < self.max_iterations:
            # Get results for current query
            retrieved_docs = self.retriever.retrieve(current_query)
            current_context = self.retriever.get_formatted_context(retrieved_docs)
            all_contexts.append(current_context)
            
            # Generate prompt with all accumulated context
            combined_context = "\n\n---\n\n".join(all_contexts)
            prompt = self.prompt_template.format(
                question=question,  # Always use original question
                context=combined_context
            )
            
            # Get and parse LLM response
            input_tokens = self._count_tokens(prompt)
            response = await self.llm.acomplete(prompt)
            output_tokens = self._count_tokens(response.text)
            
            # Update token counts
            cumulative_tokens["input_tokens"] += input_tokens
            cumulative_tokens["output_tokens"] += output_tokens
            
            # Parse response
            parsed_response = self._parse_response(response.text)
            all_analyses.append(parsed_response["analysis"])
            
            # Check if we need more information
            if "Cần thêm thông tin" not in parsed_response["decision"]:
                return {
                    "all_analyses": all_analyses,
                    "all_contexts": all_contexts,
                    "iterations": iteration + 1,
                    "final_answer": parsed_response["final_answer"],
                    "token_usage": {
                        "input_tokens": cumulative_tokens["input_tokens"],
                        "output_tokens": cumulative_tokens["output_tokens"],
                        "total_tokens": sum(cumulative_tokens.values())
                    }
                }
            
            # If we need more info but have no next query, stop
            if not parsed_response["next_query"]:
                return {
                    "all_analyses": all_analyses,
                    "all_contexts": all_contexts,
                    "iterations": iteration + 1,
                    "final_answer": "Không tìm thấy đủ thông tin để trả lời câu hỏi.",
                    "token_usage": {
                        "input_tokens": cumulative_tokens["input_tokens"],
                        "output_tokens": cumulative_tokens["output_tokens"],
                        "total_tokens": sum(cumulative_tokens.values())
                    }
                }
            
            # Update for next iteration
            current_query = parsed_response["next_query"]
            iteration += 1
        
        # If we hit max iterations
        return {
            "all_analyses": all_analyses,
            "all_contexts": all_contexts,
            "iterations": self.max_iterations,
            "final_answer": "Đã đạt giới hạn số lần tìm kiếm. Không tìm được câu trả lời đầy đủ.",
            "token_usage": {
                "input_tokens": cumulative_tokens["input_tokens"],
                "output_tokens": cumulative_tokens["output_tokens"],
                "total_tokens": sum(cumulative_tokens.values())
            }
        }
    
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