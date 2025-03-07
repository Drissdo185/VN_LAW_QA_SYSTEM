from typing import Dict, Any
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
import tiktoken
import logging
import re

from config.config import ModelConfig, LLMProvider
from retrieval.retriever import DocumentRetriever
from reasoning.prompts import SYSTEM_PROMPT
from llm.vllm_client import VLLMClient
from llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class AutoRAG:
    def __init__(
        self,
        model_config: ModelConfig,
        retriever: DocumentRetriever,
        max_iterations: int = 2
    ):
        self.model_config = model_config
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.llm = self._setup_llm()
        self.prompt_template = PromptTemplate(template=SYSTEM_PROMPT)
        self.tokenizer = self._setup_tokenizer()
        logger.info(f"Initialized Traffic AutoRAG with LLM provider: {model_config.llm_provider}")
        
    def _setup_llm(self):
        """Set up the LLM based on the provider configuration"""
        if self.model_config.llm_provider == LLMProvider.OPENAI:
            logger.info(f"Setting up OpenAI LLM with model: {self.model_config.openai_model}")
            return OpenAI(
                model=self.model_config.openai_model,
                api_key=self.model_config.openai_api_key
            )
        elif self.model_config.llm_provider == LLMProvider.VLLM:
            logger.info(f"Setting up vLLM client with model: {self.model_config.vllm_config.model_name}")
            try:
                client = VLLMClient.from_config(self.model_config.vllm_config)
               
                logger.info(f"Successfully initialized vLLM client with API URL: {client.api_url}")
                return client
            except Exception as e:
                logger.error(f"Error initializing vLLM client: {str(e)}")
                raise
        elif self.model_config.llm_provider == LLMProvider.OLLAMA:  
            logger.info(f"Setting up Ollama client with model: {self.model_config.ollama_config.model_name}")
            try:
                client = OllamaClient.from_config(self.model_config.ollama_config)
                
                logger.info(f"Successfully initialized Ollama client with API URL: {client.api_url}")
                return client
            except Exception as e:
                logger.error(f"Error initializing Ollama client: {str(e)}")
                raise
        else:
            raise ValueError(f"Unsupported LLM provider: {self.model_config.llm_provider}")

    # Also update the _setup_tokenizer method:
    def _setup_tokenizer(self):
        """Set up the tokenizer based on the LLM provider"""
        if self.model_config.llm_provider == LLMProvider.OPENAI:
            return tiktoken.encoding_for_model(self.model_config.openai_model)
        elif self.model_config.llm_provider == LLMProvider.VLLM:
            
            return tiktoken.get_encoding("cl100k_base")
        elif self.model_config.llm_provider == LLMProvider.OLLAMA:  
            
            return tiktoken.get_encoding("cl100k_base")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.model_config.llm_provider}")
    
    def _is_traffic_related(self, question: str) -> bool:
        """Check if question is related to traffic"""
       
        traffic_keywords = [
            'mũ bảo hiểm', 'giao thông', 'đường bộ', 'biển báo', 
            'luật giao thông', 'phạt', 'xe máy', 'ô tô', 'bằng lái',
            'giấy phép', 'nd168', 'nghị định', 'nồng độ cồn', 
            'vượt đèn đỏ', 'tốc độ', 'vạch kẻ đường', 'tai nạn',
            'xe', 'đường', 'đậu xe', 'đỗ xe', 'dừng xe', 'đăng kiểm',
            'giấy tờ', 'biển số', 'traffic', 'luật', 'quá tải'
        ]
        
        question_lower = question.lower()
        for keyword in traffic_keywords:
            if keyword in question_lower:
                logger.info(f"Question validated as traffic-related via keyword: {keyword}")
                return True
        
        return False
    
    async def get_answer(self, question: str) -> Dict[str, Any]:
        """Get answer for a traffic-related question using Auto RAG with iterations"""
        # Basic validation to check if question is related to traffic
        if not self._is_traffic_related(question):
            logger.warning(f"Question may not be traffic-related: {question}")
            return {
                "error": "Câu hỏi có vẻ không liên quan đến luật giao thông. Vui lòng đặt câu hỏi về luật giao thông.",
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
                print(prompt)
                
                
                
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
                    parsed_response["llm_provider"] = self.model_config.llm_provider
                    return parsed_response
                
                # If we need more information and have a next query
                if parsed_response["next_query"]:
                    current_query = parsed_response["next_query"]
                    iteration += 1
                else:
                    # No next query provided but needs more info - break loop
                    break
                    
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {str(e)}")
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
            "analysis": "Sau nhiều lần tìm kiếm, hệ thống vẫn chưa tìm thấy đủ thông tin về luật giao thông liên quan đến câu hỏi này.",
            "decision": "Không tìm thấy đủ thông tin",
            "final_answer": "Xin lỗi, tôi không tìm thấy đủ thông tin trong luật giao thông để trả lời câu hỏi của bạn.",
            "search_history": search_history,
            "token_usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens
            },
            "llm_provider": self.model_config.llm_provider
        }
        
        return final_response
    
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