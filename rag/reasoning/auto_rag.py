from typing import Dict, Any, Optional, List
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore
import tiktoken
import logging
import re
from pyvi import ViTokenizer

from config.config import ModelConfig, LLMProvider
from retrieval.retriever import DocumentRetriever
from retrieval.search_pipline import SearchPipeline
from reasoning.prompts import FULL_PROMPT, DECISION_PROMPT, OUTPUT_FORMAT, FINAL_EFFORT_PROMPT
from llm.vllm_client import VLLMClient
from llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class AutoRAG:
    def __init__(
        self,
        model_config: ModelConfig,
        retriever: DocumentRetriever,
        search_pipeline: Optional[SearchPipeline] = None,
        max_iterations: int = 3
    ):
        self.model_config = model_config
        self.retriever = retriever
        self.search_pipeline = search_pipeline
        self.max_iterations = max_iterations
        self.llm = self._setup_llm()
        self.prompt_template = PromptTemplate(template=FULL_PROMPT)
        self.decision_prompt_template = PromptTemplate(template=DECISION_PROMPT)
        self.final_effort_prompt_template = PromptTemplate(template=FINAL_EFFORT_PROMPT)
        self.tokenizer = self._setup_tokenizer()
    
    def _setup_llm(self):
        """Set up the LLM based on the provider configuration"""
        if self.model_config.llm_provider == LLMProvider.OPENAI:
            logger.info(f"Setting up OpenAI LLM: {self.model_config.openai_model}")
            return OpenAI(model=self.model_config.openai_model, api_key=self.model_config.openai_api_key)
        elif self.model_config.llm_provider == LLMProvider.VLLM:
            logger.info(f"Setting up vLLM: {self.model_config.vllm_config.model_name}")
            try:
                return VLLMClient.from_config(self.model_config.vllm_config)
            except Exception as e:
                logger.error(f"vLLM initialization error: {str(e)}")
                raise
        elif self.model_config.llm_provider == LLMProvider.OLLAMA:
            logger.info(f"Setting up Ollama: {self.model_config.ollama_config.model_name}")
            try:
                return OllamaClient.from_config(self.model_config.ollama_config)
            except Exception as e:
                logger.error(f"Ollama initialization error: {str(e)}")
                raise
        else:
            raise ValueError(f"Unsupported LLM provider: {self.model_config.llm_provider}")

    def _setup_tokenizer(self):
        if self.model_config.llm_provider == LLMProvider.OPENAI:
            return tiktoken.encoding_for_model(self.model_config.openai_model)
        elif self.model_config.llm_provider in [LLMProvider.VLLM, LLMProvider.OLLAMA]:
            return tiktoken.get_encoding("cl100k_base")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.model_config.llm_provider}")
    
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        lines = response.strip().split("\n")
        parsed = {
            "analysis": "",
            "decision": "",
            "next_query": None,
            "final_answer": None,
            "query_type": None  
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
                parsed["final_answer"] = ""
            elif current_section == "final_answer" and line is not None:
                if parsed["final_answer"]:
                    parsed["final_answer"] += "\n" + line
                else:
                    parsed["final_answer"] = line
            elif current_section and current_section != "final_answer" and line:
                parsed[current_section] += " " + line
        
        # Determine query type based on content
        if parsed["analysis"] and any(keyword in parsed["analysis"].lower() for keyword in 
                                ["vi phạm", "xử phạt", "phạt tiền", "trừ điểm", "tước giấy phép"]):
            parsed["query_type"] = "violation_penalty"
        else:
            parsed["query_type"] = "general_information"
        
        # Format next query if needed
        if parsed["next_query"]:
            if parsed["query_type"] == "violation_penalty" and "Đối với" not in parsed["next_query"]:
                original_query = parsed["next_query"]
                vehicle_type = self._extract_vehicle_type(original_query)
                violation_type = self._extract_violation_type(original_query)
                penalty_types = self._extract_penalty_types(original_query)
                
                penalty_phrase = ""
                if penalty_types:
                    penalty_phrase = f"bị xử phạt {', '.join(penalty_types)} như thế nào?"
                else:
                    penalty_phrase = "bị xử phạt như thế nào?"
                
                parsed["next_query"] = f"Đối với {vehicle_type}, vi phạm {violation_type} sẽ {penalty_phrase}"
            
            elif parsed["query_type"] == "general_information" and "Quy định về" not in parsed["next_query"]:
                original_query = parsed["next_query"]
                topic = self._extract_topic(original_query)
                vehicle_type = self._extract_vehicle_type(original_query)
                
                if vehicle_type and vehicle_type != "phương tiện":
                    parsed["next_query"] = f"Quy định về {topic} đối với {vehicle_type} là gì?"
                else:
                    parsed["next_query"] = f"Quy định về {topic} là gì?"
            
            logger.info(f"Next query: {parsed['next_query']}")
                
        return parsed
    
    def _extract_vehicle_type(self, query: str) -> str:
        if "xe máy" in query.lower() or "mô tô" in query.lower():
            return "xe máy"
        elif "ô tô" in query.lower() or "xe hơi" in query.lower():
            return "ô tô"
        else:
            return "phương tiện"

    def _extract_topic(self, query: str) -> str:
        topic_keywords = [
            "độ tuổi", "tuổi tối thiểu", "điều kiện", "yêu cầu", "thủ tục", 
            "giấy phép lái xe", "bằng lái", "hạng", "loại bằng", "đổi bằng",
            "thời hạn", "gia hạn", "cấp lại", "học", "thi", "lệ phí", "chi phí"
        ]
        
        for keyword in topic_keywords:
            if keyword in query.lower():
                parts = query.lower().split(keyword)
                if len(parts) > 1:
                    before = parts[0].strip().split()[-3:] if parts[0].strip() else []
                    after = parts[1].strip().split()[:3] if parts[1].strip() else []
                    context_words = before + [keyword] + after
                    return " ".join(context_words)
        
        if "độ tuổi" in query.lower() or "tuổi" in query.lower():
            return "độ tuổi lái xe"
        elif "bằng lái" in query.lower() or "giấy phép" in query.lower():
            return "giấy phép lái xe"
        elif "điều kiện" in query.lower() or "yêu cầu" in query.lower():
            return "điều kiện cấp giấy phép lái xe"
        elif "thủ tục" in query.lower() or "làm bằng" in query.lower():
            return "thủ tục cấp giấy phép lái xe"
        
        return "quy định giao thông đường bộ"

    def _extract_violation_type(self, query: str) -> str:
        violation = query
        if "thông tin về" in query:
            violation = query.split("thông tin về")[1].strip()
        if "cho hành vi" in query:
            violation = query.split("cho hành vi")[1].strip()
        if "sẽ bị" in violation:
            violation = violation.split("sẽ bị")[0].strip()
        return violation

    def _extract_penalty_types(self, query: str) -> List[str]:
        penalty_types = []
        if "tiền" in query.lower():
            penalty_types.append("tiền")
        if "tịch thu" in query.lower() or "thu giữ" in query.lower():
            penalty_types.append("tịch thu")
        if "trừ điểm" in query.lower():
            penalty_types.append("trừ điểm")
        if "tước giấy phép" in query.lower() or "tước bằng" in query.lower():
            penalty_types.append("tước giấy phép lái xe")
        return penalty_types
    
    async def get_answer(self, question) -> Dict[str, Any]:
        """
        Get answer for a traffic-related question using Auto RAG with iterations
        
        Args:
            question: Có thể là một chuỗi truy vấn hoặc đối tượng simplification_result
        """
        # Xử lý khi đầu vào là dictionary từ query_simplifier
        if isinstance(question, dict):
            # Trích xuất thông tin từ dictionary
            query_info = {
                "original_query": question.get("original_query", ""),
                "standardized_query": question.get("standardized_query", ""),
                "metadata": question.get("metadata", {})
            }
            
            # Sử dụng standardized_query cho việc tìm kiếm
            current_query = question.get("standardized_query", "")
            if not current_query:
                return {
                    "error": "Không thể trích xuất câu truy vấn chuẩn hóa",
                    "token_usage": {
                        "input_tokens": 0, 
                        "output_tokens": 0,
                        "total_tokens": 0
                    }
                }
        else:
            # Nếu đầu vào là chuỗi, sử dụng trực tiếp
            current_query = question
            query_info = None
        
        # Khởi tạo biến lưu trữ
        iteration = 0
        accumulated_docs = []
        total_input_tokens = 0
        total_output_tokens = 0
        search_history = []
        
        # Bắt đầu vòng lặp RAG
        while iteration < self.max_iterations:
            try:
                # Retrieve documents
                logger.info(f"Iteration {iteration+1}: Query: {current_query}")
                
                if self.search_pipeline:
                    retrieved_docs = self.search_pipeline.search(current_query)
                else:
                    from config.config import RetrievalConfig
                    temp_pipeline = SearchPipeline(
                        retriever=self.retriever,
                        model_config=self.model_config,
                        retrieval_config=RetrievalConfig()
                    )
                    retrieved_docs = temp_pipeline.search(current_query)
                
                # Handle no documents found
                if not retrieved_docs:
                    search_history.append({
                        "iteration": iteration + 1,
                        "query": current_query,
                        "num_docs": 0,
                        "error": "No documents found"
                    })
                    break
                
                # Process unique documents
                new_docs = []
                seen_texts = {doc.text for doc in accumulated_docs}
                
                for doc in retrieved_docs:
                    if doc.text not in seen_texts:
                        new_docs.append(doc)
                        seen_texts.add(doc.text)
                
                accumulated_docs.extend(new_docs)
                context = self.retriever.get_formatted_context(accumulated_docs)
                
                # Generate and process response
                # Sử dụng current_query thay vì question gốc
                prompt = self.prompt_template.format(
                    question=current_query,  
                    context=context
                )
                
                input_tokens = self._count_tokens(prompt)
                response = await self.llm.acomplete(prompt)
                output_tokens = self._count_tokens(response.text)
                
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                parsed_response = self._parse_response(response.text)
                
                # Track iteration
                search_history.append({
                    "iteration": iteration + 1,
                    "query": current_query,
                    "num_docs": len(retrieved_docs),
                    "new_docs": len(new_docs),
                    "total_docs": len(accumulated_docs),
                    "response": parsed_response
                })
                
                # Check if we have enough information
                if parsed_response["decision"].strip().lower() == "đã đủ thông tin":
                    logger.info("Found sufficient information")
                    parsed_response["search_history"] = search_history
                    parsed_response["token_usage"] = {
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "total_tokens": total_input_tokens + total_output_tokens
                    }
                    parsed_response["llm_provider"] = self.model_config.llm_provider
                    
                    # Bổ sung query_info nếu có
                    if query_info:
                        parsed_response["query_info"] = query_info
                    
                    return parsed_response
                
                # If we need more information
                if parsed_response["next_query"]:
                    current_query = parsed_response["next_query"]
                    iteration += 1
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {str(e)}")
                search_history.append({
                    "iteration": iteration + 1,
                    "query": current_query,
                    "error": str(e)
                })
                iteration += 1
                continue
        
        # After max iterations, if we still need more information, use what we have
        if accumulated_docs:
            context = self.retriever.get_formatted_context(accumulated_docs)
            
            final_prompt = self.final_effort_prompt_template.format(
                question=current_query,  # Sử dụng current_query thay vì question gốc
                context=context
            )
            
            input_tokens = self._count_tokens(final_prompt)
            response = await self.llm.acomplete(final_prompt)
            output_tokens = self._count_tokens(response.text)
            
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            
            parsed_response = self._parse_response(response.text)
            parsed_response["decision"] = "Đã đủ thông tin"
            
            parsed_response["search_history"] = search_history
            parsed_response["token_usage"] = {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens
            }
            parsed_response["llm_provider"] = self.model_config.llm_provider
            parsed_response["note"] = "Câu trả lời được tạo ra dựa trên thông tin tìm được, có thể chưa hoàn toàn đầy đủ."
            
            # Bổ sung query_info nếu có
            if query_info:
                parsed_response["query_info"] = query_info
                
            return parsed_response
        
        # If all else fails
        final_response = {
            "analysis": "Sau nhiều lần tìm kiếm, hệ thống vẫn chưa tìm thấy đủ thông tin về luật giao thông liên quan đến câu hỏi này.",
            "query_type": "unknown",
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
        
        # Bổ sung query_info nếu có
        if query_info:
            final_response["query_info"] = query_info
            
        return final_response