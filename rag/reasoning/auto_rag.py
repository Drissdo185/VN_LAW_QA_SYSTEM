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
from reasoning.prompts import (
    DECISION_PROMPT_FOR_VIOLATION,
    DECISION_PROMPT_FOR_REGULATION,
    FORMAT_OUTPUT_PROMPT_FOR_VIOLATION,
    FORMAT_OUTPUT_PROMPT_FOR_REGULATION,
    FINAL_EFFORT_PROMPT_FOR_VIOLATION,
    FINAL_EFFORT_PROMPT_FOR_REGULATION
)
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
        # No longer need a default prompt template as we'll use specific prompts for each stage
        self.tokenizer = self._setup_tokenizer()
    
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

    def _setup_tokenizer(self):
        """Set up the tokenizer based on the LLM provider"""
        if self.model_config.llm_provider == LLMProvider.OPENAI:
            return tiktoken.encoding_for_model(self.model_config.openai_model)
        elif self.model_config.llm_provider in [LLMProvider.VLLM, LLMProvider.OLLAMA]:
            return tiktoken.get_encoding("cl100k_base")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.model_config.llm_provider}")
    
    def _is_traffic_related(self, question: str) -> bool:
        """Check if question is related to traffic"""
        traffic_keywords = [
            # Từ khóa hiện tại
            'mũ bảo hiểm', 'giao thông', 'đường bộ', 'biển báo', 
            'luật giao thông', 'phạt', 'xe máy', 'ô tô', 'bằng lái',
            'giấy phép', 'nd168', 'nghị định', 'nồng độ cồn', 
            'vượt đèn đỏ', 'tốc độ', 'vạch kẻ đường', 'tai nạn',
            'xe', 'đường', 'đậu xe', 'đỗ xe', 'dừng xe', 'đăng kiểm',
            'giấy tờ', 'biển số', 'traffic', 'luật', 'quá tải',
            
            # Từ khóa bổ sung cho quy định chung
            'độ tuổi', 'tuổi lái xe', 'được phép', 'điều kiện', 
            'quy định', 'yêu cầu', 'cấp giấy phép', 'cấp bằng', 
            'hạng bằng', 'loại bằng', 'A', 'A1', 'B1 số tự động', '',
            'gắn máy', 'được lái', 'người điều khiển', 'hạng tuổi',
            'sức khỏe', 'học lái xe', 'trường dạy lái xe', 'luật số',
            'thông tư', 'quy chuẩn', 'tiêu chuẩn'
        ]
        
        question_lower = question.lower()
        for keyword in traffic_keywords:
            if keyword in question_lower:
                logger.info(f"Question validated as traffic-related via keyword: {keyword}")
                return True
        
        return False
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def _detect_question_type(self, query: str) -> str:
        """
        Detect if the question is about regulations or violations
        
        Args:
            query: The question to analyze
            
        Returns:
            "regulation" or "violation"
        """
        query_lower = query.lower()
        
        regulation_keywords = [
            'quy định', 'điều kiện', 'độ tuổi', 'được phép', 'khi nào', 
            'yêu cầu', 'cấp giấy phép', 'cấp bằng', 'hạng bằng', 
            'loại bằng', 'được lái', 'cho phép', 'luật định',
            'khi nào thì', 'làm thế nào để', 'cần những gì'
        ]
        
        violation_keywords = [
            'vi phạm', 'phạt', 'xử phạt', 'hình phạt', 'tiền phạt',
            'bị phạt', 'mức phạt', 'tịch thu', 'tước giấy phép',
            'phạt bao nhiêu', 'trừ điểm', 'phạt như thế nào'
        ]
        
        # Check for regulation keywords
        for keyword in regulation_keywords:
            if keyword in query_lower:
                logger.info(f"Detected as regulation question via keyword: {keyword}")
                return "regulation"
        
        # Check for violation keywords
        for keyword in violation_keywords:
            if keyword in query_lower:
                logger.info(f"Detected as violation question via keyword: {keyword}")
                return "violation"
        
        # Default to violation if can't determine
        logger.info("Could not determine question type, defaulting to violation")
        return "violation"

    def _parse_decision_response(self, response: str) -> Dict[str, Any]:
        """
        Parse response from decision prompts to extract analysis, decision and next query.
        
        Args:
            response: LLM response text
            
        Returns:
            Dictionary with analysis, decision, and next_query fields
        """
        lines = response.strip().split("\n")
        parsed = {
            "analysis": "",
            "decision": "",
            "next_query": None
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("Phân tích:"):
                current_section = "analysis"
                parsed["analysis"] = line.replace("Phân tích:", "").strip()
            elif line.startswith("Quyết định:"):
                current_section = "decision"
                parsed["decision"] = line.replace("Quyết định:", "").strip()
            elif line.startswith("Truy vấn tiếp theo:"):
                current_section = "next_query"
                parsed["next_query"] = line.replace("Truy vấn tiếp theo:", "").strip()
            elif current_section and line:
                # Continue adding content to the current section
                parsed[current_section] += " " + line
        
        # Ensure decision is one of the expected values
        if parsed["decision"] and parsed["decision"].strip() not in ["Đã đủ thông tin", "Cần thêm thông tin"]:
            logger.warning(f"Unexpected decision value: '{parsed['decision']}'. Defaulting to 'Cần thêm thông tin'")
            parsed["decision"] = "Cần thêm thông tin"
        
        logger.info(f"Parsed decision: '{parsed['decision']}'")
        logger.info(f"Next query: {parsed['next_query'] or 'None'}")
        
        return parsed

    async def get_answer(self, question: str) -> Dict[str, Any]:
        """Get answer for a traffic-related question using Auto RAG with iterations"""
        
        # Optional traffic-related check if needed
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
        
        # Detect question type
        question_type = self._detect_question_type(question)
        logger.info(f"Initial question type detection: {question_type}")
        
        # Initialize retrieval variables
        iteration = 0
        accumulated_docs = []    
        total_input_tokens = 0
        total_output_tokens = 0
        search_history = []
        
        current_query = question
        while iteration < self.max_iterations:
            try:
                # Get documents for current query
                logger.info(f"Iteration {iteration+1}: Retrieving documents for query: {current_query}")
                
                # Always use SearchPipeline for retrieval
                if self.search_pipeline:
                    # Use existing SearchPipeline
                    logger.info("Using SearchPipeline for retrieval")
                    retrieved_docs = self.search_pipeline.search(current_query)
                else:
                    # Create a new search pipeline with the retriever
                    from config.config import RetrievalConfig
                    logger.info("Creating temporary SearchPipeline")
                    temp_pipeline = SearchPipeline(
                        retriever=self.retriever,
                        model_config=self.model_config,
                        retrieval_config=RetrievalConfig()
                    )
                    retrieved_docs = temp_pipeline.search(current_query)
                
                logger.info(f"Retrieved {len(retrieved_docs)} documents")
                
                # Skip iteration if no documents found
                if not retrieved_docs:
                    search_history.append({
                        "iteration": iteration + 1,
                        "query": current_query,
                        "num_docs": 0,
                        "error": "No documents found"
                    })
                    logger.warning(f"No documents found for query: {current_query}")
                    break
                
                # Log retrieved document information
                logger.info("Retrieved document details:")
                for i, doc in enumerate(retrieved_docs[:3]):  
                    logger.info(f"Doc {i+1} score: {doc.score}")
                    # Log metadata if available
                    if hasattr(doc.node, 'metadata'):
                        metadata = doc.node.metadata
                        logger.info(f"Doc {i+1} metadata: {metadata}")
                
                # Filter out duplicate documents
                new_docs = []
                seen_texts = {doc.text for doc in accumulated_docs}
                
                for doc in retrieved_docs:
                    if doc.text not in seen_texts:
                        new_docs.append(doc)
                        seen_texts.add(doc.text)
                
                # Add unique new documents to accumulated collections
                accumulated_docs.extend(new_docs)
                
                # Format context from ALL accumulated documents
                context = self.retriever.get_formatted_context(accumulated_docs)
                
                # Log context length
                logger.info(f"Context length: {len(context)} characters")
                logger.info(f"Total accumulated documents: {len(accumulated_docs)}")
                
                # Step 1: Use the appropriate decision prompt based on question type
                if question_type == "violation":
                    decision_prompt = DECISION_PROMPT_FOR_VIOLATION.format(
                        question=question,
                        context=context
                    )
                    logger.info("Using violation decision prompt")
                else:
                    decision_prompt = DECISION_PROMPT_FOR_REGULATION.format(
                        question=question,
                        context=context
                    )
                    logger.info("Using regulation decision prompt")
                
                input_tokens = self._count_tokens(decision_prompt)
                logger.info(f"Decision prompt tokens: {input_tokens}")
                
                # Get decision from LLM
                decision_response = await self.llm.acomplete(decision_prompt)
                decision_tokens = self._count_tokens(decision_response.text)
                logger.info(f"Decision response tokens: {decision_tokens}")
                
                # Parse decision response with dedicated parser
                parsed_decision = self._parse_decision_response(decision_response.text)
                
                # Update token counts
                total_input_tokens += input_tokens
                total_output_tokens += decision_tokens
                
                # Track search iteration
                search_history.append({
                    "iteration": iteration + 1,
                    "query": current_query,
                    "num_docs": len(retrieved_docs),
                    "new_docs": len(new_docs),
                    "total_docs": len(accumulated_docs),
                    "decision": parsed_decision["decision"],
                    "analysis": parsed_decision["analysis"]
                })
                
                # Check if we have enough information (Step 2)
                if parsed_decision["decision"].strip().lower() == "đã đủ thông tin":
                    # We have enough information, generate final answer with format prompt
                    logger.info("Found sufficient information, generating formatted answer")
                    
                    # Use the appropriate format prompt based on question type
                    if question_type == "violation":
                        format_prompt = FORMAT_OUTPUT_PROMPT_FOR_VIOLATION.format(
                            question=question,
                            context=context
                        )
                        logger.info("Using violation format prompt")
                    else:
                        format_prompt = FORMAT_OUTPUT_PROMPT_FOR_REGULATION.format(
                            question=question,
                            context=context
                        )
                        logger.info("Using regulation format prompt")
                    
                    format_tokens = self._count_tokens(format_prompt)
                    format_response = await self.llm.acomplete(format_prompt)
                    format_response_tokens = self._count_tokens(format_response.text)
                    
                    # Update token counts
                    total_input_tokens += format_tokens
                    total_output_tokens += format_response_tokens
                    
                    # Prepare final response
                    final_response = {
                        "analysis": parsed_decision["analysis"],
                        "decision": parsed_decision["decision"],
                        "final_answer": format_response.text,
                        "search_history": search_history,
                        "token_usage": {
                            "input_tokens": total_input_tokens,
                            "output_tokens": total_output_tokens,
                            "total_tokens": total_input_tokens + total_output_tokens
                        },
                        "llm_provider": self.model_config.llm_provider,
                        "question_type": question_type
                    }
                    
                    return final_response
                
                # If we need more information (Step 3)
                if parsed_decision["next_query"]:
                    current_query = parsed_decision["next_query"]
                    
                    # Update question type if the next query indicates a different type
                    if "Theo luật giao thông đường bộ" in current_query:
                        question_type = "regulation"
                        logger.info("Updated question type to regulation based on next query")
                    elif "vi phạm" in current_query or "xử phạt" in current_query:
                        question_type = "violation"
                        logger.info("Updated question type to violation based on next query")
                    
                    logger.info(f"Need more information, next query: {current_query}")
                    iteration += 1
                else:
                    # No next query provided but needs more info - break loop
                    logger.warning("Need more information but no next query provided")
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
        
        # After max iterations or if loop was broken, handle incomplete information
        if accumulated_docs:
            logger.info("Generating best-effort final answer using all accumulated documents")
            context = self.retriever.get_formatted_context(accumulated_docs)
            
            # Use the appropriate final effort prompt based on question type
            if question_type == "violation":
                final_prompt = FINAL_EFFORT_PROMPT_FOR_VIOLATION.format(
                    question=question,
                    context=context
                )
                logger.info("Using violation final effort prompt")
            else:
                final_prompt = FINAL_EFFORT_PROMPT_FOR_REGULATION.format(
                    question=question,
                    context=context
                )
                logger.info("Using regulation final effort prompt")
            
            input_tokens = self._count_tokens(final_prompt)
            logger.info(f"Final effort input tokens: {input_tokens}")
            
            response = await self.llm.acomplete(final_prompt)
            output_tokens = self._count_tokens(response.text)
            logger.info(f"Final effort output tokens: {output_tokens}")
            
            # Update token counts
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            
            # No need to parse final effort response, just return the full text as final_answer
            final_response = {
                "analysis": "Thông tin có thể chưa đầy đủ, nhưng cố gắng đưa ra câu trả lời tốt nhất có thể.",
                "decision": "Đã đủ thông tin",
                "final_answer": response.text,
                "search_history": search_history,
                "token_usage": {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "total_tokens": total_input_tokens + total_output_tokens
                },
                "llm_provider": self.model_config.llm_provider,
                "question_type": question_type,
                "note": "Câu trả lời được tạo ra dựa trên thông tin tìm được, có thể chưa hoàn toàn đầy đủ."
            }
            
            return final_response
        
        # If all else fails (no documents found)
        logger.warning("Exiting loop without sufficient information")
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
            "llm_provider": self.model_config.llm_provider,
            "question_type": question_type
        }
        
        return final_response