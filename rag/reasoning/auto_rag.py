from typing import Dict, Any, Optional, List
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore
import tiktoken
import logging
from config.config import ModelConfig, LLMProvider
from retrieval.retriever import DocumentRetriever
from retrieval.search_pipline import SearchPipeline
from reasoning.prompts import SYSTEM_PROMPT, FINAL_EFFORT_PROMPT
from llm.vllm_client import VLLMClient


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
        self.prompt_template = PromptTemplate(template=SYSTEM_PROMPT)
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
    
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format with standardized next query"""
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
                parsed["final_answer"] = ""
            elif current_section == "final_answer" and line is not None:
                if parsed["final_answer"]:
                    parsed["final_answer"] += "\n" + line
                else:
                    parsed["final_answer"] = line
            elif current_section and current_section != "final_answer" and line:
                parsed[current_section] += " " + line
        
        if parsed["next_query"] and "Đối với" not in parsed["next_query"]:
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
            logger.info(f"Standardized next query: {parsed['next_query']}")
                
        return parsed

    def _extract_vehicle_type(self, query: str) -> str:
        """Extract vehicle type from query"""
        if "xe máy" in query.lower() or "mô tô" in query.lower():
            return "xe máy"
        elif "ô tô" in query.lower() or "xe hơi" in query.lower():
            return "ô tô"
        else:
            return "phương tiện"  

    def _extract_violation_type(self, query: str) -> str:
        """Extract violation type from query"""
        
        violation = query
        if "thông tin về" in query:
            violation = query.split("thông tin về")[1].strip()
        if "cho hành vi" in query:
            violation = query.split("cho hành vi")[1].strip()
        if "sẽ bị" in violation:
            violation = violation.split("sẽ bị")[0].strip()
        return violation

    def _extract_penalty_types(self, query: str) -> List[str]:
        """Extract penalty types mentioned in query"""
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
    
    async def get_answer(self, question: str) -> Dict[str, Any]:
        
        iteration = 0
        accumulated_context = []  # This will store all retrieved documents across iterations
        accumulated_docs = []     # Keep a copy of all NodeWithScore objects
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
                for i, doc in enumerate(retrieved_docs[:3]):  # Log first 3 docs
                    logger.info(f"Doc {i+1} score: {doc.score}")
                    # Log metadata if available
                    if hasattr(doc.node, 'metadata'):
                        metadata = doc.node.metadata
                        logger.info(f"Doc {i+1} metadata: {metadata}")
                
                
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
                
                # Generate prompt and get response
                prompt = self.prompt_template.format(
                    question=question,  
                    context=context
                )
                
                input_tokens = self._count_tokens(prompt)
                logger.info(f"Input tokens: {input_tokens}")
                
                response = await self.llm.acomplete(prompt)
                output_tokens = self._count_tokens(response.text)
                logger.info(f"Output tokens: {output_tokens}")
                
        
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                # Parse response
                parsed_response = self._parse_response(response.text)
                logger.info(f"Decision: {parsed_response['decision']}")
                
                
                search_history.append({
                    "iteration": iteration + 1,
                    "query": current_query,
                    "num_docs": len(retrieved_docs),
                    "new_docs": len(new_docs),
                    "total_docs": len(accumulated_docs),
                    "response": parsed_response
                })
                
                
                if parsed_response["decision"].strip().lower() == "đã đủ thông tin":
                
                    logger.info("Found sufficient information, returning final answer")
                    parsed_response["search_history"] = search_history
                    parsed_response["token_usage"] = {
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "total_tokens": total_input_tokens + total_output_tokens
                    }
                    parsed_response["llm_provider"] = self.model_config.llm_provider
                    return parsed_response
                
               
                if parsed_response["next_query"]:
                    current_query = parsed_response["next_query"]
                    logger.info(f"Need more information, next query: {current_query}")
                    iteration += 1
                else:
                   
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
        
        # After max iterations, if we still need more information, use what we have
        if accumulated_docs:
            
                logger.info("Generating final answer using all accumulated documents")
                context = self.retriever.get_formatted_context(accumulated_docs)
                
                # Create a special final prompt that instructs the LLM to provide a best effort answer
                final_prompt = FINAL_EFFORT_PROMPT.format(
                    question=question,
                    context=context
                )
                
                input_tokens = self._count_tokens(final_prompt)
                logger.info(f"Final input tokens: {input_tokens}")
                
                response = await self.llm.acomplete(final_prompt)
                output_tokens = self._count_tokens(response.text)
                logger.info(f"Final output tokens: {output_tokens}")
                
                # Update token counts
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                # Parse response
                parsed_response = self._parse_response(response.text)
                
                # Force the decision to "Đã đủ thông tin"
                parsed_response["decision"] = "Đã đủ thông tin"
                
                logger.info("Generated best-effort final answer")
                parsed_response["search_history"] = search_history
                parsed_response["token_usage"] = {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "total_tokens": total_input_tokens + total_output_tokens
                }
                parsed_response["llm_provider"] = self.model_config.llm_provider
                parsed_response["note"] = "Câu trả lời được tạo ra dựa trên thông tin tìm được, có thể chưa hoàn toàn đầy đủ."
                
                return parsed_response
                
        
        
        # If all else fails
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
            "llm_provider": self.model_config.llm_provider
        }
        
        return final_response