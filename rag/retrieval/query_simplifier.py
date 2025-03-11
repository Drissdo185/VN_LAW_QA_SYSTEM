import logging
from typing import Dict, Any

from llama_index.llms.openai import OpenAI
from config.config import ModelConfig, LLMProvider
from llm.vllm_client import VLLMClient
from llm.ollama_client import OllamaClient
from retrieval.traffic_synonyms import TrafficSynonymExpander
from reasoning.prompts import QUERY_STANDARDIZATION_PROMPT
import json
import re

logger = logging.getLogger(__name__)

class QuerySimplifier:
    """
    A component that standardizes user queries before retrieval to improve relevance.
    It removes noise and extracts key legal concepts related to traffic laws.
    
    Note: This updated version assumes the query may already have gone through
    synonym expansion, so we focus only on standardization.
    """
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.llm = self._setup_llm()
        self.synonym_expander = TrafficSynonymExpander()
        
    def _setup_llm(self):
        """Set up the LLM based on the provider configuration"""
        if self.model_config.llm_provider == LLMProvider.OPENAI:
            logger.info(f"Setting up OpenAI LLM for query simplification: {self.model_config.openai_model}")
            return OpenAI(
                model=self.model_config.openai_model,
                api_key=self.model_config.openai_api_key
            )
        elif self.model_config.llm_provider == LLMProvider.VLLM:
            try:
                logger.info(f"Setting up vLLM for query simplification: {self.model_config.vllm_config.model_name}")
                client = VLLMClient.from_config(self.model_config.vllm_config)
                return client
            except Exception as e:
                logger.error(f"Error initializing vLLM client: {str(e)}")
                raise
        elif self.model_config.llm_provider == LLMProvider.OLLAMA:
            try:
                logger.info(f"Setting up Ollama for query simplification: {self.model_config.ollama_config.model_name}")
                client = OllamaClient.from_config(self.model_config.ollama_config)
                return client
            except Exception as e:
                logger.error(f"Error initializing Ollama client: {str(e)}")
                raise
        else:
            raise ValueError(f"Unsupported LLM provider: {self.model_config.llm_provider}")
    

    async def simplify_query(self, query: str) -> Dict[str, Any]:
        """
        Standardize a user query by extracting relevant legal concepts and removing noise.
        
        Args:
            query: The user query (potentially already with expanded synonyms)
            
        Returns:
            Dictionary containing standardized query and extraction metadata
        """
        logger.info(f"Standardizing query: {query}")
        
        # Get legal terms for additional context in the prompt
        legal_terms = self.synonym_expander.get_legal_terms(query)
        logger.info(f"Identified legal terms: {legal_terms}")
        
        legal_terms_hint = ""
        if legal_terms:
            legal_terms_hint = f"""
            Các thuật ngữ pháp lý được nhận diện trong câu hỏi:
            {', '.join(legal_terms)}
            """
        
        # Use the imported prompt template
        prompt = QUERY_STANDARDIZATION_PROMPT.format(
            query=query,
            legal_terms_hint=legal_terms_hint
        )
        
        try:
            logger.info("Sending query to LLM for standardization")
            response = await self.llm.acomplete(prompt)
            simplified_result = self._parse_simplifier_response(response.text)
            
            # Ensure we have a standardized query, even if parsing fails
            standardized_query = simplified_result.get('standardized_query')
            if not standardized_query:
                vehicle_type = simplified_result.get('vehicle_type', 'phương tiện')
                violations_str = ', '.join(simplified_result.get('violations', ['vi phạm giao thông']))
                standardized_query = f"Đối với {vehicle_type}, vi phạm {violations_str} sẽ bị xử phạt như thế nào?"
                simplified_result['standardized_query'] = standardized_query
                logger.info(f"Created standardized query from parsed components: '{standardized_query}'")
            else:
                logger.info(f"Successfully standardized query: '{standardized_query}'")
            
            logger.info(f"Detected violations: {simplified_result.get('violations')}")
            logger.info(f"Vehicle type: {simplified_result.get('vehicle_type')}")
            logger.info(f"Penalty types: {simplified_result.get('penalty_types')}")
            
            return {
                "original_query": query,
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
                "original_query": query,
                "standardized_query": query,
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
        
        Args:
            response_text: The raw response from the LLM
            
        Returns:
            Structured dictionary of extracted data
        """
        logger.debug(f"Parsing LLM response: {response_text[:100]}...")
        
        try:
            # Try to extract JSON from response
            json_pattern = r'```json\s*([\s\S]*?)\s*```|{\s*"[\s\S]*?}'
            json_match = re.search(json_pattern, response_text)
            
            if json_match:
                json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
                
                json_str = json_str.replace('```json', '').replace('```', '')
                parsed_data = json.loads(json_str)
                logger.info("Successfully parsed JSON response")
                return parsed_data
            
            # Fallback to regex extraction if JSON parsing fails
            logger.warning("JSON parsing failed, falling back to regex extraction")
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