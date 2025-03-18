# vllm_client.py
import requests
import json
from llama_index.core.llms import CompletionResponse

class VLLMClient:
    def __init__(self, api_url="http://localhost:8000/v1/completions", model_name="Qwen/Qwen2.5-14B-Instruct-AWQ", temperature=0.2, max_tokens=4096):
        """
        Initialize VLLM Client
        
        Args:
            api_url: URL of the VLLM API endpoint
            model_name: Model to use with VLLM server
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
        """
        self.api_url = api_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def complete(self, prompt):
        """
        Send completion request to VLLM server
        
        Args:
            prompt: Text prompt for completion
            
        Returns:
            CompletionResponse object with response text
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": None
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                text = result["choices"][0]["text"]
                return CompletionResponse(text=text)
            else:
                return CompletionResponse(text="Error: No completion returned from VLLM server")
                
        except Exception as e:
            return CompletionResponse(text=f"Error calling VLLM API: {str(e)}")