import requests
import json
import re
from llama_index.core.llms import CompletionResponse

class VLLMClient:
    def __init__(
        self, 
        api_url="http://192.168.100.125:8000/v1/completions", 
        model_name="Qwen/Qwen2.5-14B-Instruct-AWQ", 
        temperature=0.2, 
        max_tokens=4096,
        debug_mode=False
    ):
        """
        Initialize VLLM Client with enhanced JSON handling
        
        Args:
            api_url: URL of the VLLM API endpoint
            model_name: Model to use with VLLM server
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            debug_mode: Enable detailed debugging output
        """
        self.api_url = api_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug_mode = debug_mode
    
    def _format_json_prompt(self, prompt):
        """
        Add explicit JSON formatting instructions to the prompt
        """
        # Check if prompt is asking for JSON response
        if "JSON" in prompt and "{" in prompt:
            # Add explicit instruction for VLLM to format JSON correctly
            prompt += "\n\nIMPORTANT: Your response MUST be valid JSON wrapped in a code block like this:\n```json\n{\"key\": \"value\"}\n```"
        
        return prompt
    
    def _extract_json_from_text(self, text):
        """
        Try multiple methods to extract valid JSON from text
        """
        # Try multiple JSON extraction patterns
        extraction_patterns = [
            r'```json\n(.*?)```',   # Standard markdown JSON block
            r'```\n(.*?)\n```',     # Code block without language
            r'```(.*?)```',         # Any code block
            r'(\{[\s\S]*?\})',      # Find JSON object with flexible whitespace
        ]
        
        for pattern in extraction_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Try to load the matched text as JSON
                    json_obj = json.loads(match.strip())
                    if self.debug_mode:
                        print(f"Successfully parsed JSON with pattern: {pattern}")
                    return json_obj
                except json.JSONDecodeError:
                    continue
        
        # Direct JSON extraction attempt
        try:
            # Find the first occurrence of what looks like a JSON object
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
            
        # If we get here, no JSON could be extracted
        return None
    
    def complete(self, prompt, max_tokens=None):
        """
        Send completion request to VLLM server with enhanced JSON handling
        
        Args:
            prompt: Text prompt for completion
            max_tokens: Optional override for the default max_tokens
            
        Returns:
            CompletionResponse object with response text
        """
        # Format prompt to encourage proper JSON formatting
        formatted_prompt = self._format_json_prompt(prompt)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Use the provided max_tokens if specified, otherwise use the default
        tokens_to_use = max_tokens if max_tokens is not None else self.max_tokens
        
        payload = {
            "model": self.model_name,
            "prompt": formatted_prompt,
            "temperature": self.temperature,
            "max_tokens": tokens_to_use,
            "stop": None
        }
        
        try:
            if self.debug_mode:
                print(f"Sending request to VLLM API: {self.api_url}")
                print(f"Using max_tokens: {tokens_to_use}")
                
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                text = result["choices"][0]["text"]
                
                if self.debug_mode:
                    print(f"Raw VLLM response:\n{text[:500]}...")
                
                # Try to extract JSON from the response if the prompt asked for JSON
                if "JSON" in prompt and self.debug_mode:
                    json_data = self._extract_json_from_text(text)
                    
                    if json_data:
                        print(f"Extracted JSON: {json.dumps(json_data, indent=2)}")
                    else:
                        print("Failed to extract JSON from response")
                
                # Create a CompletionResponse with just the text - no custom attributes
                return CompletionResponse(text=text)
            else:
                return CompletionResponse(text="Error: No completion returned from VLLM server")
                
        except Exception as e:
            error_msg = f"Error calling VLLM API: {str(e)}"
            if self.debug_mode:
                print(error_msg)
            return CompletionResponse(text=error_msg)