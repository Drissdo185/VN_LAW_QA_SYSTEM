from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import weaviate
import json
import re
from q_process import QuestionProcess
from prompt import (
    VIOLATION_QUERY_FORMAT,
    GENERAL_INFORMATION_QUERY_FORMAT,
    DECISION_VIOLATION,
    DECISION_GENERAL,
    ANSWER
)
from VLLM import VLLMClient
from weaviate.classes.init import Auth
import os

class AutoRAG:
    def __init__(
        self,
        weaviate_host="localhost",
        weaviate_port=8080,
        weaviate_grpc_port=50051,
        index_name="ND168",
        embed_model_name="bkai-foundation-models/vietnamese-bi-encoder",
        embed_cache_folder="/home/user/.cache/huggingface/hub",
        model_name="gpt-4o-mini",
        temperature=0.2,
        top_k=10,
        alpha=0.5,
        max_iterations=3,
        llm_provider="openai",
        vllm_api_url="http://localhost:8000/v1/completions",
        vllm_model_name="Qwen/Qwen2.5-14B-Instruct-AWQ",
        vllm_max_tokens=1024
    ):
        """
        Initialize the Autonomous RAG system.
        
        Args:
            weaviate_host: Host for Weaviate vector database
            weaviate_port: HTTP port for Weaviate
            weaviate_grpc_port: gRPC port for Weaviate
            index_name: Name of the index in Weaviate
            embed_model_name: HuggingFace embedding model name
            embed_cache_folder: Cache folder for embeddings
            model_name: LLM model to use
            temperature: LLM temperature setting
            top_k: Number of documents to retrieve
            alpha: Hybrid search parameter
            max_iterations: Maximum number of query iterations
        """
        # Initialize question processor
        self.question_processor = QuestionProcess()
        
        
        
        # # Connect to Weaviate cloud
        # self.client = weaviate.connect_to_local(
        #     host=weaviate_host,
        #     port=weaviate_port,
        #     grpc_port=weaviate_grpc_port
        # )
        
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")), 
        )
        
        
        # Setup vector store
        self.vector_store = WeaviateVectorStore(
            weaviate_client=self.client,
            index_name=index_name
        )
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
           model_name="bkai-foundation-models/vietnamese-bi-encoder", trust_remote_code=True,cache_folder="/home/drissdo/.cache/huggingface/hub"
        )
        
        # Create vector index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )
        
        # Setup retriever with hybrid search
        self.retriever = self.index.as_retriever(
            vector_store_query_mode="hybrid",
            similarity_top_k=top_k,
            alpha=alpha
        )
        
        self.llm_provider = llm_provider
        if llm_provider == "openai":
            self.llm = OpenAI(model=model_name, temperature=temperature)
        elif llm_provider == "vllm":
            self.llm = VLLMClient(
                api_url=vllm_api_url,
                model_name=vllm_model_name,
                temperature=temperature,
                max_tokens=vllm_max_tokens
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")  
        
        self.max_iterations = max_iterations
    
    def process_question(self, question):
        """Process user question using the QuestionProcess class"""
        return self.question_processor.process_question(question)
    
    def format_query(self, processed_output):
        """Format the processed question into a standardized query"""
        # Select appropriate prompt template based on question type
        if processed_output["question_type"] == "violation_type":
            prompt_template = VIOLATION_QUERY_FORMAT
        else:
            prompt_template = GENERAL_INFORMATION_QUERY_FORMAT
        
        # Format the prompt
        prompt = prompt_template.format(
            processed_question=processed_output["processed_question"]
        )
        
        # Get LLM response
        response = self.llm.complete(prompt, max_tokens=128)
        
        # Debug - print raw response
        print(f"LLM Provider: {self.llm_provider}")
        print("=== RAW LLM RESPONSE ===")
        print(response.text)
        print("========================")
        
        # Try multiple JSON extraction patterns
        extraction_patterns = [
            r'```json\n(.*?)```',   # Standard markdown JSON block
            r'```\n(.*?)\n```',     # Code block without language
            r'```(.*?)```',         # Any code block
            r'({.*})',              # Just find JSON object
            r'"formatted_query":\s*"([^"]*)"'  # Direct extraction of formatted query
        ]
        
        for pattern in extraction_patterns:
            match = re.search(pattern, response.text, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1)
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    continue
        
        # If we reach here, no extraction pattern worked
        # Fallback if JSON extraction fails
        print("WARNING: JSON extraction failed, using fallback")
        return {
            "formatted_query": processed_output["processed_question"],
            "vehicle_type": "ô tô" if "ô tô" in processed_output["processed_question"].lower() else "mô tô và gắn máy",
            "violation_type": processed_output["processed_question"] 
        }
    
    def retrieve_documents(self, query):
        """Retrieve relevant documents based on query"""
        retrieved_docs = self.retriever.retrieve(query)
        
        # Compile context text from retrieved documents
        context = ""
        for node in retrieved_docs:
            context += node.text
        
        return retrieved_docs, context
    
    def evaluate_information(self, question, context, question_type, violation_type=None):
        """Evaluate if retrieved information is sufficient to answer the question"""
        if question_type == "violation_type":
            prompt = DECISION_VIOLATION.format(
                question=question,
                context=context,
                violation_type=violation_type
            )
        else:
            prompt = DECISION_GENERAL.format(
                question=question,
                context=context
            )
        
        response = self.llm.complete(prompt, max_tokens=248)
        

        extraction_patterns = [
            r'```json\n(.*?)```',   
            r'```\n(.*?)\n```',    
            r'```(.*?)```',        
            r'{[\s\S]*?}',          
            r'"analysis"[\s\S]*?"decision"[\s\S]*?(?:"next_query"|"final_answer")' 
        ]
        
        for pattern in extraction_patterns:
            match = re.search(pattern, response.text, re.DOTALL)
            if match:
                try:
                    json_str = match.group(0)  
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    
                    try:
                        if match.groups():
                            json_str = match.group(1)
                            return json.loads(json_str)
                    except (json.JSONDecodeError, IndexError):
                        continue
        
        
        print("WARNING: JSON extraction failed in evaluate_information, using fallback")
        print(f"Raw response: {response.text[:100]}...")  
        
        return {
            "analysis": "Failed to parse evaluation",
            "decision": "Cần thêm thông tin",
            "next_query": f"Luật giao thông quy định như thế nào về {question}?",
            "final_answer": ""
        }
    
    def generate_answer(self, original_question, context):
        """Generate final answer based on all gathered information"""
        prompt = ANSWER.format(
            original_question=original_question,
            context_text=context
        )
        
        response = self.llm.complete(prompt, max_tokens= 516)
        return response.text
    
    def process(self, question):
        """
        Process a question through the entire RAG pipeline
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with process results including answer
        """
        # Step 1: Process the question
        processed_output = self.process_question(question)
        original_question = processed_output["original_question"]
        processed_question = processed_output["processed_question"]
        question_type = processed_output["question_type"]
        
        
        print(f"Question type: {question_type}")
        
        # Step 2: Format the query
        formatted_output = self.format_query(processed_output)
        formatted_query = formatted_output.get("formatted_query", processed_question)
        
        print(f"Formatted query: {formatted_query}")
        
        # Initialize tracking variables
        iteration = 0
        all_context = ""
        query_history = [formatted_query]
        
        # Initialize progress tracking
        missing_info = set()
        found_info = set()
        
        # Start iterative information gathering loop
        while iteration < self.max_iterations:
            print(f"\nIteration {iteration + 1}: Query - {formatted_query}")
            
            # Step 3: Retrieve documents
            retrieved_docs, context = self.retrieve_documents(formatted_query)
            
            # Only add new context if it's not empty
            if context.strip():
                all_context += context + "\n\n"
            
            # Log retrieved document count
            print(f"Retrieved {len(retrieved_docs)} documents")
            
            # Step 4: Evaluate if information is sufficient
            violation_type = formatted_output.get("violation_type", "")
            evaluation = self.evaluate_information(
                formatted_query,
                all_context,
                question_type,
                violation_type
            )
            
            print(f"Decision: {evaluation['decision']}")
            print(f"Analysis: {evaluation['analysis']}")
            
            # Track found and missing information
            if 'analysis' in evaluation:
                # Extract information about what was found and what's missing
                if "đã đủ thông tin về" in evaluation['analysis'].lower():
                    for info in re.findall(r"đã đủ thông tin về (.*?)(?:,|\.|$)", evaluation['analysis'].lower()):
                        found_info.add(info.strip())
                
                if "thiếu thông tin về" in evaluation['analysis'].lower():
                    for info in re.findall(r"thiếu thông tin về (.*?)(?:,|\.|$)", evaluation['analysis'].lower()):
                        missing_info.add(info.strip())
            
            # If we have enough information, generate the answer
            if evaluation["decision"] == "Đã đủ thông tin":
                if evaluation["final_answer"]:
                    answer = evaluation["final_answer"]
                else:
                    answer = self.generate_answer(original_question, all_context)
                
                return {
                    "original_question": original_question,
                    "processed_question": processed_question,
                    "formatted_query": formatted_query,
                    "query_history": query_history,
                    "context": all_context,
                    "answer": answer,
                    "iterations": iteration + 1,
                    "found_info": list(found_info),
                    "missing_info": list(missing_info)
                }
            
            # If we need more information and have a follow-up query
            if evaluation["next_query"] and iteration < self.max_iterations - 1:
                # Check if the next query is substantively different from previous queries
                if evaluation["next_query"] not in query_history:
                    formatted_query = evaluation["next_query"]
                    query_history.append(formatted_query)
                    iteration += 1
                else:
                    # If we're repeating queries, break the loop to avoid infinite loops
                    print("Avoiding duplicate query, breaking loop.")
                    break
            else:
                # If we've run out of iterations or don't have a follow-up query
                break
        
        # If we've exhausted iterations, generate best answer with what we have
        print("\nReached maximum iterations or no further relevant information found.")
        print(f"Found information about: {', '.join(found_info) if found_info else 'None'}")
        print(f"Missing information about: {', '.join(missing_info) if missing_info else 'None'}")
        
        final_answer = self.generate_answer(original_question, all_context)
        
        return {
            "original_question": original_question,
            "processed_question": processed_question,
            "formatted_query": formatted_query,
            "query_history": query_history,
            "context": all_context,
            "answer": final_answer,
            "iterations": iteration + 1,
            "found_info": list(found_info),
            "missing_info": list(missing_info)
        }