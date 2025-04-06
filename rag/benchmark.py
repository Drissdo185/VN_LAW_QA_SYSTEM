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
    BENCHMARK_PROMPT
    
)
from VLLM import VLLMClient
from weaviate.classes.init import Auth
import os

class BenchMarkRAG:
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
        
        prompt_template = VIOLATION_QUERY_FORMAT
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
        
        prompt = DECISION_VIOLATION.format(
                question=question,
                context=context,
                violation_type=violation_type
            )
    
        
        response = self.llm.complete(prompt, max_tokens=248)
        
        # Extract JSON decision directly from the response text
        extraction_patterns = [
            r'```json\n(.*?)```',   # Standard markdown JSON block
            r'```\n(.*?)\n```',     # Code block without language
            r'```(.*?)```',         # Any code block
            r'{[\s\S]*?}',          # Find JSON object with flexible whitespace
            r'"analysis"[\s\S]*?"decision"[\s\S]*?(?:"next_query"|"final_answer")'  # Look for expected fields
        ]
        
        for pattern in extraction_patterns:
            match = re.search(pattern, response.text, re.DOTALL)
            if match:
                try:
                    json_str = match.group(0)  # Use the entire matched string
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    # Try with group(1) if available
                    try:
                        if match.groups():
                            json_str = match.group(1)
                            return json.loads(json_str)
                    except (json.JSONDecodeError, IndexError):
                        continue
        
        # If no patterns match, use fallback
        print("WARNING: JSON extraction failed in evaluate_information, using fallback")
        print(f"Raw response: {response.text[:100]}...")  # Print beginning of response for debugging
        
        return {
            "analysis": "Failed to parse evaluation",
            "decision": "Cần thêm thông tin",
            "next_query": f"{question}?",
            "final_answer": ""
        }
    
    # def generate_answer(self, original_question, context):
    #     """Generate final answer based on all gathered information"""
    #     prompt = ANSWER.format(
    #         original_question=original_question,
    #         context_text=context
    #     )
        
    #     response = self.llm.complete(prompt, max_tokens= 516)
    #     return response.text
    
    def benchmark(self, question_data, num_questions=None):
        results = {
        "correct": 0,
        "total": 0,
        "accuracy": 0.0,
        "detailed_results": []
    }
    
        
        if num_questions is not None:
            questions_to_evaluate = question_data[:num_questions]
        else:
            questions_to_evaluate = question_data
        
        for idx, q_data in enumerate(questions_to_evaluate):
            print(f"Processing question {idx+1}/{len(questions_to_evaluate)}...")
            
            
            options = {
                "option_a": q_data["option_a"],
                "option_b": q_data["option_b"],
                "option_c": q_data["option_c"],
                "option_d": q_data["option_d"]
            }
            
            
            result = self.process(q_data["question"], multiple_choice_options=options)
            
            
            is_correct = result["answer"] == q_data["correct_answer"]
            if is_correct:
                results["correct"] += 1
            
           
            result_record = {
                "question_idx": idx,
                "question": q_data["question"],
                "model_answer": result["answer"],
                "correct_answer": q_data["correct_answer"],
                "is_correct": is_correct,
                "iterations": result["iterations"],
                "context_length": len(result["context"]),
                "found_info": result["found_info"],
                "missing_info": result["missing_info"]
            }
            
            results["detailed_results"].append(result_record)
            results["total"] += 1
        
        
        if results["total"] > 0:
            results["accuracy"] = results["correct"] / results["total"] * 100
        
        return results
    
    def process(self, question, multiple_choice_options=None):
        
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
            
            # If we have enough information or this is the last iteration
            if evaluation["decision"] == "Đã đủ thông tin" or iteration == self.max_iterations - 1:
                break
            
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
        
        # Generate answer based on whether this is a benchmark question or regular question
        if multiple_choice_options:
            # For benchmark questions, use the BENCHMARK_PROMPT to get A, B, C, or D
            benchmark_prompt = BENCHMARK_PROMPT.format(
                question=original_question,
                option_a=multiple_choice_options["option_a"],
                option_b=multiple_choice_options["option_b"],
                option_c=multiple_choice_options["option_c"],
                option_d=multiple_choice_options["option_d"],
                context=all_context
            )
            
            response = self.llm.complete(benchmark_prompt, max_tokens=8)
            answer_text = response.text.strip().upper()
            
            # Extract just the letter from the response
            selected_answer = None
            for letter in ["A", "B", "C", "D"]:
                if letter in answer_text:
                    selected_answer = letter
                    break
            
            if not selected_answer:
                # If no letter was found, use the first character if it's a valid option
                if answer_text and answer_text[0] in ["A", "B", "C", "D"]:
                    selected_answer = answer_text[0]
                else:
                    selected_answer = "No valid answer provided"
            
            answer = selected_answer
        else:
            # For regular questions, generate a natural language answer
            if evaluation.get("final_answer"):
                answer = evaluation["final_answer"]
            else:
                answer = self.generate_answer(original_question, all_context)
        
        # Return the results
        return {
            "original_question": original_question,
            "processed_question": processed_question,
            "formatted_query": formatted_query,
            "query_history": query_history,
            "context": all_context,
            "answer": answer,
            "iterations": iteration + 1,
            "found_info": list(found_info),
            "missing_info": list(missing_info),
            "is_multiple_choice": multiple_choice_options is not None
        }

if __name__ == "__main__":
    benchmark_rag = BenchMarkRAG(
        model_name="gpt-4o-mini",
        temperature=0.2,
        top_k=10,
        alpha=0.5,
        max_iterations=3
    )
    
    # Path to your questions dataset
    question_data_path = "/home/drissdo/Desktop/VN_LAW_QA_SYSTEM/TEST/benchmark_2.csv"
    
    try:
        # Load CSV data instead of JSON
        import csv
        
        questions = []
        with open(question_data_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                questions.append({
                    "question": row["question"],
                    "option_a": row["option_a"],
                    "option_b": row["option_b"],
                    "option_c": row["option_c"],
                    "option_d": row["option_d"],
                    "correct_answer": row["correct_answer"]
                })
        
        print(f"Loaded {len(questions)} questions from {question_data_path}")
        
        # Run benchmark on first 5 questions
        results = benchmark_rag.benchmark(questions, num_questions=5)
        
        # Print summary results
        print("\n===== BENCHMARK RESULTS =====")
        print(f"Correct answers: {results['correct']}/{results['total']}")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        
        # Print detailed results
        print("\n===== DETAILED RESULTS =====")
        for result in results['detailed_results']:
            print(f"Question {result['question_idx'] + 1}: {result['question'][:100]}...")
            print(f"  Model answer: {result['model_answer']}, Correct answer: {result['correct_answer']}")
            print(f"  Correct: {'✓' if result['is_correct'] else '✗'}")
            print(f"  Iterations: {result['iterations']}")
            print()
        
        # Save results to file
        output_file = "benchmark_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        import traceback
        traceback.print_exc()  # Print the full traceback for debugging
    finally:
        # Make sure to close the Weaviate client to avoid resource warnings
        if benchmark_rag and hasattr(benchmark_rag, 'client'):
            benchmark_rag.client.close()