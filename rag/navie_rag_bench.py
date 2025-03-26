from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import weaviate
import json
import re
import csv
import time
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

class NaiveRAG:
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
    
    def retrieve_documents(self, query):
        """Simple document retrieval without any query processing"""
        retrieved_docs = self.retriever.retrieve(query)
        
        # Compile context text from retrieved documents
        context = ""
        for node in retrieved_docs:
            context += node.text + "\n\n"
        
        return retrieved_docs, context
    
    def generate_answer(self, question, context):
        """Generate an answer using a simple prompt with retrieved context"""
        prompt = f"""
        Dưới đây là câu hỏi về luật giao thông Việt Nam:

        Câu hỏi: {question}

        Dựa trên thông tin sau đây để trả lời câu hỏi:
        {context}

        Hãy trả lời câu hỏi dựa trên thông tin được cung cấp. Nếu thông tin không đủ, hãy nêu rõ điều đó.
        """
        
        response = self.llm.complete(prompt)
        return response.text
    
    def generate_multiple_choice_answer(self, question, context, options):
        """Generate answer for multiple choice question"""
        prompt = f"""
        Dựa trên thông tin đã thu thập về luật giao thông Việt Nam, hãy chọn đáp án đúng nhất (A, B, C, hoặc D) cho câu hỏi sau:

        Câu hỏi: {question}

        Các lựa chọn:
        A. {options['option_a']}
        B. {options['option_b']}
        C. {options['option_c']}
        D. {options['option_d']}

        Thông tin tham khảo:
        {context}

        CHÚ Ý: Chỉ trả lời bằng MỘT chữ cái duy nhất A, B, C hoặc D. Không viết thêm nội dung nào khác.
        """
        
        response = self.llm.complete(prompt)
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
        
        return selected_answer
    
    def process(self, question, multiple_choice_options=None):
        """Process a question through the naive RAG pipeline"""
        start_time = time.time()
        
        # Step 1: Direct retrieval with the original question
        retrieved_docs, context = self.retrieve_documents(question)
        
        # Step 2: Generate answer based on retrieved context
        if multiple_choice_options:
            answer = self.generate_multiple_choice_answer(question, context, multiple_choice_options)
        else:
            answer = self.generate_answer(question, context)
        
        processing_time = time.time() - start_time
        
        return {
            "question": question,
            "context": context,
            "answer": answer,
            "processing_time": processing_time,
            "num_docs_retrieved": len(retrieved_docs)
        }
    
    def benchmark(self, question_data, num_questions=None):
        """Run benchmark on a set of multiple choice questions"""
        results = {
            "correct": 0,
            "total": 0,
            "accuracy": 0.0,
            "avg_processing_time": 0.0,
            "detailed_results": []
        }
        
        total_time = 0
        
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
            
            total_time += result["processing_time"]
            
            result_record = {
                "question_idx": idx,
                "question": q_data["question"],
                "model_answer": result["answer"],
                "correct_answer": q_data["correct_answer"],
                "is_correct": is_correct,
                "processing_time": result["processing_time"],
                "context_length": len(result["context"]),
                "num_docs_retrieved": result["num_docs_retrieved"]
            }
            
            results["detailed_results"].append(result_record)
            results["total"] += 1
        
        if results["total"] > 0:
            results["accuracy"] = results["correct"] / results["total"] * 100
            results["avg_processing_time"] = total_time / results["total"]
        
        return results

if __name__ == "__main__":
    # Initialize the benchmark RAG system
    naive_rag = NaiveRAG(
        model_name="gpt-4o-mini",
        temperature=0.2,
        top_k=10
    )
    
    # Path to your questions dataset
    question_data_path = "/home/drissdo/Desktop/VN_LAW_QA_SYSTEM/TEST/benchmark.csv"
    
    try:
        # Load CSV data
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
        
        # Run benchmark on specified number of questions
        results = naive_rag.benchmark(questions, num_questions=50)
        
        # Print summary results
        print("\n===== NAIVE RAG BENCHMARK RESULTS =====")
        print(f"Correct answers: {results['correct']}/{results['total']}")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        print(f"Average processing time: {results['avg_processing_time']:.2f} seconds")
        
        # Print detailed results
        print("\n===== DETAILED RESULTS =====")
        for result in results['detailed_results']:
            print(f"Question {result['question_idx'] + 1}: {result['question'][:100]}...")
            print(f"  Model answer: {result['model_answer']}, Correct answer: {result['correct_answer']}")
            print(f"  Correct: {'✓' if result['is_correct'] else '✗'}")
            print(f"  Processing time: {result['processing_time']:.2f}s")
            print(f"  Documents retrieved: {result['num_docs_retrieved']}")
            print()
        
        # Save results to file
        output_file = "naive_rag_benchmark_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        import traceback
        traceback.print_exc()  # Print the full traceback for debugging
    finally:
        # Make sure to close the Weaviate client to avoid resource warnings
        if naive_rag and hasattr(naive_rag, 'client'):
            naive_rag.client.close()