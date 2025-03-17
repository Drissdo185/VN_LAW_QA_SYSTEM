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

class AutoRAG:
    def __init__(
        self,
        weaviate_host="localhost",
        weaviate_port=8080,
        weaviate_grpc_port=50051,
        index_name="ND168",
        embed_model_name="dangvantuan/vietnamese-document-embedding",
        embed_cache_folder="/home/user/.cache/huggingface/hub",
        model_name="gpt-4o-mini",
        temperature=0.2,
        top_k=10,
        alpha=0.5,
        max_iterations=3
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
        
        # Connect to Weaviate
        self.client = weaviate.connect_to_local(
            host=weaviate_host,
            port=weaviate_port,
            grpc_port=weaviate_grpc_port
        )
        
        # Setup vector store
        self.vector_store = WeaviateVectorStore(
            weaviate_client=self.client,
            index_name=index_name
        )
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
           model_name="dangvantuan/vietnamese-document-embedding", trust_remote_code=True,cache_folder="/home/drissdo/.cache/huggingface/hub"
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
        
        # Initialize LLM
        self.llm = OpenAI(model=model_name, temperature=temperature)
        
        # Set max iterations for the autonomous loop
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
        response = self.llm.complete(prompt)
        
        # Extract JSON from the response
        json_pattern = r'```json\n(.*?)```'
        match = re.search(json_pattern, response.text, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        else:
            # Fallback if JSON extraction fails
            return {
                "formatted_query": processed_output["processed_question"],
                "vehicle_type": "ô tô",
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
        
        response = self.llm.complete(prompt)
        
        # Extract JSON decision
        json_pattern = r'```json\n(.*?)```'
        match = re.search(json_pattern, response.text, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        else:
            # Fallback if JSON extraction fails
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
        
        response = self.llm.complete(prompt)
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
        
        print(f"Original question: {original_question}")
        print(f"Processed question: {processed_question}")
        print(f"Question type: {question_type}")
        
        # Step 2: Format the query
        formatted_output = self.format_query(processed_output)
        formatted_query = formatted_output.get("formatted_query", processed_question)
        
        print(f"Formatted query: {formatted_query}")
        
        # Initialize tracking variables
        iteration = 0
        all_context = ""
        query_history = [formatted_query]
        
        # Start iterative information gathering loop
        while iteration < self.max_iterations:
            print(f"\nIteration {iteration + 1}: Query - {formatted_query}")
            
            # Step 3: Retrieve documents
            retrieved_docs, context = self.retrieve_documents(formatted_query)
            all_context += context + "\n"
            
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
                    "iterations": iteration + 1
                }
            
            # If we need more information and have a follow-up query
            if evaluation["next_query"] and iteration < self.max_iterations - 1:
                formatted_query = evaluation["next_query"]
                query_history.append(formatted_query)
                iteration += 1
            else:
                # If we've run out of iterations or don't have a follow-up query
                break
        
        # If we've exhausted iterations, generate best answer with what we have
        print("\nReached maximum iterations. Generating best possible answer.")
        final_answer = self.generate_answer(original_question, all_context)
        
        return {
            "original_question": original_question,
            "processed_question": processed_question,
            "formatted_query": formatted_query,
            "query_history": query_history,
            "context": all_context,
            "answer": final_answer,
            "iterations": iteration + 1
        }


# Example usage
if __name__ == "__main__":
    # Create AutoRAG instance
    auto_rag = AutoRAG(
        weaviate_host="192.168.100.125",
        weaviate_port=8080,
        weaviate_grpc_port=50051,
        index_name="ND168",
        embed_model_name="dangvantuan/vietnamese-document-embedding",
        embed_cache_folder="/home/user/.cache/huggingface/hub",
        model_name="gpt-4o-mini",
        temperature=0.2,
        top_k=10,
        alpha=0.5,
        max_iterations=3
    )
    
    # Example question
    question = "Tôi uống 1 thùng bia và quên mang cà vẹt xe"
    result = auto_rag.process(question)
    
    print("\n--- FINAL RESULT ---")
    print(f"Original Question: {result['original_question']}")
    print(f"Processed Question: {result['processed_question']}")
    print(f"Total Iterations: {result['iterations']}")
    print(f"Query History: {result['query_history']}")
    print("\nFinal Answer:")
    print(result['answer'])