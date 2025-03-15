import pandas as pd
import asyncio
import logging
import json
import os
import re
import sys
from typing import Dict, List, Any
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch

# Add the parent directory to the path to find modules
# This assumes the script is in a subdirectory of the main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('benchmark_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the necessary components from the RAG system
try:
    from retrieval.traffic_synonyms import TrafficSynonymExpander
    from retrieval.search_pipline import SearchPipeline
    from retrieval.retriever import DocumentRetriever
    from retrieval.vector_store import VectorStoreManager
    from reasoning.auto_rag import AutoRAG
    from reasoning.enhanced_autorag import EnhancedAutoRAG
    from config.config import (
        ModelConfig, 
        RetrievalConfig, 
        WeaviateConfig, 
        LLMProvider,
        VLLMConfig
    )
    logger.info("Successfully imported all required modules")
except ImportError as e:
    logger.error(f"Error importing RAG system modules: {str(e)}")
    logger.error(f"Current sys.path: {sys.path}")
    logger.error("Try running this script from the project root directory")
    raise

class BenchmarkEvaluator:
    def __init__(self, model_provider: LLMProvider = LLMProvider.OPENAI, use_enhanced_rag: bool = True):
        """
        Initialize the benchmark evaluator with configuration and components.
        
        Args:
            model_provider: The LLM provider to use (OPENAI or VLLM)
            use_enhanced_rag: Whether to use EnhancedAutoRAG with query standardization
        """
        logger.info(f"Initializing benchmark evaluator with {model_provider.value} and enhanced_rag={use_enhanced_rag}")
        
        # Initialize configurations
        self.weaviate_config = WeaviateConfig(
            url=os.getenv("WEAVIATE_TRAFFIC_URL", "http://192.168.100.125:8080"),
            api_key=os.getenv("WEAVIATE_TRAFFIC_KEY", ""),
            collection="ND168"
        )
        
        self.model_config = ModelConfig(
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            embedding_model="dangvantuan/vietnamese-document-embedding",
            cross_encoder_model="dangvantuan/vietnamese-document-embedding",
            chunk_size=512,
            chunk_overlap=50,
            llm_provider=model_provider,
            openai_model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        self.retrieval_config = RetrievalConfig(
            vector_store_query_mode="hybrid",
            similarity_top_k=20,
            alpha=0.5
        )
        
        # Initialize components
        self.vector_store_manager = None
        self.retriever = None
        self.search_pipeline = None
        self.auto_rag = None
        self.enhanced_auto_rag = None
        self.use_enhanced_rag = use_enhanced_rag
        
        # Benchmark results
        self.results = []
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all components needed for the benchmark"""
        logger.info("Initializing vector store and retrieval components")
        
        try:
            # Initialize vector store
            self.vector_store_manager = VectorStoreManager(
                weaviate_config=self.weaviate_config,
                model_config=self.model_config
            )
            self.vector_store_manager.initialize()
            
            # Initialize retriever
            self.retriever = DocumentRetriever(
                index=self.vector_store_manager.get_index(),
                config=self.retrieval_config
            )
            
            # Initialize search pipeline
            self.search_pipeline = SearchPipeline(
                retriever=self.retriever,
                model_config=self.model_config,
                retrieval_config=self.retrieval_config
            )
            
            # Initialize standard AutoRAG
            self.auto_rag = AutoRAG(
                model_config=self.model_config,
                retriever=self.retriever,
                search_pipeline=self.search_pipeline
            )
            
            # Initialize enhanced AutoRAG
            self.enhanced_auto_rag = EnhancedAutoRAG(
                model_config=self.model_config,
                retriever=self.retriever,
                search_pipeline=self.search_pipeline
            )
            
            logger.info("Successfully initialized all RAG components")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def get_active_rag(self):
        """Get the active RAG component based on configuration"""
        if self.use_enhanced_rag:
            return self.enhanced_auto_rag
        else:
            return self.auto_rag
    
    async def process_question(self, question: str, correct_answer: str, question_type: str) -> Dict[str, Any]:
        """
        Process a single benchmark question and evaluate the result
        
        Args:
            question: The question to process
            correct_answer: The correct answer from the benchmark dataset
            question_type: The type of question (e.g., don_gian, phuc_tap, etc.)
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Processing question: {question}")
        
        try:
            # Get the active RAG component
            rag = self.get_active_rag()
            
            # Get answer from RAG
            start_time = datetime.now()
            response = await rag.get_answer(question)
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Check if the system identified the correct answer
            answer_correct = self._evaluate_answer(response.get("final_answer", ""), correct_answer)
            
            # Prepare result
            result = {
                "question": question,
                "correct_answer": correct_answer,
                "question_type": question_type,
                "system_answer": response.get("final_answer", ""),
                "answer_correct": answer_correct,
                "processing_time": processing_time,
                "token_usage": response.get("token_usage", {}),
                "decision": response.get("decision", ""),
                "analysis": response.get("analysis", ""),
                "query_info": response.get("query_info", {})
            }
            
            logger.info(f"Question processed in {processing_time:.2f} seconds. Correct: {answer_correct}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "question": question,
                "correct_answer": correct_answer,
                "question_type": question_type,
                "error": str(e),
                "answer_correct": False,
                "processing_time": 0
            }
    
    def _evaluate_answer(self, system_answer: str, correct_answer: str) -> bool:
        """
        Evaluate if the system's answer correctly identified the answer choice
        
        Args:
            system_answer: The answer provided by the RAG system
            correct_answer: The correct answer (A, B, C, or D)
            
        Returns:
            Whether the system correctly identified the answer
        """
        # Extract answer letter patterns from the system's answer
        answer_patterns = [
            r"Đáp\s*án\s*(?:đúng|chính\s*xác)?\s*(?:là)?\s*([A-D])",
            r"Câu\s*trả\s*lời\s*(?:đúng|chính\s*xác)?\s*(?:là)?\s*([A-D])",
            r"(?:Lựa\s*chọn|Phương\s*án)\s*(?:đúng|chính\s*xác)?\s*(?:là)?\s*([A-D])",
            r"(?:^|\s)([A-D])\s*là\s*(?:đáp\s*án|câu\s*trả\s*lời)\s*(?:đúng|chính\s*xác)",
            r"(?:mức\s*phạt|xử\s*phạt)\s*([A-D])"
        ]
        
        # Check if the system's answer contains the correct answer letter
        for pattern in answer_patterns:
            matches = re.findall(pattern, system_answer, re.IGNORECASE)
            if matches and any(match.upper() == correct_answer.upper() for match in matches):
                return True
        
        # If no direct mention of the answer letter, check if the content matches
        answer_content_mapping = {
            "A": ["200.000 - 400.000", "400.000 - 600.000", "4.000.000 - 6.000.000", 
                  "6.000.000 - 8.000.000", "14 tuổi", "Hạng A", "30 km/h", "3 tháng", "35m", 
                  "Hạng D", "100 km/h", "Có báo hiệu", "Cả A và B", "35m", "Sau 12 tháng", 
                  "70 km/h", "40 km/h", "80 km/h", "50 km/h", "Xe ô tô chở người", "Duy trì tốc độ", 
                  "50 km/h", "Tự động được phục hồi", "Xe ô tô con", "Đủ tuổi", 
                  "Bấm còi liên tục", "3 điểm", "Tăng tốc", "Vi phạm", "Đợi 3 tháng", 
                  "Duy trì tốc độ 60 km/h", "Đủ điều kiện", "Tăng tốc để nhanh chóng", 
                  "Duy trì khoảng cách", "Mở cửa sổ", "Tiếp tục lái", 
                  "Tiếp tục lái và kiểm tra", "Phanh gấp", "Chờ thời gian"],
            "B": ["400.000 - 600.000", "600.000 - 800.000", "2.000.000 - 3.000.000", 
                  "18.000.000 - 20.000.000", "15 tuổi", "Hạng A1", "40 km/h", "4 tháng", 
                  "55m", "Hạng D1", "110 km/h", "Qua cầu, cống hẹp", "Chỉ A", "Lớn hơn 35m", 
                  "Sau 6 tháng", "80 km/h", "50 km/h", "100 km/h", "60 km/h", 
                  "Xe ô tô tải trên 7.500 kg", "Giảm tốc độ và tăng khoảng cách", 
                  "60 km/h", "Nộp đơn xin cấp lại", "Xe ô tô con đi trên đường hai chiều", 
                  "Không đủ tuổi và phải chờ thêm 2 năm", "Duy trì tốc độ và đi vòng qua", 
                  "12 điểm", "Duy trì tốc độ và làn đường", "Không vi phạm", 
                  "Đợi ít nhất 6 tháng", "Tăng tốc", "Không đủ điều kiện", 
                  "Duy trì tốc độ và bật đèn pha", "Tăng tốc để nhanh chóng đến đích", 
                  "Uống cà phê", "Tăng tốc để nhanh chóng", "Tăng tốc để xem có thay đổi", 
                  "Đánh lái sang làn khác", "Tham gia khóa đào tạo"],
            "C": ["600.000 - 800.000", "800.000 - 1.000.000", "4.000.000 - 6.000.000", 
                  "30.000.000 - 40.000.000", "16 tuổi", "Hạng A3", "50 km/h", "5 tháng", 
                  "70m", "Hạng BE", "120 km/h", "Khi có người đi bộ", "Chỉ B", "55m", 
                  "Khi tham gia khóa học", "90 km/h", "60 km/h", "120 km/h", "70 km/h", 
                  "Xe ô tô kéo rơ moóc", "Tăng tốc độ để nhanh chóng thoát khỏi", 
                  "70 km/h", "Tham gia kiểm tra kiến thức", "Xe ô tô tải 5 tấn", 
                  "Không đủ tuổi và phải chờ thêm 5 năm", "Tăng tốc", "15 điểm", 
                  "Giảm tốc độ và nhường đường", "Chỉ được phép duy trì", 
                  "Đợi 12 tháng", "Giảm tốc độ và sẵn sàng dừng lại", 
                  "Cần được kiểm tra sức khỏe", "Giảm tốc độ thích hợp", 
                  "Lái xe với tốc độ tối đa", "Bật nhạc to", "Giảm tốc độ và quan sát", 
                  "Giảm tốc độ, tìm nơi an toàn", "Giảm tốc độ, chuẩn bị phanh", 
                  "Tham gia kiểm tra kiến thức pháp luật"],
            "D": ["800.000 - 1.000.000", "1.000.000 - 2.000.000", "6.000.000 - 8.000.000", 
                  "40.000.000 - 50.000.000", "18 tuổi", "Hạng A4", "60 km/h", "6 tháng", 
                  "100m", "Hạng DE", "130 km/h", "Khi đi trên đường cao tốc", 
                  "Không ai trong số họ", "Lớn hơn 55m", "Khi nộp lệ phí", "100 km/h", 
                  "70 km/h", "140 km/h", "80 km/h", "Xe ô tô chở người giường nằm", 
                  "Duy trì tốc độ nhưng bật đèn pha sáng nhất", "80 km/h", 
                  "Tham gia khóa học lái xe mới", "Xe ô tô buýt", 
                  "Đủ tuổi nhưng cần phải có 1 năm kinh nghiệm", "Giảm tốc độ để đảm bảo an toàn", 
                  "9 điểm", "Dừng xe ngay lập tức", "Cần tăng tốc độ lên 120 km/h", 
                  "Đăng ký thi lại bằng lái", "Chuyển làn", "Chỉ được phép lái xe trong khu vực", 
                  "Dừng xe ngay lập tức", "Không quan tâm đến tốc độ", 
                  "Tìm nơi an toàn để dừng xe", "Theo sau xe khác", 
                  "Tắt nhạc và tiếp tục lái", "Bấm còi liên tục", "Nộp phạt và xin cấp lại"]
        }
        
        # Check if system answer contains distinctive phrases matching the correct answer
        for phrase in answer_content_mapping.get(correct_answer.upper(), []):
            if phrase.lower() in system_answer.lower():
                return True
        
        return False
        
    async def run_benchmark(self, csv_path: str, sample_size: int = None) -> Dict[str, Any]:
        """
        Run the benchmark on questions from the CSV file
        
        Args:
            csv_path: Path to the benchmark CSV file
            sample_size: Number of questions to sample (None = all questions)
            
        Returns:
            Dictionary with benchmark results
        """
        # Load benchmark data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} questions from {csv_path}")
        
        # Sample questions if requested
        if sample_size and sample_size < len(df):
            # Ensure we get a stratified sample by question type
            sampled_df = df.groupby('loai_cau_hoi', group_keys=False).apply(
                lambda x: x.sample(min(len(x), int(sample_size * len(x) / len(df))))
            )
            logger.info(f"Sampled {len(sampled_df)} questions stratified by question type")
            df = sampled_df
        
        # Process all questions
        self.results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
            question = row['cau_hoi']
            correct_answer = row['dap_an_dung']
            question_type = row['loai_cau_hoi']
            
            result = await self.process_question(question, correct_answer, question_type)
            self.results.append(result)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model_config.llm_provider.value
        rag_type = "enhanced" if self.use_enhanced_rag else "standard"
        results_file = f"benchmark_results_{model_name}_{rag_type}_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "config": {
                    "model_provider": self.model_config.llm_provider.value,
                    "use_enhanced_rag": self.use_enhanced_rag,
                    "sample_size": sample_size,
                    "total_questions": len(df)
                },
                "metrics": metrics,
                "results": self.results
            }, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved benchmark results to {results_file}")
        logger.info(f"Overall accuracy: {metrics['overall_accuracy']:.2f}%")
        
        return {
            "metrics": metrics,
            "results": self.results,
            "results_file": results_file
        }
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate benchmark metrics from results
        
        Returns:
            Dictionary with calculated metrics
        """
        if not self.results:
            return {
                "overall_accuracy": 0,
                "by_question_type": {},
                "processing_time": {
                    "avg": 0,
                    "min": 0,
                    "max": 0
                },
                "token_usage": {
                    "avg_total": 0,
                    "avg_input": 0,
                    "avg_output": 0
                }
            }
        
        # Calculate overall accuracy
        correct_count = sum(1 for r in self.results if r.get("answer_correct", False))
        total_count = len(self.results)
        overall_accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        
        # Calculate accuracy by question type
        by_type = {}
        for question_type in set(r.get("question_type") for r in self.results):
            type_results = [r for r in self.results if r.get("question_type") == question_type]
            type_correct = sum(1 for r in type_results if r.get("answer_correct", False))
            type_total = len(type_results)
            type_accuracy = (type_correct / type_total) * 100 if type_total > 0 else 0
            by_type[question_type] = {
                "accuracy": type_accuracy,
                "correct": type_correct,
                "total": type_total
            }
        
        # Calculate processing time statistics
        processing_times = [r.get("processing_time", 0) for r in self.results if r.get("processing_time") is not None]
        
        # Calculate token usage statistics
        token_usages = [r.get("token_usage", {}) for r in self.results if "token_usage" in r]
        avg_total_tokens = np.mean([usage.get("total_tokens", 0) for usage in token_usages]) if token_usages else 0
        avg_input_tokens = np.mean([usage.get("input_tokens", 0) for usage in token_usages]) if token_usages else 0
        avg_output_tokens = np.mean([usage.get("output_tokens", 0) for usage in token_usages]) if token_usages else 0
        
        return {
            "overall_accuracy": overall_accuracy,
            "by_question_type": by_type,
            "processing_time": {
                "avg": np.mean(processing_times) if processing_times else 0,
                "min": np.min(processing_times) if processing_times else 0,
                "max": np.max(processing_times) if processing_times else 0
            },
            "token_usage": {
                "avg_total": avg_total_tokens,
                "avg_input": avg_input_tokens,
                "avg_output": avg_output_tokens
            }
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.vector_store_manager:
            self.vector_store_manager.cleanup()

async def main():
    """Main function to run the benchmark"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run benchmark for traffic law QA system')
    parser.add_argument('--csv', type=str, default='./benchmark.csv', help='Path to benchmark CSV file')
    parser.add_argument('--model', type=str, choices=['openai', 'vllm'], default='openai', help='LLM provider to use')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced RAG with query standardization')
    parser.add_argument('--sample', type=int, default=None, help='Number of questions to sample')
    
    args = parser.parse_args()
    
    # Set up model provider
    model_provider = LLMProvider.OPENAI if args.model == 'openai' else LLMProvider.VLLM
    
    # Initialize benchmark evaluator
    evaluator = BenchmarkEvaluator(
        model_provider=model_provider,
        use_enhanced_rag=args.enhanced
    )
    
    try:
        # Run benchmark
        logger.info(f"Starting benchmark with {args.model} model and enhanced={args.enhanced}")
        results = await evaluator.run_benchmark(args.csv, sample_size=args.sample)
        
        # Print summary
        metrics = results["metrics"]
        print("\n" + "="*50)
        print(f"Benchmark Results Summary:")
        print(f"Model: {args.model.upper()}, Enhanced RAG: {args.enhanced}")
        print(f"Questions: {len(results['results'])}")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
        
        print("\nAccuracy by Question Type:")
        for qtype, data in metrics["by_question_type"].items():
            print(f"  {qtype}: {data['accuracy']:.2f}% ({data['correct']}/{data['total']})")
        
        print("\nProcessing Time:")
        print(f"  Average: {metrics['processing_time']['avg']:.2f} seconds")
        print(f"  Min: {metrics['processing_time']['min']:.2f} seconds")
        print(f"  Max: {metrics['processing_time']['max']:.2f} seconds")
        
        print("\nToken Usage:")
        print(f"  Average Total: {int(metrics['token_usage']['avg_total'])}")
        print(f"  Average Input: {int(metrics['token_usage']['avg_input'])}")
        print(f"  Average Output: {int(metrics['token_usage']['avg_output'])}")
        
        print("\nResults saved to:", results.get("results_file", ""))
        print("="*50)
    
    finally:
        # Clean up
        evaluator.cleanup()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())