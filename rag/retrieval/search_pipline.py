from typing import List
from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore
from pyvi import ViTokenizer
import logging
from config.config import ModelConfig, RetrievalConfig
from retrieval.retriever import DocumentRetriever

# Setup logger
logger = logging.getLogger(__name__)

class SearchPipeline:
    def __init__(
        self,
        retriever: DocumentRetriever,
        model_config: ModelConfig,
        retrieval_config: RetrievalConfig
    ):
        self.retriever = retriever
        self.model_config = model_config
        self.retrieval_config = retrieval_config
        
        try:
            self.cross_encoder = CrossEncoder(
                model_config.cross_encoder_model,
                device=model_config.device,
                trust_remote_code=True
            )
            logger.info(f"Initialized CrossEncoder with model: {model_config.cross_encoder_model}")
        except Exception as e:
            logger.error(f"Error initializing CrossEncoder: {str(e)}")
            logger.warning("SearchPipeline will operate without reranking")
            self.cross_encoder = None
            
        logger.info("Initialized SearchPipeline")
      
    def search(self, query: str) -> List[NodeWithScore]:
        """
        Execute the full search pipeline
        
        Steps:
        1. Initial hybrid retrieval (BM25 + Dense)
        2. Metadata filtering
        3. Cross-encoder reranking
        4. Term-based ranking
        5. Return top results
        """
        # Tokenize query for Vietnamese
        tokenized_query = ViTokenizer.tokenize(query.lower())
        logger.info(f"Processing search query: '{query}'")
        logger.info(f"Tokenized query: '{tokenized_query}'")
        
        try:
            # Step 1: Initial retrieval
            logger.info(f"Performing initial retrieval")
            if self.retriever:
                initial_results = self.retriever.retrieve(tokenized_query)
                logger.info(f"Initial retrieval returned {len(initial_results)} documents")
            else:
                logger.error("Retriever not properly initialized")
                return []
            
            if not initial_results:
                logger.warning("Initial retrieval returned no results")
                return []
                
            # Step 2: Apply metadata filtering
            logger.info("Applying metadata filtering")
            filtered_results = self._perform_metadata_filtering(initial_results, query)
            logger.info(f"Metadata filtering returned {len(filtered_results)} documents")
            
            # Step 3: Apply cross-encoder reranking
            if self.cross_encoder:
                logger.info("Applying cross-encoder reranking")
                reranked_results = self._rerank_results(query, filtered_results)
                logger.info(f"Reranking returned {len(reranked_results)} documents")
            else:
                logger.info("Skipping cross-encoder reranking (not available)")
                reranked_results = filtered_results
            
            # Step 4: Apply term-based ranking
            logger.info("Applying term-based ranking")
            final_results = self._rank_results(reranked_results, query)
            logger.info(f"Term-based ranking returned {len(final_results)} documents")
            
            # Log top result for debugging
            if final_results:
                logger.info(f"Top result score: {final_results[0].score}")
                logger.info(f"Top result snippet: {final_results[0].text[:100]}...")
            
            # Return top results
            top_k = min(self.retrieval_config.similarity_top_k, len(final_results))
            logger.info(f"Returning top {top_k} results")
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in search pipeline: {str(e)}")
            return []
    
    def _rerank_results(self, query: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Rerank results using cross-encoder
        
        """
        if not nodes:
            logger.warning("No nodes to rerank")
            return []
            
        if not self.cross_encoder:
            logger.warning("Cross-encoder not available, skipping reranking")
            return nodes
            
        try:
            # Prepare document-query pairs for cross-encoder
            pairs = [(query, node.text) for node in nodes]
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Update node scores
            scored_nodes = []
            for i, node in enumerate(nodes):
                # Create a copy to avoid modifying the original
                scored_node = NodeWithScore(
                    node=node.node,
                    score=float(scores[i])
                )
                scored_nodes.append(scored_node)
                
            # Sort by score (descending)
            return sorted(scored_nodes, key=lambda x: x.score, reverse=True)
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {str(e)}")
            # Return original nodes if reranking fails
            return nodes
    
    def _rank_results(self, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
        """
        Rank results based on term overlap and relevance score
        
        Args:
            nodes: List of nodes to rank
            query: The search query
            
        Returns:
            Ranked list of nodes
        """
        if not nodes:
            logger.warning("No nodes to rank")
            return []
            
        # Tokenize query using ViTokenizer for Vietnamese
        query_tokens = set(ViTokenizer.tokenize(query.lower()).split())
        logger.debug(f"Query tokens: {query_tokens}")
        
        scored_nodes = []
        for node in nodes:
            # Tokenize node text consistently with query
            node_tokens = set(ViTokenizer.tokenize(node.text.lower()).split())
            
            # Calculate term overlap
            overlap = len(query_tokens.intersection(node_tokens))
            overlap_score = overlap / max(1, len(query_tokens))
            
            # Calculate combined score (weighted average of similarity and term overlap)
            combined_score = (0.7 * node.score) + (0.3 * overlap_score)
            
            # Create a new node with the combined score
            scored_node = NodeWithScore(
                node=node.node,
                score=combined_score
            )
            scored_nodes.append(scored_node)
            
            logger.debug(f"Node score: {node.score}, Overlap: {overlap}, Combined: {combined_score}")
        
        # Sort by combined score
        return sorted(scored_nodes, key=lambda x: x.score, reverse=True)

    def _perform_metadata_filtering(self, nodes, query):
        """
        Filter nodes based on metadata matching query
        """
        query_lower = query.lower()
        
        # Detect violation types in query with expanded keywords
        violation_types = []
        
        # Child safety violations
        if any(keyword in query_lower for keyword in ["trẻ em", "trẻ nhỏ", "dưới 10 tuổi", "chiều cao dưới", 
                                                    "1,35 mét", "mầm non", "học sinh", "thiết bị an toàn",
                                                    "dưới 12 tuổi", "dưới 06 tuổi", "người già yếu"]):
            violation_types.append("trẻ_em")
        
        # Speed violations
        if any(keyword in query_lower for keyword in ["tốc độ", "km/h", "chạy quá", "vượt quá", "tốc độ tối thiểu", 
                                                    "chạy dưới", "tốc độ thấp", "đuổi nhau", "20 km", "10 km", 
                                                    "05 km", "tốc độ quy định"]):
            violation_types.append("tốc_độ")
            
        # Alcohol violations
        if any(keyword in query_lower for keyword in ["nồng độ cồn", "cồn", "rượu", "bia", "miligam", 
                                                    "nồng độ trong máu", "hơi thở", "say xỉn", 
                                                    "không chấp hành yêu cầu kiểm tra", "80 miligam", 
                                                    "50 miligam", "mililít máu"]):
            violation_types.append("nồng_độ_cồn")
        
        # Drug violations
        if any(keyword in query_lower for keyword in ["ma túy", "chất kích thích", "chất gây nghiện", 
                                                    "pháp luật cấm sử dụng", "chất ma túy", "dương tính", 
                                                    "trong cơ thể có chất"]):
            violation_types.append("ma_túy")
        
        # Parking violations
        if any(keyword in query_lower for keyword in ["đỗ xe", "đậu xe", "dừng xe", "đỗ trái phép", 
                                                    "đỗ sai quy định", "vạch kẻ đường", "lề đường", "vỉa hè",
                                                    "trên cầu", "trong hầm", "nơi đường giao nhau", 
                                                    "điểm đón trả khách", "tụ tập từ 03 xe"]):
            violation_types.append("đỗ_dừng_xe")
        
        # Documentation violations
        if any(keyword in query_lower for keyword in ["giấy phép", "chứng nhận", "đăng ký", "kiểm định", 
                                                    "bảo hiểm", "giấy tờ", "không mang theo", "hết hạn",
                                                    "tẩy xóa", "không đúng số khung", "đăng ký tạm thời"]):
            violation_types.append("giấy_tờ")
        
        # Highway violations
        if any(keyword in query_lower for keyword in ["đường cao tốc", "cao tốc", "làn khẩn cấp", 
                                                    "vào cao tốc", "ra cao tốc", "đi vào đường cao tốc"]):
            violation_types.append("đường_cao_tốc")
        
        # Traffic signals violations
        if any(keyword in query_lower for keyword in ["đèn tín hiệu", "đèn giao thông", "đèn đỏ", 
                                                    "vượt đèn", "không chấp hành", "biển báo", "hiệu lệnh", 
                                                    "hướng dẫn", "người điều khiển giao thông", 
                                                    "người kiểm soát giao thông"]):
            violation_types.append("biển_báo")
        
        # Lane violations
        if any(keyword in query_lower for keyword in ["làn đường", "chuyển làn", "làn xe", "phần đường", 
                                                    "vượt làn", "đi sai làn", "không đúng phần đường", 
                                                    "không đi bên phải", "dàn hàng ngang"]):
            violation_types.append("làn_đường")
        
        # Passenger violations
        if any(keyword in query_lower for keyword in ["chở người", "quá số người", "chở quá", "người ngồi", 
                                                    "chở trên mui", "thùng xe", "chở theo 02 người", 
                                                    "chở theo từ 03 người"]):
            violation_types.append("chở_người")
        
        # Cargo violations
        if any(keyword in query_lower for keyword in ["chở hàng", "trọng tải", "quá tải", "hàng hóa", 
                                                    "vượt kích thước", "kích thước giới hạn", "kéo theo xe khác", 
                                                    "kéo vật khác"]):
            violation_types.append("chở_hàng")
        
        # Phone usage violations
        if any(keyword in query_lower for keyword in ["điện thoại", "nghe điện thoại", "nhắn tin", 
                                                    "sử dụng điện thoại", "thiết bị điện tử", "dùng tay cầm"]):
            violation_types.append("điện_thoại")
        
        # Dangerous driving
        if any(keyword in query_lower for keyword in ["lạng lách", "đánh võng", "nguy hiểm", "liều lĩnh", 
                                                    "không giữ khoảng cách", "vô lăng", "buông cả hai tay", 
                                                    "dùng chân điều khiển", "nằm trên yên xe", "quay người", 
                                                    "bịt mắt", "một bánh", "chân chống", "quệt xuống đường"]):
            violation_types.append("lái_xe_nguy_hiểm")
        
        # Motorcycle-specific violations
        if any(keyword in query_lower for keyword in ["mũ bảo hiểm", "đội mũ", "cài quai", "không đội mũ", 
                                                    "mô tô ba bánh", "xe hai bánh", "xe ba bánh", 
                                                    "xe lạng lách", "gắn máy", "xe máy", "ngồi phía sau vòng tay qua", 
                                                    "thành nhóm", "thành đoàn"]):
            violation_types.append("xe_máy_đặc_thù")
        
        # Vehicle accessories and equipment
        if any(keyword in query_lower for keyword in ["còi", "đèn soi biển số", "đèn báo hãm", "gương chiếu hậu", 
                                                    "đèn tín hiệu", "đèn chiếu sáng", "ô (dù)", "thiết bị âm thanh",
                                                    "rú ga", "nẹt pô", "đèn chiếu xa", "đèn chiếu gần"]):
            violation_types.append("kỹ_thuật_xe")
        
        # Environmental violations
        if any(keyword in query_lower for keyword in ["khói", "bụi", "ô nhiễm", "tiếng ồn", "môi trường", 
                                                    "giảm thanh", "giảm khói", "quy chuẩn môi trường"]):
            violation_types.append("môi_trường")
        
        # Wrong-way driving and turning
        if any(keyword in query_lower for keyword in ["lùi xe", "quay đầu", "đảo chiều", "quay xe", 
                                                    "cấm quay đầu", "ngược chiều", "đường một chiều", 
                                                    "cấm đi ngược chiều", "không quan sát hai bên"]):
            violation_types.append("lùi_quay_đầu")
        
        # Overtaking violations
        if any(keyword in query_lower for keyword in ["vượt xe", "không được vượt", "vượt bên phải", 
                                                    "nơi cấm vượt", "không có tín hiệu", "tín hiệu vượt xe"]):
            violation_types.append("vượt_xe")
        
        # Accident related
        if any(keyword in query_lower for keyword in ["tai nạn", "va chạm", "không dừng lại", "bỏ chạy", 
                                                    "không trợ giúp", "trốn khỏi", "không giữ nguyên hiện trường", 
                                                    "không dừng ngay phương tiện", "liên quan trực tiếp"]):
            violation_types.append("tai_nạn")
        
        # Safety violations
        if any(keyword in query_lower for keyword in ["an toàn", "khoảng cách an toàn", "xảy ra va chạm", 
                                                    "chuyển hướng không quan sát", "xe phía sau", 
                                                    "giảm tốc độ", "dừng lại"]):
            violation_types.append("an_toàn")
        
        # Priority violations
        if any(keyword in query_lower for keyword in ["nhường đường", "quyền ưu tiên", "xe ưu tiên", 
                                                    "xe được quyền ưu tiên", "xe cứu thương", "xe cứu hỏa", 
                                                    "phát tín hiệu ưu tiên", "đường ưu tiên", 
                                                    "không giảm tốc độ", "không nhường đường"]):
            violation_types.append("xe_ưu_tiên")
        
        # License point deduction
        if any(keyword in query_lower for keyword in ["trừ điểm", "giấy phép lái xe", "02 điểm", "04 điểm", 
                                                    "06 điểm", "10 điểm"]):
            violation_types.append("trừ_điểm")
        
        # License suspension
        if any(keyword in query_lower for keyword in ["tước quyền", "giấy phép lái xe", "10 tháng", "12 tháng", 
                                                    "22 tháng", "24 tháng"]):
            violation_types.append("tước_giấy_phép")
        
        # Vehicle confiscation
        if any(keyword in query_lower for keyword in ["tịch thu", "thu giữ", "phương tiện", "tái phạm"]):
            violation_types.append("tịch_thu")
        
        # Intersection violations
        if any(keyword in query_lower for keyword in ["giao nhau", "ngã ba", "ngã tư", "vòng xuyến", 
                                                    "đường giao nhau", "giao lộ"]):
            violation_types.append("giao_nhau")
        
        # License plate violations
        if any(keyword in query_lower for keyword in ["biển số", "không gắn biển số", "gắn biển số không đúng", 
                                                    "che lấp", "bẻ cong", "thay đổi chữ", "thay đổi số", 
                                                    "vị trí", "quy cách"]):
            violation_types.append("biển_số")
        
        # Vehicle-specific queries
        if any(keyword in query_lower for keyword in ["xe máy", "mô tô", "xe gắn máy", "xe Honda",
                                                    "xe Yamaha", "xe Suzuki", "xe Piaggio", "xe SYM",
                                                    "xe Vespa", "xe SH", "xe Air Blade", "xe Wave", "xe SH 350i",
                                                    "xe SH 150i",
                                                    "xe Dream", "xe Future", "xe Click", "xe Lead",
                                                    "xe Vision", "xe Wave Alpha", "xe Wave RSX",
                                                    "xe Wave RSX 110", "xe Wave S", "xe Wave S 110",
                                                    "xe Wave RS", "xe Wave RS 110", "xe Wave Alpha 110",
                                                    "xe Wave Alpha 100", "xe Wave RSX 100", "xe Wave S 100",
                                                    "xe Wave RS 100", "xe Wave RSX 110 Fi", "xe Wave S 110 Fi",
                                                    "xe Wave RS 110 Fi", "xe Wave Alpha 110 Fi", "xe Wave Alpha 100 Fi"]):
            vehicle_type = "mô tô, gắn máy"
        elif any(keyword in query_lower for keyword in ["ô tô", "xe hơi", "xe bốn bánh", "xe con","Mercedes",
                                                    "BMW", "Audi", "Toyota", "Honda", "Hyundai", "Kia",
                                                    "Mazda", "Ford", "Chevrolet", "Nissan", "Suzuki",
                                                    "Peugeot", "Renault", "Lexus", "Volvo", "Volkswagen",
                                                    "Mitsubishi", "Subaru", "Isuzu", "Hino"
                                        ]):
            vehicle_type = "ô tô"
        else:
            vehicle_type = None
        
        # Remove duplicates
        violation_types = list(set(violation_types))
        
        logger.info(f"Detected violation types: {violation_types}")
        if vehicle_type:
            logger.info(f"Detected vehicle type: {vehicle_type}")
        
        # First try filtering by both violation type and vehicle type if both exist
        if violation_types and vehicle_type:
            filtered_nodes = [
                node for node in nodes 
                if (hasattr(node.node, "metadata") and 
                    node.node.metadata.get("violation_type") in violation_types and
                    vehicle_type.lower() in node.text.lower())
            ]
            
            if filtered_nodes:
                logger.info(f"Filtered by both violation and vehicle types: {len(filtered_nodes)} nodes")
                return filtered_nodes
                
        # Otherwise try just violation type
        if violation_types:
            filtered_nodes = [
                node for node in nodes 
                if (hasattr(node.node, "metadata") and 
                    node.node.metadata.get("violation_type") in violation_types)
            ]
            
            if filtered_nodes:
                logger.info(f"Filtered by violation type: {len(filtered_nodes)} nodes")
                return filtered_nodes
                
        # Otherwise try just vehicle type
        if vehicle_type:
            filtered_nodes = [
                node for node in nodes 
                if vehicle_type.lower() in node.text.lower()
            ]
            
            if filtered_nodes:
                logger.info(f"Filtered by vehicle type: {len(filtered_nodes)} nodes")
                return filtered_nodes
        
        logger.info("No specific filtering applied or no matches found. Using all results.")
        return nodes