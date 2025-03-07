from typing import List
from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore
from pyvi import ViTokenizer
from config.config import ModelConfig, RetrievalConfig
from retrieval.retriever import DocumentRetriever

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
        self.cross_encoder = CrossEncoder(
            model_config.cross_encoder_model,
            device=model_config.device,
            trust_remote_code=True
        )
      
    def search(self, query: str) -> List[NodeWithScore]:
        """
        Execute the full search pipeline
        
        Steps:
        1. Initial hybrid retrieval (BM25 + Dense)
        2. Metadata filtering
        3. Cross-encoder reranking
        4. Term-based ranking
        5. Return top 3 results
        """
        # Step 1: Initial retrieval
        initial_results = self.retriever.retrieve(query)
        
        # Step 2: Apply metadata filtering
        filtered_results = self._perform_metadata_filtering(initial_results, query)
        
        # Step 3: Rank results based on relevance to query
        final_results = self._rank_results(query, filtered_results)
    
        
        # Return top results
        return final_results[:3]
    
    def _rank_results(self, nodes, query):
        """
        Rank results based on relevance to query
        """
        
        
        query_tokens = set(ViTokenizer.tokenize(query.lower()).split())
        
        scored_nodes = []
        for node in nodes:
            # Calculate term overlap
            node_tokens = set(node.text.split())
            overlap = len(query_tokens.intersection(node_tokens))
            
            # Calculate score based on vector similarity and term overlap
            combined_score = (0.7 * node.score) + (0.3 * (overlap / max(1, len(query_tokens))))
            
            scored_nodes.append((node, combined_score))
        
        # Sort by combined score
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return [node for node, _ in scored_nodes]

    def _perform_metadata_filtering(sefl, nodes, query):
        """
        Add comprehensive metadata-based filtering to improve relevance
        """
        query_lower = query.lower()
        
        # Detect violation types in query with expanded keywords
        violation_types = []
        
        # Child safety violations
        if any(keyword in query_lower for keyword in ["trẻ em", "trẻ nhỏ", "1,35 mét", "mầm non", "học sinh", 
                                                    "dưới 10 tuổi", "chiều cao", "ghế trẻ em"]):
            violation_types.append("trẻ_em")
        
        # Speed violations
        if any(keyword in query_lower for keyword in ["tốc độ", "km/h", "chạy quá", "vượt quá tốc độ", 
                                                    "tốc độ tối đa", "tốc độ tối thiểu", "chạy nhanh"]):
            violation_types.append("tốc_độ")
        
        # Alcohol violations
        if any(keyword in query_lower for keyword in ["nồng độ cồn", "cồn", "rượu", "bia", "miligam", 
                                                    "nồng độ trong máu", "hơi thở", "say xỉn"]):
            violation_types.append("nồng_độ_cồn")
        
        # Drug violations
        if any(keyword in query_lower for keyword in ["ma túy", "chất kích thích", "chất gây nghiện", 
                                                    "chất ma túy", "dương tính"]):
            violation_types.append("ma_túy")
        
        # Parking violations
        if any(keyword in query_lower for keyword in ["đỗ xe", "đậu xe", "dừng xe", "đỗ trái phép", 
                                                    "đỗ sai quy định", "vạch kẻ đường", "lề đường", "vỉa hè"]):
            violation_types.append("đỗ_dừng_xe")
        
        # Document violations
        if any(keyword in query_lower for keyword in ["giấy phép", "chứng nhận", "đăng ký", "đăng kiểm", 
                                                    "bảo hiểm", "giấy tờ", "không mang theo", "hết hạn"]):
            violation_types.append("giấy_tờ")
        
        # Highway violations
        if any(keyword in query_lower for keyword in ["đường cao tốc", "cao tốc", "làn khẩn cấp", 
                                                    "vào cao tốc", "ra cao tốc"]):
            violation_types.append("đường_cao_tốc")
        
        # Traffic signals violations
        if any(keyword in query_lower for keyword in ["đèn tín hiệu", "đèn giao thông", "đèn đỏ", 
                                                    "vượt đèn", "không chấp hành", "biển báo"]):
            violation_types.append("biển_báo")
        
        # Lane violations
        if any(keyword in query_lower for keyword in ["làn đường", "chuyển làn", "làn xe", "phần đường", 
                                                    "vượt làn", "đi sai làn"]):
            violation_types.append("làn_đường")
        
        # Passenger violations
        if any(keyword in query_lower for keyword in ["chở người", "quá số người", "chở quá", "người ngồi", 
                                                    "chở trên mui", "thùng xe"]):
            violation_types.append("chở_người")
        
        # Cargo violations
        if any(keyword in query_lower for keyword in ["chở hàng", "trọng tải", "quá tải", "hàng hóa", 
                                                    "vượt kích thước", "kích thước giới hạn"]):
            violation_types.append("chở_hàng")
        
        # Phone usage violations
        if any(keyword in query_lower for keyword in ["điện thoại", "nghe điện thoại", "nhắn tin", 
                                                    "sử dụng điện thoại", "thiết bị điện tử"]):
            violation_types.append("điện_thoại")
        
        # Dangerous driving
        if any(keyword in query_lower for keyword in ["lạng lách", "đánh võng", "nguy hiểm", "liều lĩnh", 
                                                    "không giữ khoảng cách", "vô lăng"]):
            violation_types.append("lái_xe_nguy_hiểm")
        
        # Accident related
        if any(keyword in query_lower for keyword in ["tai nạn", "va chạm", "không dừng lại", 
                                                    "bỏ chạy", "không trợ giúp", "trốn khỏi"]):
            violation_types.append("tai_nạn")
        
        # Traffic lights & signs
        if any(keyword in query_lower for keyword in ["biển báo", "biển hiệu", "không chấm hành", 
                                                    "hiệu lệnh", "đèn tín hiệu"]):
            violation_types.append("biển_báo")
        
        # Overtaking violations
        if any(keyword in query_lower for keyword in ["vượt xe", "không được vượt", "vượt bên phải", 
                                                    "vượt ẩu", "vượt trái phép"]):
            violation_types.append("vượt_xe")
        
        # Safety equipment violations
        if any(keyword in query_lower for keyword in ["dây an toàn", "mũ bảo hiểm", "thiết bị an toàn", 
                                                    "không đội mũ", "không thắt dây"]):
            violation_types.append("an_toàn")
        
        # Technical violations
        if any(keyword in query_lower for keyword in ["kính chắn gió", "biển số", "gương chiếu hậu", 
                                                    "bánh lốp", "hệ thống phanh", "thiết bị"]):
            violation_types.append("kỹ_thuật_xe")
        
        # Environmental violations
        if any(keyword in query_lower for keyword in ["khói", "bụi", "ô nhiễm", "rơi vãi", "tiếng ồn", 
                                                    "xả thải", "môi trường"]):
            violation_types.append("môi_trường")
        
        # Backing up and turning around violations
        if any(keyword in query_lower for keyword in ["lùi xe", "quay đầu", "đảo chiều", "quay xe", 
                                                    "điểm quay đầu", "cấm quay đầu"]):
            violation_types.append("lùi_quay_đầu")
        
        # Intersection violations
        if any(keyword in query_lower for keyword in ["giao nhau", "ngã ba", "ngã tư", "giao lộ", 
                                                    "vòng xuyến", "đường ưu tiên"]):
            violation_types.append("giao_nhau")
        
        # Emergency vehicle violations
        if any(keyword in query_lower for keyword in ["xe ưu tiên", "xe cứu thương", "xe cứu hỏa", 
                                                    "xe công an", "xe quân sự", "không nhường đường"]):
            violation_types.append("xe_ưu_tiên")
        
        # Remove duplicates
        violation_types = list(set(violation_types))
        
        # If violation types detected, filter results
        if violation_types:
            print(f"Detected violation types: {violation_types}")
            filtered_nodes = [
                node for node in nodes 
                if "violation_type" in node.metadata and node.metadata["violation_type"] in violation_types
            ]
            if filtered_nodes:
                print(f"Filtered results from {len(nodes)} to {len(filtered_nodes)} nodes")
                return filtered_nodes
        
        print("No specific violation type detected or filtering yielded no results. Using all results.")
        return nodes 