from typing import Dict, Any, List
from llama_index.core.schema import NodeRelationship, NodeWithScore
from core.types import SearchResult

class ResultFormatter:
    """Handles formatting of search results."""
    
    @staticmethod
    def format_search_results(
        results: 'SearchResult',
        include_raw_results: bool = False
    ) -> Dict[str, Any]:
        """Format search results for display or further processing."""
        formatted_results = {
            "question_type": results.question_type.value,
            "confidence_score": results.combined_score,
            "documents": ResultFormatter._format_documents(results.documents)
        }
        
        if include_raw_results and results.raw_results:
            formatted_results["raw_results"] = ResultFormatter._format_documents(
                results.raw_results
            )
            
        return formatted_results

    @staticmethod
    def _format_documents(nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
        """Format document information."""
        formatted_docs = []
        for node in nodes:
            doc_info = {
                "content": node.node.text,
                "metadata": node.node.metadata,
                "score": node.score,
                "node_id": node.node.node_id
            }
            
            if NodeRelationship.SOURCE in node.node.relationships:
                source = node.node.relationships[NodeRelationship.SOURCE]
                doc_info["parent_id"] = source.node_id
            
            formatted_docs.append(doc_info)
            
        return formatted_docs