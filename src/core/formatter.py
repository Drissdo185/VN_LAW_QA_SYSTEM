from typing import Dict, Any, List
from llama_index.core.schema import NodeRelationship, NodeWithScore
from core.types import SearchResult
from logging import getLogger

logger = getLogger(__name__)

class ResultFormatter:
    """Handles formatting of search results."""
    
    @staticmethod
    def format_results(results: SearchResult, include_raw: bool = False) -> Dict[str, Any]:
        """Format search results for output."""
        try:
            formatted = {
                "question_type": results.question_type.value,
                "confidence_score": results.combined_score,
                "documents": ResultFormatter._format_documents(results.documents)
            }
            
            if include_raw and results.raw_results:
                formatted["raw_results"] = ResultFormatter._format_documents(
                    results.raw_results,
                    include_metadata=False
                )
                
            return formatted
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            raise

    @staticmethod
    def _format_documents(
        nodes: List[NodeWithScore],
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Format document information with optional metadata."""
        formatted_docs = []
        for node in nodes:
            doc_info = {
                "content": node.node.text,
                "score": node.score,
                "node_id": node.node.node_id
            }
            
            if include_metadata:
                doc_info["metadata"] = node.node.metadata
                if NodeRelationship.SOURCE in node.node.relationships:
                    doc_info["parent_id"] = node.node.relationships[
                        NodeRelationship.SOURCE
                    ].node_id
            
            formatted_docs.append(doc_info)
            
        return formatted_docs