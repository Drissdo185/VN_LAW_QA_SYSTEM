from typing import List
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class PostProcessingPipeline:
    """Handles sequential post-processing of search results."""
    
    def __init__(
        self,
        processors: List[BaseNodePostprocessor]
    ):
        self.processors = processors
        
    def process(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """Apply all post-processors in sequence."""
        current_nodes = nodes
        for processor in self.processors:
            try:
                current_nodes = processor.postprocess_nodes(
                    current_nodes,
                    query_bundle
                )
            except Exception as e:
                logger.error(f"Error in post-processor {processor.class_name()}: {e}")
                continue
        return current_nodes