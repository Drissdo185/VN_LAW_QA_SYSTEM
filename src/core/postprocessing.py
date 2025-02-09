from typing import List
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from logging import getLogger

logger = getLogger(__name__)

class PostProcessingPipeline:
    """Handles sequential post-processing of search results."""
    
    def __init__(self, processors: List[BaseNodePostprocessor]):
        """Initialize with a list of processors."""
        self.processors = processors
        
    def process(self, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
        """Apply all post-processors in sequence."""
        try:
            current_nodes = nodes
            for processor in self.processors:
                try:
                    current_nodes = processor.postprocess_nodes(
                        current_nodes,
                        query
                    )
                except Exception as e:
                    logger.error(
                        f"Error in post-processor {processor.__class__.__name__}: {e}"
                    )
                    continue
            return current_nodes
        except Exception as e:
            logger.error(f"Error in post-processing pipeline: {e}")
            return nodes