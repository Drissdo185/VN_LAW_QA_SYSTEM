import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from llama_index.core.schema import NodeWithScore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuestionType(Enum):
    LEGAL = "legal"
    STANDALONE = "standalone"

@dataclass
class SearchResult:
    """Data class to hold search results and scores."""
    documents: List[NodeWithScore]
    combined_score: float
    question_type: QuestionType
    raw_results: Optional[List[NodeWithScore]] = None