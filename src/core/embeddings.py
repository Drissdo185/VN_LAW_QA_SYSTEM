import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import torch
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode, ObjectType
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pyvi import ViTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for a legal document."""
    filename: str
    source: str
    len_tokenizer: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        return cls(
            filename=data.get('filename', ''),
            source=data.get('source', ''),
            len_tokenizer=data.get('len_tokenizer', 0)
        )

class VietnameseLegalEmbeddings:
    """Handler for Vietnamese legal document embeddings."""
    
    def __init__(
        self,
        embedding_model: str = "dangvantuan/vietnamese-document-embedding",
        device: Optional[str] = None,
        parent_chunk_size: int = 900,
        parent_chunk_overlap: int = 0,
        child_chunk_size: int = 600,
        child_chunk_overlap: int = 90
    ):
        """Initialize the embedding system."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            max_length=256,
            device=self.device
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        
        # Initialize chunkers
        self.parent_chunker = TokenTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separator=" ",
            backup_separators=["__", "..", "--"],
            include_prev_next_rel=False
        )
        
        self.child_chunker = SentenceSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separator=" ",
            include_prev_next_rel=False
        )
        
        logger.info(f"Initialized embedding system with device: {self.device}")

    def load_documents(self, json_path: str | Path) -> List[Document]:
        """Load and preprocess documents from JSON."""
        try:
            json_path = Path(json_path)
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_path}")
                
            with json_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
                
            documents = []
            empty_count = 0
            
            for doc_data in data:
                for child in doc_data.get("child_data", []):
                    if text := child.get("lower_segmented_text", "").strip():
                        doc = Document(
                            text=child["full_text"],
                            metadata=DocumentMetadata(
                                filename=doc_data["meta_data"]["file_name"],
                                source=f"{doc_data['meta_data']['id_doc']}, {', '.join(child['pointer_link'])}",
                                len_tokenizer=child["len_tokenizer"]
                            ).__dict__,
                            text_template="{content}",
                            excluded_embed_metadata_keys=["filename", "source", "len_tokenizer"],
                            excluded_llm_metadata_keys=["filename", "source", "len_tokenizer"]
                        )
                        documents.append(doc)
                    else:
                        empty_count += 1
            
            logger.info(f"Loaded {len(documents)} documents. Skipped {empty_count} empty documents.")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

    def process_documents(self, documents: List[Document]) -> List[TextNode]:
        """Process documents through chunking pipeline."""
        try:
            # Create and process parent nodes
            parent_nodes = self.parent_chunker.get_nodes_from_documents(
                documents,
                show_progress=True
            )
            
            for node in tqdm(parent_nodes, desc="Processing parent nodes"):
                node.metadata["parent_text"] = node.text
                node.excluded_embed_metadata_keys.append("parent_text")
                node.excluded_llm_metadata_keys.append("parent_text")
            
            # Create and process child nodes
            child_nodes = self.child_chunker.get_nodes_from_documents(
                parent_nodes,
                show_progress=True
            )
            
            # Process child nodes
            processed_nodes = []
            for node in tqdm(child_nodes, desc="Processing child nodes"):
                node.text = ViTokenizer.tokenize(node.text.lower())
                processed_nodes.append(self._clean_node_relationships(node))
            
            logger.info(f"Processed {len(processed_nodes)} nodes")
            return processed_nodes
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    def _clean_node_relationships(self, node: TextNode) -> TextNode:
        """Clean and standardize node relationships."""
        if NodeRelationship.SOURCE in node.relationships:
            source = node.relationships[NodeRelationship.SOURCE]
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=source.node_id,
                node_type=ObjectType.TEXT,
                hash=source.hash,
                metadata={}
            )
        return node

    async def get_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts asynchronously."""
        try:
            processed_texts = [
                ViTokenizer.tokenize(text.lower())
                for text in texts
            ]
            return await self.embed_model.aget_text_embedding_batch(processed_texts)
        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            raise

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for a single text."""
        try:
            processed_text = ViTokenizer.tokenize(text.lower())
            return self.embed_model.get_text_embedding(processed_text)
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts."""
        try:
            processed_texts = [
                ViTokenizer.tokenize(text.lower())
                for text in texts
            ]
            return self.embed_model.get_text_embedding_batch(processed_texts)
        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            raise