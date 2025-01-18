import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

import torch
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode, ObjectType
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pyvi import ViTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VietnameseLegalEmbeddings:
    def __init__(
        self,
        embedding_model: str = "qducnguyen/vietnamese-bi-encoder",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        parent_chunk_size: int = 900,
        parent_chunk_overlap: int = 0,
        child_chunk_size: int = 600,
        child_chunk_overlap: int = 90
    ):
        """
        Initialize the Vietnamese Legal Document Embedding System.
        """
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            max_length=256,
            device=device
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        
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

    def clean_relationships(self, node: TextNode) -> TextNode:
        """
        Clean and fix node relationships to match expected schema.
        
        Args:
            node: TextNode to clean
            
        Returns:
            TextNode with fixed relationships
        """
        if NodeRelationship.SOURCE in node.relationships:
            related_info = node.relationships[NodeRelationship.SOURCE]
            
            # Create a new RelatedNodeInfo with all required fields
            fixed_info = RelatedNodeInfo(
                node_id=related_info.node_id,
                node_type=ObjectType.TEXT,  # Set appropriate type
                hash=related_info.hash,
                metadata={},  # Empty metadata to avoid serialization issues
            )
            
            # Update the node's relationships
            node.relationships[NodeRelationship.SOURCE] = fixed_info
            
        return node

    def load_documents(self, json_path: str) -> List[Document]:
        """
        Load documents from a JSON file.
        """
        try:
            with open(json_path) as f:
                all_data = json.load(f)
            
            documents = []
            empty_count = 0
            
            for doc_data in all_data:
                for child_data in doc_data["child_data"]:
                    if len(child_data["lower_segmented_text"].strip()):
                        documents.append(Document(
                            text=child_data["full_text"],
                            metadata={
                                "filename": doc_data["meta_data"]["file_name"],
                                "source": doc_data["meta_data"]["id_doc"] + ", " + ", ".join(child for child in child_data['pointer_link']),
                                "len_tokenizer": child_data["len_tokenizer"]
                            },
                            text_template="{content}",
                            excluded_embed_metadata_keys=["filename", "source", "len_tokenizer"],
                            excluded_llm_metadata_keys=["filename", "source", "len_tokenizer"]
                        ))
                    else:
                        empty_count += 1
            
            logger.info(f"Loaded {len(documents)} documents. Found {empty_count} empty documents.")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents through parent and child chunking.
        """
        try:
            # Create parent nodes
            logger.info("Creating parent nodes...")
            parent_nodes = self.parent_chunker.get_nodes_from_documents(
                documents,
                show_progress=True
            )
            
            # Add parent text to metadata
            logger.info("Processing parent nodes...")
            for parent_node in tqdm(parent_nodes):
                parent_node.metadata["parent_text"] = parent_node.text
                parent_node.excluded_embed_metadata_keys.append("parent_text")
                parent_node.excluded_llm_metadata_keys.append("parent_text")
            
            # Create child nodes
            logger.info("Creating child nodes...")
            child_nodes = self.child_chunker.get_nodes_from_documents(
                parent_nodes,
                show_progress=True
            )
            
            # Process child nodes
            logger.info("Processing child nodes...")
            processed_nodes = []
            for child_node in tqdm(child_nodes):
                # Apply Vietnamese tokenization
                child_node.text = ViTokenizer.tokenize(child_node.text.lower())
                
                # Clean relationships to fix schema issues
                cleaned_node = self.clean_relationships(child_node)
                processed_nodes.append(cleaned_node)
            
            return processed_nodes
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise

    def get_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings for a single text.
        """
        try:
            processed_text = ViTokenizer.tokenize(text.lower())
            return self.embed_model.get_text_embedding(processed_text)
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts.
        """
        try:
            processed_texts = [ViTokenizer.tokenize(text.lower()) for text in texts]
            return self.embed_model.get_text_embedding_batch(processed_texts)
        except Exception as e:
            logger.error(f"Error batch embedding texts: {e}")
            raise