import json
import logging
from typing import List, Dict, Optional

import torch
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pyvi import ViTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class VietnameseLegalEmbeddings:
    def __init__(
            self,
            embedding_model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
            parent_chunk_size: int = 900,
            parent_chunk_overlap: int = 0,
            child_chunk_size: int = 600,
            child_chunk_overlap: int = 90,
            device: str = "cuda" if torch.cuda.is_available() else " cpu"
            ):
        """
        Initialize the Vietnamese Legal Document Embedding System.

        Args:
            embedding_model (str): HuggingFace model name for embeddings
            device (str): Device to run embeddings on ('cuda' or 'cpu')
            parent_chunk_size (int): Size of parent chunks
            parent_chunk_overlap (int): Overlap between parent chunks
            child_chunk_size (int): Size of child chunks
            child_chunk_overlap (int): Overlap between child chunks
        """

        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            max_length = 256,
            device = device
        )

        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

        self.parent_chunker = TokenTextSplitter(
            chunk_size = parent_chunk_size,
            chunk_overlap= parent_chunk_overlap,
            separator= " ",
            backup_separators=["__", "..", "--"],
            include_prev_next_rel=False
        )

        self.child_chunker = SentenceSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separator=" ",
            include_prev_next_rel=False
        )

    def load_documents(self, json_path: str) -> List[Document]:
        """
        Load documents from a JSON file.
        
        Args:
            json_path (str): Path to JSON file containing documents
            
        Returns:
            List[Document]: List of processed documents
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
        
        Args:
            documents (List[Document]): List of documents to process
            
        Returns:
            List[Document]: List of processed child nodes
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
            
            # Process child nodes - tokenize and lowercase text
            logger.info("Processing child nodes...")
            for child_node in tqdm(child_nodes):
                child_node.text = ViTokenizer.tokenize(child_node.text.lower())
                try:
                    # Clean up metadata in relationships
                    del child_node.relationships[NodeRelationship.SOURCE].metadata
                except AttributeError:
                    continue
            
            return child_nodes
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise

    def get_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings for a single text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            # Preprocess text same way as documents
            processed_text = ViTokenizer.tokenize(text.lower())
            return self.embed_model.get_text_embedding(processed_text)
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            # Preprocess all texts
            processed_texts = [ViTokenizer.tokenize(text.lower()) for text in texts]
            return self.embed_model.get_text_embedding_batch(processed_texts)
        except Exception as e:
            logger.error(f"Error batch embedding texts: {e}")
            raise

        