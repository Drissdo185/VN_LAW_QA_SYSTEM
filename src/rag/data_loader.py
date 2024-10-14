from llama_index.core import SimpleDirectoryReader, Document
from typing import List
import logging
import os
import asyncio
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Text transformation function
def transform_text(documents: List[Document]) -> List[Document]:
    """
    Cleans, tokenizes, removes stopwords, and lemmatizes the text in each document.
    
    Args:
        documents (List[Document]): List of Document objects to transform.
        
    Returns:
        List[Document]: Transformed documents with processed text.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    transformed_documents = []

    for doc in documents:
        try:
            # Extract text and convert to lowercase
            text = doc.text.lower()

            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            # Tokenize text
            tokens = nltk.word_tokenize(text)

            # Remove stopwords and apply lemmatization
            cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

            # Reassemble cleaned text
            cleaned_text = ' '.join(cleaned_tokens)

            # Create a new Document with the transformed text
            transformed_doc = Document(text=cleaned_text, metadata=doc.metadata)
            transformed_documents.append(transformed_doc)

        except Exception as e:
            logger.error(f"Error processing document {doc.metadata.get('file_name', 'unknown')}: {str(e)}")
            continue

    return transformed_documents

# DataLoader class for reading documents asynchronously
class DataLoader:
    def __init__(self, directory: str = "./data"):
        """
        Initializes the DataLoader with a directory from which to load documents.
        
        Args:
            directory (str): The directory containing documents to load.
        """
        self.directory = directory

    async def load_document(self, file_path: str) -> Document:
        """
        Asynchronously loads a single document from a file path.
        
        Args:
            file_path (str): The path to the file to load.
        
        Returns:
            Document: The loaded document, or None if an error occurs.
        """
        try:
            reader = SimpleDirectoryReader(input_files=[file_path])
            docs = reader.load_data()

            if docs:
                logger.info(f"Successfully loaded {file_path}")
                return docs[0]  # Assuming one document per file
            else:
                logger.warning(f"No data found in file {file_path}")
                return None

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return None

    async def load_data(self) -> List[Document]:
        """
        Asynchronously loads all documents from the specified directory.
        
        Returns:
            List[Document]: A list of loaded documents.
        """
        if not os.path.exists(self.directory):
            logger.error(f"Directory {self.directory} does not exist.")
            return []

        # Gather all file paths in the directory
        file_paths = [os.path.join(self.directory, f) for f in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, f))]

        # Load documents asynchronously
        tasks = [self.load_document(file_path) for file_path in file_paths]
        documents = await asyncio.gather(*tasks)

        # Filter out any None results (failed loads)
        documents = [doc for doc in documents if doc is not None]

        logger.info(f"Successfully loaded {len(documents)} documents from {self.directory}")
        return documents

    def load_data_sync(self) -> List[Document]:
        """
        Synchronously loads all documents using asyncio.
        
        Returns:
            List[Document]: A list of loaded documents.
        """
        return asyncio.run(self.load_data())

