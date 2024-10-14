import chromadb
import openai
import os
import logging
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from data_loader import DataLoader, transform_text 

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Indexing:
    def __init__(self, path: str = "./chroma_db", chunk_size: int = 200, chunk_overlap: int = 10):
        self.path = path
        self.db = chromadb.PersistentClient(path=self.path)
        self.collection_name = "quickstart"
        self.model = HuggingFaceEmbedding('all-MiniLM-L6-v2')  # Local embedding model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = None
        logger.info("Indexing instance created. ChromaDB path: %s", self.path)

    def create_index(self, documents):
        """
        Create an index and store document embeddings in Chroma DB.
        """
        try:
            chroma_collection = self.db.get_or_create_collection(self.collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            logger.info("Chroma collection and vector store initialized.")

            node_parser = SimpleNodeParser.from_defaults(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.model,
                transformations=[node_parser]
            )
            logger.info("Index created successfully with chunked documents.")

            storage_context.persist(persist_dir=self.path)
            logger.info("Index persisted to disk at: %s", self.path)

        except Exception as e:
            logger.error("Error creating the index: %s", str(e))

    def embed_and_index(self):
        """
        Load documents, generate embeddings, and create the index.
        """
        try:
            logger.info("Loading documents from directory: ./data")
            loader = DataLoader(directory="./data")
            docs = loader.load_data_sync()
            logger.info("Successfully loaded %d documents.", len(docs))

            logger.info("Transforming and cleaning the documents.")
            transformed_docs = transform_text(docs)

            logger.info("Creating index with chunking and storing in Chroma.")
            self.create_index(transformed_docs)

        except Exception as e:
            logger.error("Error during embedding and indexing: %s", str(e))

    def load_index(self):
        """
        Load the existing index from disk.
        """
        try:
            chroma_collection = self.db.get_collection(self.collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=self.path)
            
            self.index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context,
                embed_model=self.model
            )
            logger.info("Index loaded successfully from disk.")
        except Exception as e:
            logger.error("Error loading the index: %s", str(e))

    def retrieve(self, query: str):
        """
        Retrieve the relevant answer for the query.

        Args:
            query (str): The search query string.

        Returns:
            str: The relevant answer for the query.
        """
        try:
            logger.info("Retrieving answer for query: %s", query)

            # Load the index if it hasn't been loaded yet
            if self.index is None:
                self.load_index()

            # Create a query engine from the index
            query_engine = self.index.as_query_engine()

            # Execute the query
            response = query_engine.query(query)
            # Directly return the response since it's a string
            return response  # Assuming response is the generated answer

        except Exception as e:
            logger.error("Error during retrieval: %s", str(e))
            return "Sorry, I couldn't find relevant information."

    # def generate_answer(self, query: str):
    #     """
    #     Generate an answer using GPT-4 turbo.
    #
    #     Args:
    #         query (str): The user query.
    #
    #     Returns:
    #         str: Generated answer from GPT.
    #     """
    #     # First, retrieve the relevant answer
    #     retrieved_answer = self.retrieve(query)
    #
    #     if retrieved_answer == "Sorry, I couldn't find relevant information.":
    #         return retrieved_answer
    #
    #     # Prepare prompt for GPT
    #     prompt = f"Based on the following information, please answer the question:\n{retrieved_answer}\nQuestion: {query}"
    #
    #     try:
    #         # Ensure to have your OpenAI API key stored in the environment
    #         openai.api_key = os.getenv("OPENAI_API_KEY")
    #
    #         # Send the prompt to GPT-4-turbo (or another GPT model)
    #         response = openai.ChatCompletion.create(
    #             model="gpt-4-turbo",
    #             messages=[
    #                 {"role": "system", "content": "You are a helpful assistant."},
    #                 {"role": "user", "content": prompt}
    #             ]
    #         )
    #
    #         # Extract and return the response from GPT
    #         answer = response['choices'][0]['message']['content']
    #         return answer
    #
    #     except Exception as e:
    #         logger.error("Error generating response: %s", str(e))
    #         return "Error generating response."
# Create an instance of the Indexing class
if __name__ == "__main__":
    index_manager = Indexing(path="./chroma_db", chunk_size=200, chunk_overlap=10)
    
     # Embed and index documents
    index_manager.embed_and_index()  # Make sure documents are indexed first

    # Generate a response using a query
    answer = index_manager.retrieve(query="List all univeristy in partnership programs")
    print(answer)
