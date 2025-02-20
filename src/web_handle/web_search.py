from typing import List, Dict, Any, Optional
import os
import requests
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import NodeWithScore

from web_handle.web_data import WebToMarkdown
from retrieval.search_pipline import SearchPipeline
from config.config import ModelConfig, RetrievalConfig, Domain

class WebSearchIntegrator:
    def __init__(
        self,
        google_api_key: str,
        google_cse_id: str,
        model_config: ModelConfig,
        retrieval_config: RetrievalConfig,
        domain: Domain
    ):
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.model_config = model_config
        self.retrieval_config = retrieval_config
        self.domain = domain
        self.web_crawler = WebToMarkdown()
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=model_config.chunk_size,
            chunk_overlap=model_config.chunk_overlap
        )

    async def search_and_retrieve(self, query: str, num_results: int = 2) -> List[NodeWithScore]:
        """
        Perform web search and retrieve relevant content chunks
        """
        # Get search results
        search_results = self._google_search(query, search_depth=1)
        if not search_results:
            return []

        # Get first URL and extract content
        first_url = search_results[0]['link']
        content = self._extract_webpage_content(first_url)
        if not content:
            return []

        # Create document and chunk into nodes
        document = Document(text=content)
        nodes = self.node_parser.get_nodes_from_documents([document])

        # Create temporary search pipeline for ranking chunks
        temp_pipeline = SearchPipeline(
            retriever=None,  # We'll handle retrieval manually
            model_config=self.model_config,
            retrieval_config=self.retrieval_config,
            domain=self.domain
        )

        # Rerank nodes using cross-encoder
        nodes_with_scores = [NodeWithScore(node=node, score=0.0) for node in nodes]
        reranked_nodes = temp_pipeline._rerank_results(query, nodes_with_scores)

        # Return top N results
        return reranked_nodes[:num_results]

    def _google_search(self, query: str, search_depth: int = 1) -> List[Dict[str, Any]]:
        """
        Perform Google Custom Search
        """
        service_url = 'https://www.googleapis.com/customsearch/v1'
        params = {
            'q': query,
            'key': self.google_api_key,
            'cx': self.google_cse_id,
            'num': search_depth
        }

        try:
            response = requests.get(service_url, params=params)
            response.raise_for_status()
            results = response.json()
            return results.get('items', [])
        except Exception as e:
            print(f"Error during Google search: {str(e)}")
            return []

    def _extract_webpage_content(self, url: str) -> Optional[str]:
        """
        Extract main content from webpage
        """
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            content_elements = self.web_crawler.extract_main_content(soup)
            
            # Convert elements to markdown and join
            markdown_content = []
            for element in content_elements:
                markdown_content.append(
                    self.web_crawler.convert_element_to_markdown(element)
                )
            
            return ''.join(markdown_content)
        except Exception as e:
            print(f"Error extracting content from {url}: {str(e)}")
            return None

# Integration with AutoRAG
class WebEnabledAutoRAG:
    def __init__(
        self,
        auto_rag,
        web_search: WebSearchIntegrator,
        fallback_threshold: float = 0.5
    ):
        self.auto_rag = auto_rag
        self.web_search = web_search
        self.fallback_threshold = fallback_threshold

    async def get_answer(self, question: str) -> Dict[str, Any]:
        """
        Get answer with web search fallback
        """
        # First try getting answer from vector store
        initial_response = await self.auto_rag.get_answer(question)

        # Check if we need web search fallback
        if (
            initial_response.get("decision", "").lower() == "không tìm thấy đủ thông tin" or
            not initial_response.get("final_answer")
        ):
            # Get web search results
            web_nodes = await self.web_search.search_and_retrieve(question)
            if web_nodes:
                # Format context from web nodes
                web_context = self.auto_rag.retriever.get_formatted_context(web_nodes)
                
                # Generate new response with web content
                prompt = self.auto_rag.prompt_template.format(
                    question=question,
                    context=web_context
                )
                
                web_response = await self.auto_rag.llm.acomplete(prompt)
                parsed_response = self.auto_rag._parse_response(web_response.text)
                
                # Add token usage and source information
                parsed_response["token_usage"] = {
                    "input_tokens": self.auto_rag._count_tokens(prompt),
                    "output_tokens": self.auto_rag._count_tokens(web_response.text),
                    "total_tokens": (
                        self.auto_rag._count_tokens(prompt) + 
                        self.auto_rag._count_tokens(web_response.text)
                    )
                }
                parsed_response["source"] = "web_search"
                
                return parsed_response

        return initial_response