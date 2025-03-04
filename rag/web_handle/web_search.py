from typing import List, Dict, Any, Optional
import requests
import logging
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import NodeWithScore

from web_handle.web_data import WebToMarkdown
from retrieval.search_pipline import SearchPipeline
from config.config import ModelConfig, RetrievalConfig

# Setup logger
logger = logging.getLogger(__name__)

class WebSearchIntegrator:
    def __init__(
        self,
        google_api_key: str,
        google_cse_id: str,
        model_config: ModelConfig,
        retrieval_config: RetrievalConfig
    ):
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.model_config = model_config
        self.retrieval_config = retrieval_config
        self.web_crawler = WebToMarkdown()
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=215,
            chunk_overlap=0
        )
        logger.info("Initialized WebSearchIntegrator")

    async def search_and_retrieve(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """
        Perform web search and retrieve relevant content chunks
        
        Returns:
            Dict containing:
            - nodes: List[NodeWithScore] - selected nodes
            - url: str - source URL
            - search_results: List[Dict] - full search results
        """
        logger.info(f"Starting web search for query: {query}")
        
        # Get search results
        search_results = self._google_search(query, search_depth=1)
        if not search_results:
            logger.warning("No search results found")
            return {"nodes": [], "url": None, "search_results": []}

        # Get first URL and extract content
        first_url = search_results[0]['link']
        logger.info(f"Processing content from URL: {first_url}")
        
        content = self._extract_webpage_content(first_url)
        if not content:
            logger.warning(f"Failed to extract content from URL: {first_url}")
            return {"nodes": [], "url": first_url, "search_results": search_results}

        # Create document and chunk into nodes
        document = Document(text=content)
        nodes = self.node_parser.get_nodes_from_documents([document])
        logger.info(f"Created {len(nodes)} nodes from document")

        # Create temporary search pipeline for ranking chunks
        temp_pipeline = SearchPipeline(
            retriever=None,
            model_config=self.model_config,
            retrieval_config=self.retrieval_config
        )

        # Rerank nodes using cross-encoder
        nodes_with_scores = [NodeWithScore(node=node, score=0.0) for node in nodes]
        reranked_nodes = temp_pipeline._rerank_results(query, nodes_with_scores)

        # Get top N results
        selected_nodes = reranked_nodes[:num_results]
        
        # Log selected nodes
        logger.info(f"Selected {len(selected_nodes)} most relevant nodes:")
        for i, node in enumerate(selected_nodes, 1):
            logger.info(f"Node {i} (score: {node.score:.3f}):")
            logger.info(f"Text snippet: {node.text[:200]}...")

        return {
            "nodes": selected_nodes,
            "url": first_url,
            "search_results": search_results
        }

    def _google_search(self, query: str, search_depth: int = 2) -> List[Dict[str, Any]]:
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
            search_results = results.get('items', [])
            logger.info(f"Found {len(search_results)} search results")
            return search_results
        except Exception as e:
            logger.error(f"Error during Google search: {str(e)}")
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
            
            content = ''.join(markdown_content)
            logger.info(f"Successfully extracted {len(content)} characters of content")
            return content
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return None

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
        logger.info("Initialized WebEnabledAutoRAG")

    async def get_answer(self, question: str) -> Dict[str, Any]:
        """
        Get answer with web search fallback
        """
        logger.info(f"Processing question: {question}")
        
        # First try getting answer from vector store
        initial_response = await self.auto_rag.get_answer(question)
        logger.info("Received response from knowledge base")

        # Check if we need web search fallback
        if (
            initial_response.get("decision", "").lower() == "không tìm thấy đủ thông tin" or
            not initial_response.get("final_answer")
        ):
            logger.info("Insufficient answer from knowledge base, attempting web search")
            
            # Get web search results
            web_search_results = await self.web_search.search_and_retrieve(question)
            web_nodes = web_search_results["nodes"]
            
            if web_nodes:
                logger.info("Found relevant content from web search")
                
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
                token_usage = {
                    "input_tokens": self.auto_rag._count_tokens(prompt),
                    "output_tokens": self.auto_rag._count_tokens(web_response.text),
                    "total_tokens": (
                        self.auto_rag._count_tokens(prompt) + 
                        self.auto_rag._count_tokens(web_response.text)
                    )
                }
                
                parsed_response["token_usage"] = token_usage
                parsed_response["source"] = "web_search"
                parsed_response["llm_provider"] = self.auto_rag.model_config.llm_provider
                parsed_response["web_source"] = {
                    "url": web_search_results["url"],
                    "nodes": [
                        {
                            "text": node.text,
                            "score": node.score
                        }
                        for node in web_nodes
                    ]
                }
                
                logger.info(f"Generated response from web content (tokens: {token_usage['total_tokens']})")
                return parsed_response
            else:
                logger.warning("No relevant content found from web search")

        return initial_response