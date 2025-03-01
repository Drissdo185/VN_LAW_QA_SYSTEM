import requests
from bs4 import BeautifulSoup
import os
import re
from typing import List

class WebToMarkdown:
    def __init__(self, output_dir: str = "markdown_content"):
        """Initialize the web crawler"""
        self.output_dir = output_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        os.makedirs(output_dir, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        text = re.sub(r'\s+', ' ', text.strip())
        return text

    def extract_main_content(self, soup: BeautifulSoup) -> List[BeautifulSoup]:
        """Extract main content elements from the page"""
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'iframe', 'header', 'aside']):
            element.decompose()

        # Try to find main content area
        content_areas = soup.find_all(['article', 'main', 'div'], class_=re.compile(r'(content|article|post|entry)'))
        
        if content_areas:
            # Use the largest content area
            main_content = max(content_areas, key=lambda x: len(x.get_text()))
        else:
            main_content = soup.find('body') or soup

        # Return list of content elements
        return main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'pre', 'code'])

    def convert_element_to_markdown(self, element) -> str:
        """Convert a single HTML element to Markdown"""
        tag_name = element.name
        text = self.clean_text(element.get_text())
        
        if not text:
            return ""

        # Convert headings
        if tag_name.startswith('h'):
            level = int(tag_name[1])
            return f"{'#' * level} {text}\n"

        # Convert lists
        if tag_name in ('ul', 'ol'):
            items = []
            for li in element.find_all('li', recursive=False):
                items.append(f"* {self.clean_text(li.get_text())}")
            return '\n'.join(items) + '\n'

        # Convert code blocks
        if tag_name in ('pre', 'code'):
            return f"```\n{text}\n```\n"

        # Convert paragraphs
        return f"{text}\n"

    def crawl_and_save(self, url: str, index: int) -> bool:
        """
        Crawl webpage and save content as markdown
        
        Args:
            url: URL to crawl
            index: File index for naming
        
        Returns:
            bool: Success status
        """
        try:
            # Fetch page
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract and convert content
            markdown_content = []
            for element in self.extract_main_content(soup):
                markdown_content.append(self.convert_element_to_markdown(element))
            
            # Save content
            filename = f"page_{index}.md"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(''.join(markdown_content))
            
            return True
            
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return False

def crawl_urls(urls: List[str], output_dir: str = "markdown_content"):
    """
    Crawl multiple URLs and save as markdown
    
    Args:
        urls: List of URLs to crawl
        output_dir: Output directory for markdown files
    """
    crawler = WebToMarkdown(output_dir)
    
    for i, url in enumerate(urls, 1):
        success = crawler.crawl_and_save(url, i)
        if success:
            print(f"Successfully processed: {url}")
        else:
            print(f"Failed to process: {url}")

# Example usage
if __name__ == "__main__":
    urls = [
        "https://thuvienphapluat.vn/banan/tin-tuc/nam-2025-cho-cho-meo-tren-xe-may-co-bi-phat-khong-muc-phat-loi-cho-cho-meo-tren-xe-may-theo-nghi-dinh-168-la-bao-nhieu-13192"
    ]
    crawl_urls(urls)