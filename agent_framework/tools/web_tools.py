# agent_framework/tools/web_tools.py
from typing import Dict, List, Any, Optional
from serpapi import GoogleSearch
from .base_tool import BaseTool
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class WebSearch(BaseTool):
    """Tool for performing web searches using Serper API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the web search tool with Serper API.
        
        Args:
            api_key: API key for the Serper API service (required)
        """
        super().__init__(
            name="web_search",
            description="Search the web for information on a given query using Serper API"
        )
        self.api_key = api_key or os.environ.get("SERPER_API_KEY")
    
    def execute(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Execute a web search using Serper API.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List of search results, each containing title, snippet, and URL
        """
        try:            
            if not self.api_key:
                return {"success": False, "error": "Serper API key is required. Please provide an API key."}

            params = {
                "api_key": self.api_key,
                "engine": "duckduckgo",
                "q": query,
                "kl": "us-en",
                "num": num_results
            }

            search = GoogleSearch(params)
            data = search.get_dict()
            
            content = []
            if 'organic_results' in data:
                for item in data['organic_results'][:num_results]:
                    content.append({
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'url': item.get('link', '')
                    })
                return {"success": True, "content": content}
            else:
                return {"success": False, "error": "No search results found."}
        except ImportError:
            return {"success": False, "error": "Requests package not installed. Install with 'pip install requests'"}
        except Exception as e:
            return {"success": False, "error": f"Search failed: {str(e)}"}

class WebScraper(BaseTool):
    """Tool for scraping content from web pages."""
    
    def __init__(self):
        """Initialize the web scraper tool."""
        super().__init__(
            name="web_scraper",
            description="Extract content from a webpage URL"
        )
    
    def execute(self, url: str, selector: str = None) -> Dict[str, Any]:
        """
        Scrape content from a webpage.
        
        Args:
            url: The URL to scrape
            selector: Optional CSS selector to target specific elements
            
        Returns:
            Dictionary containing the extracted content
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if selector:
                elements = soup.select(selector)
                content = [elem.get_text(strip=True) for elem in elements]
            else:
                # Extract title and main content
                title = soup.title.string if soup.title else "No title"
                
                # Try to extract main content
                main_content = ""
                for tag in ["main", "article", "div.content", "div.main"]:
                    content_elem = soup.select_one(tag)
                    if content_elem:
                        main_content = content_elem.get_text(strip=True)
                        break
                
                if not main_content:
                    main_content = soup.body.get_text(strip=True) if soup.body else "No content"
                
                content = {
                    "title": title,
                    "content": main_content[:5000]
                }
            
            return {"success": True, "content": content}
        
        except ImportError:
            return {"success": False, "error": "Required packages not installed. Install with 'pip install requests beautifulsoup4'"}
        except Exception as e:
            return {"success": False, "error": str(e)}
