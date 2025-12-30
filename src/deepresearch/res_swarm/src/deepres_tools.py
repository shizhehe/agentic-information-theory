"""
DeepResearch Web Search Tools

Premium web search using SerpAPI (Google) and Jina Reader API for high-quality research.
Adapted from minions utilities for DeepResearch Bench integration.
"""

import os
import time
import re
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from threading import Lock

# Search API imports
try:
    import serpapi
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False
    print("⚠️ SerpAPI not installed. Install with: pip install serpapi")


class WebSearchTool:
    """
    Advanced web search tool with parallel processing and content quality filtering.
    
    Features:
    - Parallel web scraping for improved performance
    - Content quality assessment and filtering
    - Smart retry logic with exponential backoff
    - Domain filtering and content type validation
    - Thread-safe operation with connection pooling
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        timeout: int = 30,
        max_workers: int = 10,
        min_content_length: int = 200,
        content_quality_threshold: float = 0.3,
    ):
        """
        Initialize advanced web search tool.
        
        Args:
            max_retries: Maximum retry attempts for failed requests
            timeout: Timeout for web requests in seconds
            max_workers: Maximum number of parallel workers for scraping
            min_content_length: Minimum content length to consider valid
            content_quality_threshold: Minimum quality score for content (0-1)
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_workers = max_workers
        self.min_content_length = min_content_length
        self.content_quality_threshold = content_quality_threshold
        self._session_lock = Lock()
        self._session = None
        
        # API Configuration
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.jina_api_key = os.getenv("JINA_API_KEY")
        
        # Log which search methods are available
        if self.serpapi_key and SERPAPI_AVAILABLE:
            print("✅ Using SerpAPI for search (Google backend)")
        else:
            print("❌ SerpAPI not available - search will fail without API key and package")
            
        if self.jina_api_key:
            print("✅ Using Jina Reader API for content extraction (supports PDFs)")
        else:
            print("⚠️ No JINA_API_KEY found, content extraction will be limited")
    
    def search_and_scrape(
        self, 
        query: str, 
        max_results: int = 5,
        scrape_content: bool = True,
        parallel_scraping: bool = True,
        quality_filter: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search the web and optionally scrape content from results with parallel processing.
        
        Args:
            query: Search query string
            max_results: Maximum number of search results to return
            scrape_content: Whether to scrape full content from URLs
            parallel_scraping: Whether to use parallel processing for scraping
            quality_filter: Whether to apply content quality filtering
            
        Returns:
            List of search results with optional scraped content, sorted by quality
        """
        print(f"🔍 Searching for: {query}")
        
        # Step 1: Get search URLs
        search_urls = self._search_web(query, max_results)
        
        if not search_urls:
            print(f"❌ No search results found for: {query}")
            return []
        
        print(f"✅ Found {len(search_urls)} search results")
        
        # Step 2: Scrape content if requested
        results = []
        for url_data in search_urls:
            result = {
                "url": url_data["url"],
                "title": url_data.get("title", ""),
                "snippet": url_data.get("snippet", ""),
                "content": "",
                "scraped": False,
                "error": None,
                "quality_score": 0.0,
                "word_count": 0,
                "scrape_time": 0.0
            }
            results.append(result)
        
        if scrape_content:
            if parallel_scraping and len(search_urls) > 1:
                print(f"🚀 Starting parallel scraping for {len(search_urls)} URLs")
                results = self._scrape_urls_parallel(results)
            else:
                print(f"🌐 Starting sequential scraping for {len(search_urls)} URLs")
                results = self._scrape_urls_sequential(results)
            
            # Apply content quality filtering
            if quality_filter:
                results = self._filter_by_quality(results)
                print(f"📊 Quality filtering: {len(results)} results remain")
        
        return results
    
    def _scrape_urls_parallel(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs in parallel using ThreadPoolExecutor.
        
        Args:
            results: List of result dictionaries to populate with scraped content
            
        Returns:
            Updated results list with scraped content
        """
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit scraping tasks
            future_to_result = {}
            for result in results:
                if self._is_valid_url(result["url"]):
                    future = executor.submit(self._scrape_url_with_timing, result["url"])
                    future_to_result[future] = result
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_result, timeout=self.timeout * 2):
                result = future_to_result[future]
                try:
                    content_data = future.result(timeout=5.0)
                    if content_data:
                        result["content"] = content_data.get("markdown", "")
                        result["scraped"] = True
                        result["scrape_time"] = content_data.get("scrape_time", 0.0)
                        result["word_count"] = len(result["content"].split())
                        result["quality_score"] = self._assess_content_quality(result["content"])
                        print(f"✅ Scraped {result['url']} ({result['word_count']} words, quality: {result['quality_score']:.2f})")
                    else:
                        result["error"] = "Failed to scrape content"
                        print(f"❌ Failed to scrape: {result['url']}")
                except Exception as e:
                    result["error"] = f"Scraping error: {str(e)}"
                    print(f"❌ Error scraping {result['url']}: {e}")
        
        total_time = time.time() - start_time
        successful_scrapes = sum(1 for r in results if r["scraped"])
        print(f"🚀 Parallel scraping completed: {successful_scrapes}/{len(results)} successful in {total_time:.1f}s")
        
        return results
    
    def _scrape_urls_sequential(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scrape URLs sequentially (fallback method).
        
        Args:
            results: List of result dictionaries to populate with scraped content
            
        Returns:
            Updated results list with scraped content
        """
        for i, result in enumerate(results):
            print(f"🌐 Scraping {i+1}/{len(results)}: {result['url']}")
            
            content_data = self._scrape_url_with_timing(result["url"])
            if content_data:
                result["content"] = content_data.get("markdown", "")
                result["scraped"] = True
                result["scrape_time"] = content_data.get("scrape_time", 0.0)
                result["word_count"] = len(result["content"].split())
                result["quality_score"] = self._assess_content_quality(result["content"])
                print(f"✅ Successfully scraped content ({result['word_count']} words, quality: {result['quality_score']:.2f})")
            else:
                result["error"] = "Failed to scrape content"
                print(f"❌ Failed to scrape: {result['url']}")
        
        return results
    
    def _scrape_url_with_timing(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape URL with timing information.
        
        Args:
            url: URL to scrape
            
        Returns:
            Content data with timing information
        """
        start_time = time.time()
        content_data = self._scrape_url(url)
        
        if content_data:
            content_data["scrape_time"] = time.time() - start_time
        
        return content_data
    
    def _search_web_serpapi(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search using SerpAPI (Google backend) for high-quality, spam-free results.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results with URL, title, and snippet
        """
        if not self.serpapi_key or not SERPAPI_AVAILABLE:
            print("❌ SerpAPI not available - missing API key or package not installed")
            return []
        
        try:
            # Create SerpAPI client
            client = serpapi.Client(api_key=self.serpapi_key)
            
            print(f"🔍 Searching with SerpAPI: {query}")
            results = client.search(
                q=query,
                engine="google",
                num=min(num_results, 20),  # Google returns max 20 per request
                gl="us",  # Location (US for English results)
                hl="en",  # Language
                safe="off",  # Don't filter results
                no_cache="false"  # Use cache for faster results
            )
            
            if "error" in results:
                print(f"❌ SerpAPI error: {results['error']}")
                return []
            
            # Extract organic results
            organic_results = results.get("organic_results", [])
            if not organic_results:
                print("❌ No organic results found in SerpAPI response")
                return []
            
            # Convert to expected format
            search_results = []
            for i, result in enumerate(organic_results[:num_results]):
                search_results.append({
                    "url": result.get("link", ""),
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", "")
                })
            
            print(f"✅ SerpAPI returned {len(search_results)} high-quality results")
            return search_results
            
        except Exception as e:
            print(f"❌ Error with SerpAPI: {e}")
            return []
    
    def _scrape_url_jina(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Use Jina Reader API to extract clean content from URLs, including PDFs.
        
        Args:
            url: URL to scrape content from
            
        Returns:
            Dictionary with scraped content or None if failed
        """
        if not self.jina_api_key:
            print("Jina API key not available, content extraction skipped")
            return None
        
        if not self._is_valid_url(url):
            print(f"Invalid URL for Jina: {url}")
            return None
        
        try:
            import requests
            
            # Jina Reader API endpoint
            jina_url = f"https://r.jina.ai/{url}"
            
            headers = {
                "Authorization": f"Bearer {self.jina_api_key}",
                "Accept": "application/json",
                "User-Agent": "ResSwarm/1.0 (DeepResearch)"
            }
            
            print(f"📖 Reading with Jina API: {url}")
            start_time = time.time()
            
            response = requests.get(jina_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            scrape_time = time.time() - start_time
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    content = data.get("data", {}).get("content", "")
                    title = data.get("data", {}).get("title", "")
                    
                    if not content:
                        print(f"No content extracted by Jina for {url}")
                        return None
                    
                    # Check if content meets minimum length requirement
                    if len(content.strip()) < self.min_content_length:
                        print(f"Jina content too short for {url}: {len(content.strip())} chars")
                        return None
                    
                    word_count = len(content.split())
                    is_pdf = url.lower().endswith('.pdf') or 'pdf' in url.lower()
                    
                    result = {
                        "markdown": content,
                        "content": content,  # For compatibility with existing code
                        "html": "",  # Jina provides markdown, not HTML
                        "url": url,
                        "title": title,
                        "content_length": len(content),
                        "word_count": word_count,
                        "scrape_time": scrape_time,
                        "is_pdf": is_pdf,
                        "source": "jina"
                    }
                    
                    print(f"✅ Jina extracted {word_count} words from {url} in {scrape_time:.1f}s")
                    return result
                    
                except (KeyError, ValueError) as e:
                    print(f"Failed to parse Jina response for {url}: {e}")
                    return None
            else:
                print(f"Jina API returned status {response.status_code} for {url}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Jina API request failed for {url}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error with Jina API for {url}: {e}")
            return None
    
    def _assess_content_quality(self, content: str) -> float:
        """
        Assess the quality of scraped content using multiple heuristics.
        
        Args:
            content: Scraped content to assess
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not content or len(content.strip()) < self.min_content_length:
            return 0.0
        
        score = 0.0
        factors = 0
        
        # Factor 1: Content length (longer content typically better, up to a point)
        length = len(content)
        if length > 500:
            score += min(1.0, length / 2000.0) * 0.3
        factors += 0.3
        
        # Factor 2: Sentence structure (presence of proper sentences)
        sentences = re.split(r'[.!?]+', content)
        proper_sentences = [s for s in sentences if len(s.strip().split()) > 3]
        if len(sentences) > 0:
            sentence_quality = len(proper_sentences) / len(sentences)
            score += sentence_quality * 0.2
        factors += 0.2
        
        # Factor 3: Word variety (unique words vs total words)
        words = content.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += min(unique_ratio * 2, 1.0) * 0.2
        factors += 0.2
        
        # Factor 4: Presence of navigation/boilerplate text (negative factor)
        boilerplate_patterns = [
            r'\bnavigation\b', r'\bmenu\b', r'\bcopyright\b', r'\bprivacy policy\b',
            r'\bterms of service\b', r'\bcookie\b', r'\badvertisement\b',
            r'\bsubscribe\b', r'\bsign up\b', r'\blog in\b'
        ]
        boilerplate_count = sum(len(re.findall(pattern, content.lower())) for pattern in boilerplate_patterns)
        boilerplate_penalty = min(boilerplate_count / 10.0, 0.3)
        score -= boilerplate_penalty * 0.15
        factors += 0.15
        
        # Factor 5: Paragraph structure
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        substantial_paragraphs = [p for p in paragraphs if len(p.split()) > 10]
        if paragraphs:
            paragraph_quality = len(substantial_paragraphs) / len(paragraphs)
            score += paragraph_quality * 0.15
        factors += 0.15
        
        # Normalize score
        final_score = max(0.0, min(1.0, score / factors if factors > 0 else 0.0))
        
        return final_score
    
    def _filter_by_quality(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter results by content quality and sort by quality score.
        
        Args:
            results: List of search results with quality scores
            
        Returns:
            Filtered and sorted results list
        """
        # Filter by quality threshold
        quality_results = [
            r for r in results 
            if r.get("scraped", False) and 
               r.get("quality_score", 0.0) >= self.content_quality_threshold and
               r.get("word_count", 0) >= self.min_content_length // 10  # Roughly 20 words minimum
        ]
        
        # Include non-scraped results (for reference)
        non_scraped = [r for r in results if not r.get("scraped", False)]
        
        # Sort quality results by score (descending)
        quality_results.sort(key=lambda x: x.get("quality_score", 0.0), reverse=True)
        
        # Combine quality results with non-scraped (quality first)
        return quality_results + non_scraped
    
    def get_session(self):
        """
        Get or create a session for HTTP requests with connection pooling.
        
        Returns:
            requests.Session object
        """
        with self._session_lock:
            if self._session is None:
                try:
                    import requests
                    from requests.adapters import HTTPAdapter
                    from urllib3.util.retry import Retry
                    
                    self._session = requests.Session()
                    
                    # Configure retry strategy
                    retry_strategy = Retry(
                        total=self.max_retries,
                        backoff_factor=1,
                        status_forcelist=[429, 500, 502, 503, 504],
                    )
                    
                    # Mount adapter with retry strategy
                    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
                    self._session.mount("http://", adapter)
                    self._session.mount("https://", adapter)
                    
                except ImportError:
                    print("requests library not available")
                    return None
            
            return self._session
    
    
    def _search_web(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search the web using SerpAPI (Google backend) for high-quality results.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results with URL, title, and snippet
        """
        return self._search_web_serpapi(query, num_results)
    
    
    def _scrape_url(self, url: str, use_session: bool = True) -> Optional[Dict[str, Any]]:
        """
        Scrape content from a URL using Jina Reader API.
        
        Args:
            url: URL to scrape
            use_session: Kept for backward compatibility (not used)
            
        Returns:
            Dictionary with scraped content or None if failed
        """
        # Use Jina Reader API (supports PDFs and gives cleaner content)
        if self.jina_api_key:
            result = self._scrape_url_jina(url)
            if result and result.get("markdown"):
                return result
        else:
            print("❌ Content extraction failed - JINA_API_KEY not configured")
        
        return None
    
    def _clean_scraped_content(self, content: str) -> str:
        """
        Advanced cleaning of scraped markdown content.
        
        Args:
            content: Raw markdown content
            
        Returns:
            Cleaned markdown content
        """
        if not content:
            return ""
        
        # Split into lines for processing
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip lines that are likely navigation/boilerplate
            skip_patterns = [
                r'^#+\s*(menu|navigation|nav|sidebar|footer|header)\s*$',
                r'^\*+\s*(home|about|contact|login|register|subscribe)\s*\*+$',
                r'^\s*\|.*\|\s*$',  # Table separators
                r'^\s*[-=]{3,}\s*$',  # Horizontal rules
                r'^\s*[\*\-\+]\s*$',  # Single list bullets
            ]
            
            if any(re.match(pattern, line, re.I) for pattern in skip_patterns):
                continue
            
            # Skip very short lines unless they're headers
            if len(line) < 10 and not line.startswith('#'):
                continue
            
            # Skip lines with excessive punctuation (likely navigation)
            punct_ratio = sum(1 for c in line if not c.isalnum() and not c.isspace()) / len(line)
            if punct_ratio > 0.5 and len(line) < 50:
                continue
            
            cleaned_lines.append(line)
        
        # Join lines back together
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Remove excessive whitespace
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
        
        # Remove common footer/header patterns
        footer_patterns = [
            r'\n.*?copyright.*?\n',
            r'\n.*?all rights reserved.*?\n',
            r'\n.*?terms of service.*?\n',
            r'\n.*?privacy policy.*?\n',
        ]
        
        for pattern in footer_patterns:
            cleaned_content = re.sub(pattern, '\n', cleaned_content, flags=re.I | re.DOTALL)
        
        return cleaned_content.strip()
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and scrapeable with enhanced filtering."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Skip certain file types that are unlikely to contain useful text
            skip_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                             '.zip', '.tar', '.gz', '.jpg', '.jpeg', '.png', '.gif', 
                             '.mp4', '.mp3', '.wav', '.avi', '.mov', '.wmv', '.flv',
                             '.swf', '.exe', '.dmg', '.iso', '.bin']
            
            if any(url.lower().endswith(ext) for ext in skip_extensions):
                return False
            
            # Skip known problematic domains
            skip_domains = ['youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com',
                          'facebook.com', 'twitter.com', 'instagram.com', 'tiktok.com',
                          'pinterest.com', 'reddit.com']
            
            domain = parsed.netloc.lower()
            if any(skip_domain in domain for skip_domain in skip_domains):
                return False
            
            # Check for suspicious URL patterns
            suspicious_patterns = [r'/login', r'/register', r'/signup', r'/auth', 
                                 r'/admin', r'/dashboard', r'/profile']
            
            if any(re.search(pattern, url.lower()) for pattern in suspicious_patterns):
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_content_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistics about scraped content quality.
        
        Args:
            results: List of search results
            
        Returns:
            Dictionary with content statistics
        """
        total_results = len(results)
        scraped_results = [r for r in results if r.get("scraped", False)]
        quality_results = [r for r in scraped_results if r.get("quality_score", 0) >= self.content_quality_threshold]
        
        stats = {
            "total_results": total_results,
            "scraped_count": len(scraped_results),
            "quality_count": len(quality_results),
            "scrape_success_rate": len(scraped_results) / total_results if total_results > 0 else 0,
            "quality_success_rate": len(quality_results) / len(scraped_results) if scraped_results else 0,
            "average_quality_score": sum(r.get("quality_score", 0) for r in scraped_results) / len(scraped_results) if scraped_results else 0,
            "average_word_count": sum(r.get("word_count", 0) for r in scraped_results) / len(scraped_results) if scraped_results else 0,
            "total_words": sum(r.get("word_count", 0) for r in scraped_results),
            "average_scrape_time": sum(r.get("scrape_time", 0) for r in scraped_results) / len(scraped_results) if scraped_results else 0
        }
        
        return stats


class ContentQualityAnalyzer:
    """
    Advanced content quality analysis and filtering.
    """
    
    @staticmethod
    def analyze_content_relevance(content: str, query: str) -> float:
        """
        Analyze how relevant content is to a search query.
        
        Args:
            content: Content to analyze
            query: Original search query
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not content or not query:
            return 0.0
        
        content_lower = content.lower()
        query_terms = [term.strip() for term in query.lower().split()]
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        query_terms = [term for term in query_terms if term not in stop_words and len(term) > 2]
        
        if not query_terms:
            return 0.5  # Default relevance for empty query
        
        # Calculate term frequency
        total_matches = 0
        for term in query_terms:
            matches = len(re.findall(re.escape(term), content_lower))
            total_matches += matches
        
        # Normalize by content length and query terms
        content_words = len(content.split())
        if content_words == 0:
            return 0.0
        
        relevance = min(1.0, (total_matches / len(query_terms)) / max(1, content_words / 100))
        
        return relevance
    
    @staticmethod
    def extract_key_sentences(content: str, max_sentences: int = 3) -> List[str]:
        """
        Extract key sentences from content using simple heuristics.
        
        Args:
            content: Content to analyze
            max_sentences: Maximum number of sentences to extract
            
        Returns:
            List of key sentences
        """
        if not content:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return []
        
        # Score sentences based on length and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0.0
            
            # Length score (prefer medium-length sentences)
            length = len(sentence.split())
            if 10 <= length <= 30:
                score += 1.0
            elif 5 <= length <= 40:
                score += 0.5
            
            # Position score (prefer earlier sentences)
            position_score = 1.0 - (i / len(sentences))
            score += position_score * 0.5
            
            # Keyword density score
            word_variety = len(set(sentence.lower().split())) / length if length > 0 else 0
            score += word_variety * 0.3
            
            scored_sentences.append((score, sentence))
        
        # Sort by score and return top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        return [sentence for _, sentence in scored_sentences[:max_sentences]]


# Utility functions for backward compatibility
def scrape_url(url: str, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Scrape a single URL using Jina Reader API.
    
    Args:
        url: URL to scrape
        api_key: Deprecated parameter, kept for backward compatibility
        
    Returns:
        Scraped content dictionary or None
    """
    tool = WebSearchTool()
    return tool._scrape_url(url)


def get_web_urls(query: str, num_urls: int = 5) -> List[str]:
    """
    Search the web and return URLs only.
    
    Args:
        query: Search query
        num_urls: Number of URLs to return
        
    Returns:
        List of URLs
    """
    tool = WebSearchTool()
    results = tool._search_web(query, num_urls)
    return [result['url'] for result in results if result.get('url')]


def get_high_quality_content(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Get high-quality content for a search query with parallel processing.
    
    Args:
        query: Search query
        max_results: Maximum number of high-quality results to return
        
    Returns:
        List of high-quality search results
    """
    tool = WebSearchTool(content_quality_threshold=0.5)
    results = tool.search_and_scrape(
        query, 
        max_results=max_results * 2,  # Search more, filter to best
        scrape_content=True, 
        parallel_scraping=True,
        quality_filter=True
    )
    
    # Return only the highest quality results
    high_quality = [r for r in results if r.get('scraped', False) and r.get('quality_score', 0) >= 0.5]
    return high_quality[:max_results]