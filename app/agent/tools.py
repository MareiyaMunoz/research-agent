import os

import httpx
import trafilatura
from dotenv import load_dotenv
from tavily import TavilyClient

from app.agent import cache
from app.log import get_logger

logger = get_logger(__name__)

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MAX_RESULTS = 5          # search results per query
MAX_PAGE_CHARS = 6000    # keeps one page from flooding the history
USER_AGENT = "Mozilla/5.0 (research-agent learning project)"


# --- The actual functions (your code runs these) ---

def search(query: str) -> dict:
    query = query.strip()
    if not query:
        return {"error": "Empty search query"}

    cached = cache.get("search", query)
    if cached is not None:
        logger.debug("search cache hit: %s", query)
        return cached

    try:
        response = tavily.search(query=query, max_results=MAX_RESULTS)
    except Exception as e:
        return {"error": f"Search failed: {type(e).__name__}: {e}"}
    output = {
        "query": query,
        "results": [
            {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("content")}
            for r in response.get("results", [])
        ],
    }
    cache.put("search", query, output)
    return output


def read_page(url: str) -> dict:
    if not url.startswith(("http://", "https://")):
        return {"error": "URL must start with http:// or https://"}

    cached = cache.get("page", url)
    if cached is not None:
        logger.debug("page cache hit: %s", url)
        return cached

    try:
        resp = httpx.get(
            url,
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return {"error": f"Could not fetch page: {type(e).__name__}: {e}"}

    if "html" not in resp.headers.get("content-type", ""):
        return {"error": "This URL is not a normal web page (maybe a PDF or file)"}

    text = trafilatura.extract(resp.text)
    if not text:
        return {"error": "Could not extract readable text from this page"}

    output = {
        "url": url,
        "text": text[:MAX_PAGE_CHARS],
        "truncated": len(text) > MAX_PAGE_CHARS,
        "total_chars": len(text),
    }
    cache.put("page", url, output)
    return output


# --- Registry: tool name -> function ---
TOOL_FUNCTIONS = {
    "search": search,
    "read_page": read_page,
}

# --- Descriptions the model reads ---
TOOL_DECLARATIONS = [
    {
        "name": "search",
        "description": (
            "Search the web. Returns a list of results with title, URL, and a "
            "short snippet. Use read_page to read a result in full."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_page",
        "description": (
            "Download a web page and return its readable text (may be truncated). "
            "Use a URL from search results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full page URL starting with http"}
            },
            "required": ["url"],
        },
    },
]