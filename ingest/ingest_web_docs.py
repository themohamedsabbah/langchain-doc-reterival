import asyncio
import os
import re
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()
tavily_crawl = TavilyCrawl()

async def main():
    print("Hello from langchain-doc-reterival!")
    CRAWL_URL: str = os.environ.get("CRAWL_URL")
    CRAWL_MAX_DEPTH: int = os.environ.get("CRAWL_MAX_DEPTH")
    res = tavily_crawl.invoke(
        {
            "url": CRAWL_URL,
            "max_depth": int(CRAWL_MAX_DEPTH),
            "extract_depth": "advanced",
        }
    )
    print(res)
    # Set up output folder
    output_dir = "./crawled_pages"
    os.makedirs(output_dir, exist_ok=True)

    # Handle dictionary vs list response structures
    pages = res if isinstance(res, list) else res.get("results", [])

    print(f"Retrieved {len(pages)} page(s). Saving files to '{output_dir}'...\n")

    for idx, page in enumerate(pages, start=1):
        url = page.get("url", f"page_{idx}")
        # Extract raw HTML/markdown or text content
        content = page.get("raw_content") or page.get("content") or str(page)

        filename = sanitize_filename(url)
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[{idx}/{len(pages)}] Saved: {filepath}")

    print("\nExtraction complete.")

def sanitize_filename(url: str) -> str:
    """Converts a URL into a safe filesystem filename."""
    parsed = urlparse(url)
    path = parsed.netloc + parsed.path
    # Replace invalid filename characters with underscores
    clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", path).strip("_")
    return f"{clean_name or 'page'}.txt"

if __name__ == "__main__":
    asyncio.run(main())
