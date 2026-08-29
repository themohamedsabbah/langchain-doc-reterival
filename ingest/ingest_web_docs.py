import asyncio
import os
import ssl
import re
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from urllib.parse import urlparse
import certifi

from dotenv import load_dotenv
load_dotenv()

tavily_map = TavilyMap(
    max_depth= 5,
    max_breadth = 15,
    limit= 500
)

tavily_extract = TavilyExtract()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

async def main():
    print("Hello from langchain-doc-reterival!")
    CRAWL_URL: str = os.environ.get("CRAWL_URL")
    site_map = tavily_map.invoke(CRAWL_URL)
    urls = site_map.get('results', [])
    print(f"Successfuly mapped {len(urls)}")

    pages = await extract_doc(urls= urls)

    output_dir = "./crawled_pages"
    os.makedirs(output_dir, exist_ok=True)

    for idx, page in enumerate(pages, start=1):
        content = page.get("raw_content") or ""

        raw_name = page.get('title') or page.get('url') or f"page_{idx}"
        filename = sanitize_filename(str(raw_name))
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[{idx}/{len(pages)}] Saved: {filepath}")

async def extract_doc(urls):
    all_extracted_docs = []
    BATCH_SIZE = 20
    for i in range(0, len(urls), BATCH_SIZE):
        batch = urls[i: i+BATCH_SIZE]
        batch_number = i // BATCH_SIZE + 1
        print(f"[Batch {batch_number}] Extracting {len(batch)} URLs...")
        try:
            result = await tavily_extract.ainvoke({
                "urls": batch
            })
            if "results" in result:
                docs = result["results"]
                all_extracted_docs.extend(docs)
            else:
                print(f"[Batch {batch_number}] Failed: {result}")
        except Exception as e:
            print(f"[Batch {batch_number}] - Exception: {e}")
    return all_extracted_docs

def sanitize_filename(name: str) -> str:
    """Converts a URL into a safe filesystem filename."""
    if not name:
        return "page.txt"
    if name.startswith("http://") or name.startswith("https://"):
        parsed = urlparse(name)
        name = parsed.netloc + parsed.path

    clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_")
    clean_name = re.sub(r"_+", "_", clean_name)[:200]
    
    return f"{clean_name or 'page'}.txt"

if __name__ == "__main__":
    asyncio.run(main())
