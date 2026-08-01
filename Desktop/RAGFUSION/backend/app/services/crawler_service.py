"""Website crawler service using crawl4ai."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import aiohttp
from crawl4ai import (AsyncWebCrawler, BrowserConfig, CacheMode,
                      CrawlerRunConfig)

from app.schemas.website import (CrawlDepth, CrawledPage, WebsiteCrawlRequest,
                                 WebsiteProcessingConfig)


@dataclass
class CrawlResult:
    """Result of a crawl operation."""

    pages: list[CrawledPage]
    base_url: str
    errors: list[str]


class CrawlerError(Exception):
    """Raised when crawling fails."""

    def __init__(self, message: str, code: str = "CRAWL_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class CrawlerTimeoutError(CrawlerError):
    """Raised when crawling times out."""

    def __init__(self, message: str):
        super().__init__(message, code="CRAWL_TIMEOUT")


class CrawlerValidationError(CrawlerError):
    """Raised when URL validation fails."""

    def __init__(self, message: str):
        super().__init__(message, code="CRAWL_VALIDATION_ERROR")


def validate_url(url: str) -> None:
    """
    Validate a URL for crawling.

    Args:
        url: URL to validate

    Raises:
        CrawlerValidationError: If URL is invalid or not allowed
    """
    parsed = urlparse(url)

    if not parsed.scheme or not parsed.netloc:
        raise CrawlerValidationError("Invalid URL format")

    if parsed.scheme not in ("http", "https"):
        raise CrawlerValidationError("Only HTTP and HTTPS URLs are allowed")

    # Block local/private addresses
    blocked_patterns = [
        r"^localhost$",
        r"^127\.\d+\.\d+\.\d+$",
        r"^10\.\d+\.\d+\.\d+$",
        r"^172\.(1[6-9]|2[0-9]|3[0-1])\.\d+\.\d+$",
        r"^192\.168\.\d+\.\d+$",
        r"^169\.254\.\d+\.\d+$",
        r"^::1$",
        r"^fe80::",
    ]

    hostname = parsed.hostname or ""
    for pattern in blocked_patterns:
        if re.match(pattern, hostname):
            raise CrawlerValidationError(f"Access to {hostname} is not allowed")


def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs belong to the same domain."""
    return urlparse(url1).netloc == urlparse(url2).netloc


def normalize_url(url: str, base_url: str) -> str:
    """Normalize a URL relative to a base URL."""
    return urljoin(base_url, url)


async def fetch_page(
    session: aiohttp.ClientSession,
    url: str,
    timeout_seconds: int,
    follow_redirects: bool,
) -> tuple[int, str | None, str | None, list[str]]:
    """
    Fetch a single page and extract links.

    Returns:
        Tuple of (status_code, content_type, html_content, links)
    """
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with session.get(
            url,
            timeout=timeout,
            allow_redirects=follow_redirects,
            headers={"User-Agent": "DOCPRO-Bot/1.0 (+https://docpro.example.com/bot)"},
        ) as response:
            content_type = response.headers.get("Content-Type", "")
            html = await response.text()

            # Extract links from HTML
            links = extract_links(html, url)

            return response.status, content_type, html, links

    except asyncio.TimeoutError:
        raise CrawlerTimeoutError(f"Request timeout for {url}")
    except aiohttp.ClientError as e:
        raise CrawlerError(f"Request failed for {url}: {e}")
    except Exception as e:
        raise CrawlerError(f"Unexpected error fetching {url}: {e}")


def extract_links(html: str, base_url: str) -> list[str]:
    """Extract all links from HTML content."""
    links = []
    # Simple regex for href attributes
    href_pattern = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)

    for match in href_pattern.finditer(html):
        href = match.group(1).strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        normalized = normalize_url(href, base_url)
        parsed = urlparse(normalized)

        # Only keep HTTP/HTTPS links
        if parsed.scheme in ("http", "https"):
            links.append(normalized)

    return list(set(links))  # Deduplicate


async def crawl_with_crawl4ai(
    urls: list[str],
    config: WebsiteProcessingConfig,
    timeout_seconds: int,
) -> list[CrawledPage]:
    """
    Crawl URLs using crawl4ai for better extraction.

    This uses the AsyncWebCrawler which handles JavaScript rendering,
    dynamic content, and provides clean markdown output.
    """
    pages = []

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
    )

    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=config.min_content_length,
        excluded_tags=[
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "noscript",
        ],
        remove_overlay_elements=True,
        process_iframes=False,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Process URLs in batches to avoid overwhelming the crawler
        batch_size = 5
        for i in range(0, len(urls), batch_size):
            batch = urls[i : i + batch_size]

            tasks = []
            for url in batch:
                task = crawler.arun(
                    url=url,
                    config=crawler_config,
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for url, result in zip(batch, results):
                if isinstance(result, Exception):
                    pages.append(
                        CrawledPage(
                            url=url,
                            status_code=0,
                            text_content="",
                            error=str(result),
                        )
                    )
                    continue

                # crawl4ai returns a CrawlResult object
                pages.append(
                    CrawledPage(
                        url=result.url,
                        status_code=result.status_code or 200,
                        content_type="text/html",
                        text_content=result.markdown or result.cleaned_html or "",
                        word_count=len((result.markdown or "").split()),
                        character_count=len(result.markdown or ""),
                        links=result.links.get("internal", [])
                        + result.links.get("external", []),
                        metadata=None,  # Will be filled by extraction service
                    )
                )

    return pages


async def crawl_website(
    request: WebsiteCrawlRequest, config: WebsiteProcessingConfig
) -> CrawlResult:
    """
    Crawl a website starting from the given URL.

    Uses a combination of crawl4ai (for JavaScript-heavy sites) and
    aiohttp (for simple static sites) for optimal performance.
    """
    validate_url(str(request.url))

    base_url = str(request.url)
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc

    visited: set[str] = set()
    to_visit: list[str] = [base_url]
    all_pages: list[CrawledPage] = []
    errors: list[str] = []

    # Determine max pages based on crawl depth
    max_pages = request.max_pages
    if request.crawl_depth == CrawlDepth.single:
        max_pages = 1
    elif request.crawl_depth == CrawlDepth.shallow:
        max_pages = min(max_pages, 50)

    # Use aiohttp for initial discovery (faster for static sites)
    connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
    timeout = aiohttp.ClientTimeout(total=request.timeout_seconds)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while to_visit and len(visited) < max_pages:
            current_url = to_visit.pop(0)

            if current_url in visited:
                continue

            # Check domain restriction
            if not is_same_domain(current_url, base_url):
                continue

            visited.add(current_url)

            try:
                status_code, content_type, html, links = await fetch_page(
                    session,
                    current_url,
                    request.timeout_seconds,
                    request.follow_redirects,
                )

                if status_code >= 400:
                    errors.append(f"{current_url}: HTTP {status_code}")
                    all_pages.append(
                        CrawledPage(
                            url=current_url,
                            status_code=status_code,
                            content_type=content_type,
                            text_content="",
                            error=f"HTTP {status_code}",
                        )
                    )
                    continue

                # Only process HTML content
                if content_type and "text/html" not in content_type.lower():
                    all_pages.append(
                        CrawledPage(
                            url=current_url,
                            status_code=status_code,
                            content_type=content_type,
                            text_content="",
                            error=f"Non-HTML content type: {content_type}",
                        )
                    )
                    continue

                # Extract text content (basic)
                from app.services.extraction_service import \
                    extract_text_from_html

                text_content, metadata = extract_text_from_html(html, current_url)

                page = CrawledPage(
                    url=current_url,
                    status_code=status_code,
                    content_type=content_type,
                    text_content=text_content,
                    metadata=metadata,
                    word_count=len(text_content.split()),
                    character_count=len(text_content),
                    links=links,
                )
                all_pages.append(page)

                # Add new links to crawl queue
                if request.crawl_depth != CrawlDepth.single:
                    for link in links:
                        if link not in visited and is_same_domain(link, base_url):
                            to_visit.append(link)

            except CrawlerTimeoutError:
                errors.append(f"{current_url}: Timeout")
                all_pages.append(
                    CrawledPage(
                        url=current_url,
                        status_code=0,
                        text_content="",
                        error="Request timeout",
                    )
                )
            except CrawlerError as e:
                errors.append(f"{current_url}: {e.message}")
                all_pages.append(
                    CrawledPage(
                        url=current_url,
                        status_code=0,
                        text_content="",
                        error=e.message,
                    )
                )
            except Exception as e:
                errors.append(f"{current_url}: Unexpected error - {e}")
                all_pages.append(
                    CrawledPage(
                        url=current_url,
                        status_code=0,
                        text_content="",
                        error=str(e),
                    )
                )

    # For deeper crawls, use crawl4ai for better JavaScript support
    if request.crawl_depth == CrawlDepth.deep and len(all_pages) < max_pages:
        # Get URLs that haven't been fully processed
        remaining_urls = [
            p.url
            for p in all_pages
            if p.error or p.word_count < config.min_content_length
        ]
        if remaining_urls:
            crawl4ai_pages = await crawl_with_crawl4ai(
                remaining_urls[:10], config, request.timeout_seconds
            )
            # Merge results (replace failed pages with crawl4ai results)
            url_to_crawl4ai = {p.url: p for p in crawl4ai_pages}
            for i, page in enumerate(all_pages):
                if page.url in url_to_crawl4ai and (
                    page.error or page.word_count < config.min_content_length
                ):
                    crawl4ai_page = url_to_crawl4ai[page.url]
                    if (
                        not crawl4ai_page.error
                        and crawl4ai_page.word_count >= config.min_content_length
                    ):
                        all_pages[i] = crawl4ai_page

    return CrawlResult(
        pages=all_pages,
        base_url=base_url,
        errors=errors,
    )


async def crawl_single_page(url: str, timeout_seconds: int = 30) -> CrawledPage:
    """
    Crawl a single page using aiohttp.

    Useful for quick validation or single-page ingestion.
    """
    validate_url(url)

    connector = aiohttp.TCPConnector(limit=1)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        try:
            status_code, content_type, html, links = await fetch_page(
                session, url, timeout_seconds, True
            )

            if status_code >= 400:
                return CrawledPage(
                    url=url,
                    status_code=status_code,
                    content_type=content_type,
                    text_content="",
                    error=f"HTTP {status_code}",
                )

            if content_type and "text/html" not in content_type.lower():
                return CrawledPage(
                    url=url,
                    status_code=status_code,
                    content_type=content_type,
                    text_content="",
                    error=f"Non-HTML content type: {content_type}",
                )

            from app.services.extraction_service import extract_text_from_html

            text_content, metadata = extract_text_from_html(html, url)

            return CrawledPage(
                url=url,
                status_code=status_code,
                content_type=content_type,
                text_content=text_content,
                metadata=metadata,
                word_count=len(text_content.split()),
                character_count=len(text_content),
                links=links,
            )

        except CrawlerTimeoutError:
            return CrawledPage(
                url=url,
                status_code=0,
                text_content="",
                error="Request timeout",
            )
        except CrawlerError as e:
            return CrawledPage(
                url=url,
                status_code=0,
                text_content="",
                error=e.message,
            )
        except Exception as e:
            return CrawledPage(
                url=url,
                status_code=0,
                text_content="",
                error=str(e),
            )
