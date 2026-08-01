"""HTML extraction and cleaning service using BeautifulSoup and trafilatura."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse

import trafilatura
from bs4 import BeautifulSoup
from bs4.element import Comment
from trafilatura.metadata import extract_metadata

from app.schemas.website import WebsiteMetadata, WebsiteProcessingConfig


@dataclass
class ExtractedContent:
    """Extracted content from a web page."""

    text: str
    title: str | None
    metadata: WebsiteMetadata
    links: list[str]
    images: list[str]
    tables: list[dict[str, Any]]
    code_blocks: list[str]


class ExtractionError(Exception):
    """Raised when extraction fails."""

    def __init__(self, message: str, code: str = "EXTRACTION_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


def is_visible(element: Any) -> bool:
    """Check if a BeautifulSoup element is visible."""
    if element.parent.name in [
        "style",
        "script",
        "head",
        "title",
        "meta",
        "[document]",
    ]:
        return False
    if isinstance(element, Comment):
        return False
    return True


def clean_html(html: str, config: WebsiteProcessingConfig) -> str:
    """
    Clean HTML by removing unwanted elements.

    Args:
        html: Raw HTML content
        config: Processing configuration

    Returns:
        Cleaned HTML string
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, and other non-content elements
    for tag in soup(["script", "style", "noscript", "iframe", "embed", "object"]):
        tag.decompose()

    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove hidden elements
    for tag in soup.find_all(
        style=re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)
    ):
        tag.decompose()

    # Remove navigation, footer, header, aside if configured
    for tag in soup(["nav", "footer", "header", "aside"]):
        tag.decompose()

    # Remove elements with common ad/tracking classes
    ad_selectors = [
        "[class*='ad-']",
        "[class*='ads-']",
        "[class*='advert']",
        "[class*='banner']",
        "[class*='popup']",
        "[class*='modal']",
        "[class*='cookie']",
        "[class*='gdpr']",
        "[class*='tracking']",
        "[id*='google_ads']",
        "[id*='adsense']",
    ]
    for selector in ad_selectors:
        for tag in soup.select(selector):
            tag.decompose()

    return str(soup)


def extract_text_from_html(html: str, url: str) -> tuple[str, WebsiteMetadata]:
    """
    Extract clean text and metadata from HTML using trafilatura.

    This is the primary extraction method as it handles:
    - Boilerplate removal
    - Article extraction
    - Metadata extraction
    - Multiple language support

    Args:
        html: Raw HTML content
        url: Source URL

    Returns:
        Tuple of (clean_text, metadata)
    """
    # Use trafilatura for extraction
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        include_images=False,
        include_links=True,
        deduplicate=True,
        target_language=None,  # Auto-detect
    )

    if not extracted:
        # Fallback to basic BeautifulSoup extraction
        extracted = basic_text_extraction(html)

    # Extract metadata
    metadata = extract_metadata(html)
    site_metadata = (
        parse_metadata(metadata, url) if metadata else WebsiteMetadata(url=url)
    )

    return extracted or "", site_metadata


def basic_text_extraction(html: str) -> str:
    """Fallback text extraction using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(
        ["script", "style", "noscript", "iframe", "embed", "object", "head"]
    ):
        tag.decompose()

    # Get text
    text = soup.get_text(separator="\n", strip=True)

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n\n".join(lines)


def parse_metadata(trafilatura_metadata: Any, url: str) -> WebsiteMetadata:
    """Parse trafilatura metadata into WebsiteMetadata schema."""
    meta = WebsiteMetadata(url=url)

    if hasattr(trafilatura_metadata, "title"):
        meta.title = trafilatura_metadata.title
    if hasattr(trafilatura_metadata, "description"):
        meta.description = trafilatura_metadata.description
    if hasattr(trafilatura_metadata, "author"):
        meta.author = trafilatura_metadata.author
    if hasattr(trafilatura_metadata, "date"):
        try:
            meta.published_date = datetime.fromisoformat(
                trafilatura_metadata.date.replace("Z", "+00:00")
            )
        except Exception:
            pass
    if hasattr(trafilatura_metadata, "language"):
        meta.language = trafilatura_metadata.language
    if hasattr(trafilatura_metadata, "url"):
        meta.canonical_url = trafilatura_metadata.url
    if hasattr(trafilatura_metadata, "categories"):
        meta.custom_properties["categories"] = trafilatura_metadata.categories
    if hasattr(trafilatura_metadata, "tags"):
        meta.custom_properties["tags"] = trafilatura_metadata.tags

    return meta


def extract_detailed_content(
    html: str, url: str, config: WebsiteProcessingConfig
) -> ExtractedContent:
    """
    Extract detailed content including links, images, tables, and code blocks.

    Args:
        html: Raw HTML content
        url: Source URL
        config: Processing configuration

    Returns:
        ExtractedContent with all extracted elements
    """
    soup = BeautifulSoup(html, "html.parser")

    # Clean HTML first
    cleaned_html = clean_html(html, config)
    soup = BeautifulSoup(cleaned_html, "html.parser")

    # Extract title
    title = None
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Extract main content using trafilatura
    text = trafilatura.extract(
        cleaned_html,
        include_comments=False,
        include_tables=config.chunking.preserve_tables,
        include_images=False,
        include_links=True,
        deduplicate=True,
    ) or basic_text_extraction(cleaned_html)

    # Extract metadata
    metadata = extract_metadata(cleaned_html)
    site_metadata = (
        parse_metadata(metadata, url) if metadata else WebsiteMetadata(url=url)
    )

    # Extract links
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
            normalized = urljoin(url, href)
            parsed = urlparse(normalized)
            if parsed.scheme in ("http", "https"):
                links.append(normalized)
    links = list(set(links))

    # Extract images
    images = []
    for img in soup.find_all("img", src=True):
        src = img["src"]
        if src:
            normalized = urljoin(url, src)
            images.append(normalized)

    # Extract tables
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append({"rows": rows})

    # Extract code blocks
    code_blocks = []
    for code in soup.find_all(["code", "pre"]):
        code_text = code.get_text(strip=True)
        if code_text and len(code_text) > 10:  # Skip tiny snippets
            code_blocks.append(code_text)

    return ExtractedContent(
        text=text,
        title=title,
        metadata=site_metadata,
        links=links,
        images=images,
        tables=tables,
        code_blocks=code_blocks,
    )


def extract_og_metadata(soup: BeautifulSoup) -> dict[str, str]:
    """Extract Open Graph metadata."""
    og_data = {}
    for meta in soup.find_all("meta", property=re.compile(r"^og:")):
        prop = meta.get("property", "")
        content = meta.get("content", "")
        if prop and content:
            og_data[prop] = content
    return og_data


def extract_twitter_metadata(soup: BeautifulSoup) -> dict[str, str]:
    """Extract Twitter Card metadata."""
    twitter_data = {}
    for meta in soup.find_all("meta", attrs={"name": re.compile(r"^twitter:")}):
        name = meta.get("name", "")
        content = meta.get("content", "")
        if name and content:
            twitter_data[name] = content
    return twitter_data


def extract_schema_org(soup: BeautifulSoup) -> dict[str, Any]:
    """Extract Schema.org structured data (JSON-LD)."""
    schema_data = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            import json

            data = json.loads(script.string or "{}")
            # Merge or store by @type
            if isinstance(data, dict):
                schema_type = data.get("@type", "unknown")
                schema_data[schema_type] = data
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        schema_type = item.get("@type", "unknown")
                        schema_data[schema_type] = item
        except Exception:
            pass
    return schema_data


def enhance_metadata_with_html(
    html: str, url: str, metadata: WebsiteMetadata
) -> WebsiteMetadata:
    """Enhance metadata by parsing HTML directly for OG, Twitter, and Schema.org."""
    soup = BeautifulSoup(html, "html.parser")

    # Open Graph
    og = extract_og_metadata(soup)
    if og.get("og:title") and not metadata.og_title:
        metadata.og_title = og["og:title"]
    if og.get("og:description") and not metadata.og_description:
        metadata.og_description = og["og:description"]
    if og.get("og:image") and not metadata.og_image:
        metadata.og_image = og["og:image"]
    if og.get("og:type") and not metadata.og_type:
        metadata.og_type = og["og:type"]

    # Twitter
    twitter = extract_twitter_metadata(soup)
    if twitter.get("twitter:card") and not metadata.twitter_card:
        metadata.twitter_card = twitter["twitter:card"]
    if twitter.get("twitter:title") and not metadata.twitter_title:
        metadata.twitter_title = twitter["twitter:title"]
    if twitter.get("twitter:description") and not metadata.twitter_description:
        metadata.twitter_description = twitter["twitter:description"]
    if twitter.get("twitter:image") and not metadata.twitter_image:
        metadata.twitter_image = twitter["twitter:image"]

    # Schema.org
    schema = extract_schema_org(soup)
    if schema:
        metadata.schema_org = schema

    return metadata


def clean_text(text: str) -> str:
    """
    Clean extracted text by normalizing whitespace and removing artifacts.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive blank lines (more than 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing whitespace from lines
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Normalize spaces (but preserve intentional formatting)
    text = re.sub(r"[ \t]+", " ", text)

    return text


def count_tokens(text: str) -> int:
    """Estimate token count from text (rough approximation: 1 token ≈ 4 chars)."""
    return max(1, len(text) // 4)
