"""Website content chunking service."""

from __future__ import annotations

import re

from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text

from app.schemas.website import (WebsiteChunk, WebsiteChunkingConfig,
                                 WebsiteProcessingConfig)


class ChunkingError(Exception):
    """Raised when chunking fails."""

    def __init__(self, message: str, code: str = "CHUNKING_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 0.75 words)."""
    words = len(text.split())
    return max(1, int(words * 1.3))


def split_code_blocks(text: str) -> list[tuple[str, bool]]:
    """
    Split text preserving code blocks.

    Returns:
        List of (text_segment, is_code_block) tuples
    """
    # Pattern to match code blocks (markdown style)
    code_pattern = re.compile(r"(```[\s\S]*?```|`[^`\n]+`)")

    parts = []
    last_end = 0

    for match in code_pattern.finditer(text):
        start, end = match.span()

        # Add text before code block
        if start > last_end:
            parts.append((text[last_end:start], False))

        # Add code block
        parts.append((match.group(0), True))
        last_end = end

    # Add remaining text
    if last_end < len(text):
        parts.append((text[last_end:], False))

    return parts


def chunk_text_fixed(
    text: str,
    config: WebsiteChunkingConfig,
    source_url: str,
) -> list[WebsiteChunk]:
    """
    Fixed-size chunking with overlap.

    Args:
        text: Text to chunk
        config: Chunking configuration
        source_url: Source URL for chunks

    Returns:
        List of WebsiteChunk objects
    """
    chunks = []
    words = text.split()

    # Convert token sizes to approximate word sizes
    chunk_size_words = int(config.chunk_size * 0.75)
    overlap_words = int(config.chunk_overlap * 0.75)

    if chunk_size_words <= overlap_words:
        chunk_size_words = overlap_words + 1

    for i in range(0, len(words), chunk_size_words - overlap_words):
        chunk_words = words[i : i + chunk_size_words]
        if not chunk_words:
            break

        chunk_text = " ".join(chunk_words)
        chunks.append(
            WebsiteChunk(
                index=len(chunks),
                content=chunk_text,
                token_count=estimate_tokens(chunk_text),
                source_url=source_url,
                metadata={
                    "strategy": "fixed",
                    "word_start": i,
                    "word_end": i + len(chunk_words),
                },
            )
        )

        # If we've consumed all words, break
        if i + chunk_size_words >= len(words):
            break

    return chunks


def chunk_text_recursive(
    text: str,
    config: WebsiteChunkingConfig,
    source_url: str,
) -> list[WebsiteChunk]:
    """
    Recursive character text splitting (similar to LangChain's RecursiveCharacterTextSplitter).

    Args:
        text: Text to chunk
        config: Chunking configuration
        source_url: Source URL for chunks

    Returns:
        List of WebsiteChunk objects
    """
    # Use unstructured's basic chunking
    elements = partition_text(text=text)
    chunked_elements = chunk_elements(
        elements,
        max_characters=config.chunk_size * 4,
        overlap=config.chunk_overlap * 4,
    )

    chunks = []
    for i, element in enumerate(chunked_elements):
        chunk_text = str(element)
        if chunk_text.strip():
            chunks.append(
                WebsiteChunk(
                    index=i,
                    content=chunk_text,
                    token_count=estimate_tokens(chunk_text),
                    source_url=source_url,
                    metadata={"strategy": "recursive"},
                )
            )

    return chunks


def chunk_text_semantic(
    text: str,
    config: WebsiteChunkingConfig,
    source_url: str,
) -> list[WebsiteChunk]:
    """
    Semantic chunking based on document structure (titles, headings).

    Args:
        text: Text to chunk
        config: Chunking configuration
        source_url: Source URL for chunks

    Returns:
        List of WebsiteChunk objects
    """
    elements = partition_text(text=text)
    chunked_elements = chunk_by_title(
        elements,
        max_characters=config.chunk_size * 4,
        overlap=config.chunk_overlap * 4,
    )

    chunks = []
    for i, element in enumerate(chunked_elements):
        chunk_text = str(element)
        if chunk_text.strip():
            chunks.append(
                WebsiteChunk(
                    index=i,
                    content=chunk_text,
                    token_count=estimate_tokens(chunk_text),
                    source_url=source_url,
                    metadata={"strategy": "semantic"},
                )
            )

    return chunks


def chunk_html_semantic(
    html: str,
    config: WebsiteChunkingConfig,
    source_url: str,
) -> list[WebsiteChunk]:
    """
    Semantic chunking of HTML content preserving structure.

    Args:
        html: HTML content to chunk
        config: Chunking configuration
        source_url: Source URL for chunks

    Returns:
        List of WebsiteChunk objects
    """
    elements = partition_html(text=html)
    chunked_elements = chunk_by_title(
        elements,
        max_characters=config.chunk_size * 4,
        overlap=config.chunk_overlap * 4,
    )

    chunks = []
    for i, element in enumerate(chunked_elements):
        chunk_text = str(element)
        if chunk_text.strip():
            # Extract metadata from element
            elem_metadata = {"strategy": "semantic_html"}
            if hasattr(element, "metadata") and element.metadata:
                if hasattr(element.metadata, "category"):
                    elem_metadata["category"] = element.metadata.category
                if hasattr(element.metadata, "section"):
                    elem_metadata["section"] = element.metadata.section

            chunks.append(
                WebsiteChunk(
                    index=i,
                    content=chunk_text,
                    token_count=estimate_tokens(chunk_text),
                    source_url=source_url,
                    metadata=elem_metadata,
                )
            )

    return chunks


def chunk_with_code_preservation(
    text: str,
    config: WebsiteChunkingConfig,
    source_url: str,
) -> list[WebsiteChunk]:
    """
    Chunk text while preserving code blocks intact.

    Args:
        text: Text to chunk
        config: Chunking configuration
        source_url: Source URL for chunks

    Returns:
        List of WebsiteChunk objects
    """
    if not config.preserve_code_blocks:
        return chunk_text_recursive(text, config, source_url)

    parts = split_code_blocks(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0
    chunk_index = 0

    for part_text, is_code in parts:
        part_tokens = estimate_tokens(part_text)

        # If code block is too large, we need to split it
        if is_code and part_tokens > config.chunk_size:
            # Add current chunk if it has content
            if current_chunk.strip():
                chunks.append(
                    WebsiteChunk(
                        index=chunk_index,
                        content=current_chunk.strip(),
                        token_count=current_tokens,
                        source_url=source_url,
                        metadata={"strategy": "recursive", "has_code": True},
                    )
                )
                chunk_index += 1
                current_chunk = ""
                current_tokens = 0

            # Split large code block using fixed chunking
            code_chunks = chunk_text_fixed(
                part_text,
                WebsiteChunkingConfig(
                    chunk_size=config.chunk_size,
                    chunk_overlap=0,
                    strategy="fixed",
                ),
                source_url,
            )
            for cc in code_chunks:
                cc.index = chunk_index
                cc.metadata["strategy"] = "code_fixed"
                cc.metadata["is_code"] = True
                chunks.append(cc)
                chunk_index += 1
            continue

        # Check if adding this part would exceed chunk size
        if current_tokens + part_tokens > config.chunk_size and current_chunk.strip():
            chunks.append(
                WebsiteChunk(
                    index=chunk_index,
                    content=current_chunk.strip(),
                    token_count=current_tokens,
                    source_url=source_url,
                    metadata={
                        "strategy": "recursive",
                        "has_code": any(
                            p[1] for p in parts[: parts.index((part_text, is_code)) + 1]
                        ),
                    },
                )
            )
            chunk_index += 1
            current_chunk = part_text
            current_tokens = part_tokens
        else:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += part_text
            current_tokens += part_tokens

    # Add remaining chunk
    if current_chunk.strip():
        chunks.append(
            WebsiteChunk(
                index=chunk_index,
                content=current_chunk.strip(),
                token_count=current_tokens,
                source_url=source_url,
                metadata={
                    "strategy": "recursive",
                    "has_code": any(p[1] for p in parts),
                },
            )
        )

    return chunks


def chunk_website_content(
    pages_content: list[tuple[str, str]],  # List of (url, text_content)
    config: WebsiteProcessingConfig,
) -> list[WebsiteChunk]:
    """
    Chunk multiple pages of website content.

    Args:
        pages_content: List of (url, text_content) tuples
        config: Processing configuration

    Returns:
        List of WebsiteChunk objects
    """
    all_chunks = []
    chunk_index = 0

    for url, text in pages_content:
        if not text or len(text.strip()) < config.min_content_length:
            continue

        # Choose chunking strategy
        if config.chunking.strategy == "fixed":
            chunks = chunk_text_fixed(text, config.chunking, url)
        elif config.chunking.strategy == "semantic":
            chunks = chunk_text_semantic(text, config.chunking, url)
        else:  # recursive
            if config.chunking.preserve_code_blocks:
                chunks = chunk_with_code_preservation(text, config.chunking, url)
            else:
                chunks = chunk_text_recursive(text, config.chunking, url)

        # Update chunk indices to be globally unique
        for chunk in chunks:
            chunk.index = chunk_index
            chunk_index += 1
            all_chunks.append(chunk)

    return all_chunks


def chunk_single_page(
    text: str,
    url: str,
    config: WebsiteProcessingConfig,
) -> list[WebsiteChunk]:
    """
    Chunk a single page's content.

    Args:
        text: Page text content
        url: Source URL
        config: Processing configuration

    Returns:
        List of WebsiteChunk objects
    """
    if not text or len(text.strip()) < config.min_content_length:
        return []

    if config.chunking.strategy == "fixed":
        return chunk_text_fixed(text, config.chunking, url)
    elif config.chunking.strategy == "semantic":
        return chunk_text_semantic(text, config.chunking, url)
    else:  # recursive
        if config.chunking.preserve_code_blocks:
            return chunk_with_code_preservation(text, config.chunking, url)
        return chunk_text_recursive(text, config.chunking, url)
