"""Website ingestion orchestration service."""

from __future__ import annotations

import time
import uuid
from typing import Any

from app.db.session import get_session_factory
from app.models.entities import Embedding, SourceStatus, Website
from app.schemas.website import (WebsiteIngestRequest, WebsiteProcessingConfig,
                                 WebsiteProcessingResult)
from app.services.chunking_service import chunk_website_content
from app.services.crawler_service import CrawlerError, crawl_website
from app.services.embedding_helper import generate_embeddings


class IngestionError(Exception):
    """Raised when ingestion fails."""

    def __init__(self, message: str, code: str = "INGESTION_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


async def store_website_embeddings(
    website_id: uuid.UUID,
    chunks: list[Any],
    embeddings: list[list[float]],
    model_name: str,
) -> None:
    """
    Store website chunks and embeddings in database.

    Args:
        website_id: Website UUID
        chunks: List of WebsiteChunk objects
        embeddings: List of embedding vectors
        model_name: Embedding model name
    """
    async with get_session_factory()() as session:
        # Store each chunk with its embedding
        for chunk, embedding in zip(chunks, embeddings):
            emb = Embedding(
                website_id=website_id,
                chunk_index=chunk.index,
                content=chunk.content,
                vector=embedding,
                model_name=model_name,
                token_count=chunk.token_count,
                metadata_=chunk.metadata,
            )
            session.add(emb)

        # Update website status
        website = await session.get(Website, website_id)
        if website:
            website.status = SourceStatus.ready
            website.metadata_["total_chunks"] = len(chunks)
            website.metadata_["total_tokens"] = sum(c.token_count or 0 for c in chunks)

        await session.commit()


async def ingest_website(
    request: WebsiteIngestRequest,
    user_id: uuid.UUID,
    config: WebsiteProcessingConfig | None = None,
) -> WebsiteProcessingResult:
    """
    Complete website ingestion pipeline:
    validate -> crawl -> extract -> chunk -> embed -> store

    Args:
        request: Website ingestion request
        user_id: User UUID
        config: Processing configuration

    Returns:
        WebsiteProcessingResult with all processing metrics

    Raises:
        IngestionError: If any step fails
    """
    start_time = time.time()
    config = config or WebsiteProcessingConfig()

    # Create website record
    website_id = uuid.uuid4()
    async with get_session_factory()() as session:
        website = Website(
            id=website_id,
            user_id=user_id,
            url=str(request.url),
            status=SourceStatus.pending,
            metadata_={
                "crawl_depth": request.crawl_depth.value,
                "max_pages": request.max_pages,
            },
        )
        session.add(website)
        await session.commit()

    try:
        # Step 1: Update status to crawling
        async with get_session_factory()() as session:
            website = await session.get(Website, website_id)
            if website:
                website.status = SourceStatus.processing
            await session.commit()

        # Step 2: Crawl website
        from app.schemas.website import WebsiteCrawlRequest

        crawl_request = WebsiteCrawlRequest(
            url=request.url,
            crawl_depth=request.crawl_depth,
            max_pages=request.max_pages,
            follow_redirects=request.follow_redirects,
            timeout_seconds=request.timeout_seconds,
        )

        crawl_result = await crawl_website(crawl_request, config)

        if not crawl_result.pages:
            raise IngestionError(
                "No pages crawled successfully", code="NO_PAGES_CRAWLED"
            )

        # Filter out pages with errors or empty content
        valid_pages = [
            p
            for p in crawl_result.pages
            if not p.error
            and p.text_content
            and len(p.text_content.strip()) >= config.min_content_length
        ]

        if not valid_pages:
            raise IngestionError(
                "No valid pages with sufficient content", code="NO_VALID_CONTENT"
            )

        # Step 3: Extract and clean content (already done during crawling)
        # Update website with title from first page
        async with get_session_factory()() as session:
            website = await session.get(Website, website_id)
            if website and valid_pages:
                first_page = valid_pages[0]
                if first_page.metadata and first_page.metadata.title:
                    website.title = first_page.metadata.title
                website.metadata_["total_pages"] = len(valid_pages)
                website.metadata_["total_pages_crawled"] = len(crawl_result.pages)
                website.metadata_["failed_pages"] = len(crawl_result.pages) - len(
                    valid_pages
                )
            await session.commit()

        # Step 4: Chunk content
        pages_content = [(p.url, p.text_content) for p in valid_pages]
        chunks = chunk_website_content(pages_content, config)

        if not chunks:
            raise IngestionError(
                "No chunks generated from content", code="NO_CHUNKS_GENERATED"
            )

        # Step 5: Generate embeddings
        embeddings = await generate_embeddings(
            chunks,
            config.embedding.model_name,
            config.embedding.dimensions,
        )

        # Step 6: Store embeddings
        await store_website_embeddings(
            website_id,
            chunks,
            embeddings,
            config.embedding.model_name,
        )

        # Calculate metrics
        processing_time_ms = int((time.time() - start_time) * 1000)
        total_tokens = sum(c.token_count or 0 for c in chunks)

        # Update last_crawled_at
        from datetime import datetime

        async with get_session_factory()() as session:
            website = await session.get(Website, website_id)
            if website:
                website.last_crawled_at = datetime.utcnow()
            await session.commit()

        return WebsiteProcessingResult(
            website_id=website_id,
            url=str(request.url),
            total_pages=len(valid_pages),
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            processing_time_ms=processing_time_ms,
            pages=valid_pages,
            chunks=chunks,
            metadata={
                "crawl_depth": request.crawl_depth.value,
                "max_pages": request.max_pages,
                "errors": crawl_result.errors,
            },
        )

    except CrawlerError as e:
        # Update status to failed
        async with get_session_factory()() as session:
            website = await session.get(Website, website_id)
            if website:
                website.status = SourceStatus.failed
                website.metadata_["error"] = e.message
                website.metadata_["error_code"] = e.code
            await session.commit()
        raise IngestionError(f"Crawling failed: {e.message}", code=e.code) from e

    except Exception as e:
        # Update status to failed
        async with get_session_factory()() as session:
            website = await session.get(Website, website_id)
            if website:
                website.status = SourceStatus.failed
                website.metadata_["error"] = str(e)
            await session.commit()
        raise IngestionError(f"Ingestion failed: {e}", code="INGESTION_FAILED") from e


async def get_website_status(website_id: uuid.UUID) -> dict[str, Any]:
    """
    Get website processing status.

    Args:
        website_id: Website UUID

    Returns:
        Dictionary with status information

    Raises:
        IngestionError: If website not found
    """
    async with get_session_factory()() as session:
        website = await session.get(Website, website_id)
        if not website:
            raise IngestionError("Website not found", code="WEBSITE_NOT_FOUND")

        return {
            "id": str(website.id),
            "url": website.url,
            "title": website.title,
            "status": website.status.value,
            "last_crawled_at": (
                website.last_crawled_at.isoformat() if website.last_crawled_at else None
            ),
            "metadata": website.metadata_,
            "created_at": website.created_at.isoformat(),
            "updated_at": website.updated_at.isoformat(),
        }


async def delete_website(website_id: uuid.UUID, user_id: uuid.UUID) -> None:
    """
    Delete website and all its embeddings.

    Args:
        website_id: Website UUID
        user_id: User UUID (for authorization)

    Raises:
        IngestionError: If website not found or unauthorized
    """

    session_factory = get_session_factory()

    async with session_factory() as session:
        website = await session.get(Website, website_id)

        if not website:
            raise IngestionError(
                "Website not found",
                code="WEBSITE_NOT_FOUND",
            )

        if website.user_id != user_id:
            raise IngestionError(
                "Unauthorized to delete this website",
                code="UNAUTHORIZED",
            )

        # Delete website (embeddings will be cascade deleted)
        await session.delete(website)
        await session.commit()
