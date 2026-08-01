"""Website ingestion API routes."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user_id as get_current_user
from app.db.session import get_db_session
from app.models.entities import SourceStatus, User, Website
from app.schemas.website import (CrawlDepth, WebsiteDeleteResponse,
                                 WebsiteIngestRequest, WebsiteIngestResponse,
                                 WebsiteListResponse, WebsiteProcessingConfig,
                                 WebsiteResponse, WebsiteStatus)
from app.services.crawler_service import CrawlerError, crawl_website
from app.services.ingestion_service import (IngestionError, delete_website,
                                            get_website_status, ingest_website)

router = APIRouter(prefix="/website", tags=["website"])


@router.post(
    "/ingest",
    response_model=WebsiteIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a website",
    description="""
    Ingest a website into the RAG system.

    The website will be processed through the pipeline:
    validation -> crawling -> extraction -> chunking -> embedding -> storage

    Crawl depths:
    - single: Only crawl the provided URL
    - shallow: Crawl up to 50 pages from the same domain
    - deep: Crawl up to max_pages from the same domain

    Processing is asynchronous. Use GET /website/status/{id} to check progress.
    """,
)
async def ingest_website_route(
    request: WebsiteIngestRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> WebsiteIngestResponse:
    """Ingest a website for RAG."""
    # Check if website already exists for this user
    query = select(Website).where(
        Website.user_id == current_user.id,
        Website.url == str(request.url),
    )
    result = await db.execute(query)
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Website {request.url} already exists. Use DELETE to remove it first.",
        )

    # Create processing config
    config = WebsiteProcessingConfig(
        extract_metadata=request.extract_metadata,
        clean_content=request.clean_content,
    )

    try:
        # Start ingestion (runs synchronously for now)
        # TODO: Move to background task queue for true async processing
        result = await ingest_website(request, current_user.id, config)

        return WebsiteIngestResponse(
            website_id=result.website_id,
            url=result.url,
            status=WebsiteStatus.ready,
            message=f"Website ingestion completed. Processed {result.total_pages} pages into {result.total_chunks} chunks.",
        )

    except CrawlerError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Crawling failed: {e.message}",
        )
    except IngestionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {e.message}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.post(
    "/crawl",
    response_model=dict,
    summary="Crawl a website (discovery only)",
    description="""
    Crawl a website to discover URLs without ingesting content.

    Useful for previewing what will be ingested before committing to full ingestion.

    Returns a list of discovered URLs with their status.
    """,
)
async def crawl_website_route(
    url: Annotated[str, Query(description="Website URL to crawl")],
    current_user: Annotated[User, Depends(get_current_user)],
    crawl_depth: Annotated[
        CrawlDepth, Query(description="Crawl depth")
    ] = CrawlDepth.shallow,
    max_pages: Annotated[int, Query(ge=1, le=5000, description="Maximum pages")] = 50,
    timeout_seconds: Annotated[
        int, Query(ge=5, le=300, description="Request timeout")
    ] = 30,
) -> dict:
    """Crawl a website to discover URLs."""
    from app.schemas.website import WebsiteCrawlRequest

    request = WebsiteCrawlRequest(
        url=url,
        crawl_depth=crawl_depth,
        max_pages=max_pages,
        follow_redirects=True,
        timeout_seconds=timeout_seconds,
    )

    try:
        config = WebsiteProcessingConfig()
        result = await crawl_website(request, config)

        return {
            "url": result.base_url,
            "total_pages": len(result.pages),
            "successful_pages": len([p for p in result.pages if not p.error]),
            "failed_pages": len([p for p in result.pages if p.error]),
            "pages": [
                {
                    "url": p.url,
                    "status_code": p.status_code,
                    "word_count": p.word_count,
                    "error": p.error,
                }
                for p in result.pages
            ],
            "errors": result.errors,
        }

    except CrawlerError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Crawling failed: {e.message}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.get(
    "/status/{website_id}",
    response_model=dict,
    summary="Get website processing status",
    description="Get the current processing status and metadata for a website.",
)
async def get_website_status_route(
    website_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict:
    """Get website processing status."""
    # Verify ownership
    query = select(Website).where(
        Website.id == website_id,
        Website.user_id == current_user.id,
    )
    result = await db.execute(query)
    website = result.scalar_one_or_none()

    if not website:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Website not found",
        )

    try:
        status_info = await get_website_status(website_id)
        return status_info
    except IngestionError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )


@router.get(
    "",
    response_model=WebsiteListResponse,
    summary="List user's websites",
    description="Get paginated list of websites ingested by the current user.",
)
async def list_websites(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    status_filter: Annotated[
        WebsiteStatus | None, Query(description="Filter by status")
    ] = None,
) -> WebsiteListResponse:
    """List websites for the current user."""
    query = select(Website).where(Website.user_id == current_user.id)

    if status_filter:
        query = query.where(Website.status == SourceStatus(status_filter.value))

    # Get total count
    from sqlalchemy import func

    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query) or 0

    # Get paginated results
    query = query.order_by(Website.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await db.execute(query)
    websites = result.scalars().all()

    items = [
        WebsiteResponse(
            id=website.id,
            url=website.url,
            title=website.title,
            status=WebsiteStatus(website.status.value),
            last_crawled_at=website.last_crawled_at,
            metadata=website.metadata_,
            created_at=website.created_at,
            updated_at=website.updated_at,
        )
        for website in websites
    ]

    return WebsiteListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get(
    "/{website_id}",
    response_model=WebsiteResponse,
    summary="Get website details",
    description="Get detailed information about a specific website.",
)
async def get_website(
    website_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> WebsiteResponse:
    """Get website by ID."""
    query = select(Website).where(
        Website.id == website_id,
        Website.user_id == current_user.id,
    )
    result = await db.execute(query)
    website = result.scalar_one_or_none()

    if not website:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Website not found",
        )

    return WebsiteResponse(
        id=website.id,
        url=website.url,
        title=website.title,
        status=WebsiteStatus(website.status.value),
        last_crawled_at=website.last_crawled_at,
        metadata=website.metadata_,
        created_at=website.created_at,
        updated_at=website.updated_at,
    )


@router.delete(
    "/{website_id}",
    response_model=WebsiteDeleteResponse,
    summary="Delete a website",
    description="Delete a website and all its associated embeddings.",
)
async def delete_website_route(
    website_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> WebsiteDeleteResponse:
    """Delete a website."""
    try:
        await delete_website(website_id, current_user.id)

        return WebsiteDeleteResponse(
            website_id=website_id,
            message="Website and all associated embeddings deleted successfully.",
        )

    except IngestionError as e:
        if e.code == "WEBSITE_NOT_FOUND":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=e.message,
            )
        elif e.code == "UNAUTHORIZED":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=e.message,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=e.message,
            )
