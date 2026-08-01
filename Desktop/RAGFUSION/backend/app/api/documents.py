"""Document ingestion API routes."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import (APIRouter, Depends, File, HTTPException, Query,
                     UploadFile, status)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user_id as get_current_user
from app.config.settings import get_settings
from app.db.session import get_db_session
from app.models.entities import Document, SourceStatus, User
from app.schemas.document import (DocumentFormat, DocumentListResponse,
                                  DocumentResponse, DocumentStatus,
                                  DocumentUploadResponse, ProcessingConfig)
from app.services.document import (DocumentProcessingError,
                                   DocumentValidationError, process_document,
                                   upload_document)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for processing",
    description="""
    Upload a document for ingestion into the RAG system.

    Supported formats:
    - PDF (application/pdf)
    - DOCX (application/vnd.openxmlformats-officedocument.wordprocessingml.document)
    - TXT (text/plain)
    - CSV (text/csv)
    - Markdown (text/markdown, text/x-markdown)

    Maximum file size: 100MB

    The document will be processed asynchronously through the pipeline:
    validation -> cleaning -> chunking -> embedding -> storage
    """,
)
async def upload_document_route(
    file: Annotated[UploadFile, File(description="Document file to upload")],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    chunk_size: Annotated[
        int, Query(ge=100, le=10000, description="Chunk size in tokens")
    ] = 1000,
    chunk_overlap: Annotated[
        int, Query(ge=0, le=1000, description="Chunk overlap in tokens")
    ] = 200,
    chunking_strategy: Annotated[
        str, Query(description="Chunking strategy")
    ] = "recursive",
    embedding_model: Annotated[
        str, Query(description="Embedding model name")
    ] = "text-embedding-3-small",
    extract_metadata: Annotated[
        bool, Query(description="Extract document metadata")
    ] = True,
    clean_text: Annotated[bool, Query(description="Clean extracted text")] = True,
) -> DocumentUploadResponse:
    """Upload a document for processing."""
    settings = get_settings()

    # Validate file size
    if file.size and file.size > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum of {settings.max_file_size_mb}MB",
        )

    # Read file content
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file",
        )

    # Validate content type
    content_type = file.content_type or "application/octet-stream"
    if content_type not in [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/csv",
        "text/markdown",
        "text/x-markdown",
    ]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {content_type}. Supported: PDF, DOCX, TXT, CSV, Markdown",
        )

    # Create processing config
    config = ProcessingConfig(
        chunking={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "strategy": chunking_strategy,
        },
        embedding={"model_name": embedding_model},
        extract_metadata=extract_metadata,
        clean_text=clean_text,
    )

    try:
        # Upload document
        document = await upload_document(
            file_content=content,
            filename=file.filename or "unknown",
            content_type=content_type,
            user_id=current_user.id,
            config=config,
        )

        # Process document in background (for now, synchronously)
        # TODO: Move to background task queue
        try:
            file_path = Path(settings.upload_dir) / document.storage_key
            await process_document(
                document.id,
                file_path,
                detect_format_from_content_type(content_type),
                config,
            )
        except Exception:
            # Document status will be updated to failed in process_document
            pass

        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            status=DocumentStatus(document.status.value),
            message="Document uploaded successfully. Processing started.",
        )

    except DocumentValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
    except DocumentProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}",
        )


def detect_format_from_content_type(content_type: str) -> DocumentFormat:
    """Map content type to DocumentFormat."""
    mapping = {
        "application/pdf": DocumentFormat.pdf,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentFormat.docx,
        "text/plain": DocumentFormat.txt,
        "text/csv": DocumentFormat.csv,
        "text/markdown": DocumentFormat.markdown,
        "text/x-markdown": DocumentFormat.markdown,
    }
    return mapping.get(content_type, DocumentFormat.txt)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List user's documents",
    description="Get paginated list of documents uploaded by the current user.",
)
async def list_documents(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    status_filter: Annotated[
        DocumentStatus | None, Query(description="Filter by status")
    ] = None,
) -> DocumentListResponse:
    """List documents for the current user."""
    query = select(Document).where(Document.user_id == current_user.id)

    if status_filter:
        query = query.where(Document.status == SourceStatus(status_filter.value))

    # Get total count
    from sqlalchemy import func

    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query) or 0

    # Get paginated results
    query = query.order_by(Document.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await db.execute(query)
    documents = result.scalars().all()

    items = [
        DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            storage_key=doc.storage_key,
            mime_type=doc.mime_type,
            size_bytes=doc.size_bytes,
            checksum=doc.checksum,
            status=DocumentStatus(doc.status.value),
            metadata=None,  # Would need to parse metadata_ JSON
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )
        for doc in documents
    ]

    return DocumentListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document details",
    description="Get detailed information about a specific document.",
)
async def get_document(
    document_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> DocumentResponse:
    """Get document by ID."""
    query = select(Document).where(
        Document.id == document_id,
        Document.user_id == current_user.id,
    )
    result = await db.execute(query)
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Parse metadata if available
    metadata = None
    if document.metadata_:
        from app.schemas.document import DocumentMetadata

        try:
            metadata = DocumentMetadata(**document.metadata_)
        except Exception:
            pass

    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        storage_key=document.storage_key,
        mime_type=document.mime_type,
        size_bytes=document.size_bytes,
        checksum=document.checksum,
        status=DocumentStatus(document.status.value),
        metadata=metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.delete(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Delete a document",
    description="Delete a document and all its associated embeddings.",
)
async def delete_document(
    document_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> DocumentResponse:
    """Delete a document."""
    query = select(Document).where(
        Document.id == document_id,
        Document.user_id == current_user.id,
    )
    result = await db.execute(query)
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Delete file from storage
    settings = get_settings()
    file_path = Path(settings.upload_dir) / document.storage_key
    if file_path.exists():
        file_path.unlink()

    # Delete document (cascades to embeddings)
    await db.delete(document)
    await db.commit()

    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        storage_key=document.storage_key,
        mime_type=document.mime_type,
        size_bytes=document.size_bytes,
        checksum=document.checksum,
        status=DocumentStatus(document.status.value),
        metadata=None,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.post(
    "/{document_id}/reprocess",
    response_model=DocumentUploadResponse,
    summary="Reprocess a document",
    description="Reprocess a document through the ingestion pipeline with new configuration.",
)
async def reprocess_document(
    document_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    chunk_size: Annotated[int, Query(ge=100, le=10000)] = 1000,
    chunk_overlap: Annotated[int, Query(ge=0, le=1000)] = 200,
    chunking_strategy: Annotated[str, Query()] = "recursive",
    embedding_model: Annotated[str, Query()] = "text-embedding-3-small",
    extract_metadata: Annotated[bool, Query()] = True,
    clean_text: Annotated[bool, Query()] = True,
) -> DocumentUploadResponse:
    """Reprocess a document with new configuration."""
    query = select(Document).where(
        Document.id == document_id,
        Document.user_id == current_user.id,
    )
    result = await db.execute(query)
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    settings = get_settings()
    file_path = Path(settings.upload_dir) / document.storage_key

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found in storage",
        )

    config = ProcessingConfig(
        chunking={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "strategy": chunking_strategy,
        },
        embedding={"model_name": embedding_model},
        extract_metadata=extract_metadata,
        clean_text=clean_text,
    )

    try:
        format_ = detect_format_from_content_type(
            document.mime_type or "application/octet-stream"
        )
        await process_document(document.id, file_path, format_, config)

        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            status=DocumentStatus.ready,
            message="Document reprocessed successfully.",
        )
    except DocumentProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reprocessing failed: {str(e)}",
        )
