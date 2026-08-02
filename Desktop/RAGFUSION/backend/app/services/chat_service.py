"""Chat service with RAG integration and streaming support."""

import asyncio
import logging
import time
from typing import Any, AsyncIterator
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import MessageRole
from app.rag.pipelines.rag_pipeline import RAGPipeline
from app.schemas.chat import Citation, StreamChunk
from app.services.memory_service import MemoryService

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat operations with RAG."""

    def __init__(
        self,
        db: AsyncSession,
        user_id: UUID,
        rag_pipeline: RAGPipeline | None = None,
        max_context_messages: int = 10,
        max_tokens: int = 4000,
        timeout_seconds: int = 60,
    ):
        """Initialize chat service.

        Args:
            db: Database session
            user_id: Current user ID
            rag_pipeline: RAG pipeline instance (will create if None)
            max_context_messages: Maximum messages in context
            max_tokens: Maximum tokens in context
            timeout_seconds: Timeout for LLM responses
        """
        self.db = db
        self.user_id = user_id
        self.rag_pipeline = rag_pipeline
        self.memory_service = MemoryService(
            db=db,
            max_context_messages=max_context_messages,
            max_tokens=max_tokens,
        )
        self.timeout_seconds = timeout_seconds

    def _get_rag_pipeline(self) -> RAGPipeline:
        """Get or create RAG pipeline lazily."""
        if self.rag_pipeline is None:
            # Initialize with default settings
            self.rag_pipeline = RAGPipeline()
        return self.rag_pipeline

    async def chat(
        self,
        message: str,
        session_id: UUID | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int = 5,
        include_sources: bool = True,
    ) -> dict[str, Any]:
        """Process a chat message with RAG.

        Args:
            message: User message
            session_id: Existing session ID or None for new session
            max_tokens: Max tokens for LLM response
            temperature: LLM temperature
            top_k: Number of documents to retrieve
            include_sources: Whether to include source citations

        Returns:
            Dict with session_id, message, sources, token_usage, and processing_time_ms
        """
        start_time = time.time()

        # Get or create session
        if session_id:
            session = await self.memory_service.get_session(session_id, self.user_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
        else:
            session = await self.memory_service.create_session(
                user_id=self.user_id,
                title=message[:50] + "..." if len(message) > 50 else message,
            )
            session_id = session.id

        # Add user message to history
        user_msg = await self.memory_service.add_message(
            session_id=session_id,
            role=MessageRole.user,
            content=message,
        )

        # Get conversation context
        context_messages = await self.memory_service.get_context_with_token_limit(
            session_id
        )

        # Build messages for RAG pipeline
        llm_messages = self.memory_service.messages_to_llm_format(context_messages)

        try:
            # Run RAG pipeline with timeout
            response_data = await asyncio.wait_for(
                self._run_rag_pipeline(
                    message=message,
                    history=llm_messages[:-1],  # Exclude the current message
                    top_k=top_k,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                timeout=self.timeout_seconds,
            )

            # Extract response and citations
            response_content = response_data.get("response", "")
            citations_data = response_data.get("citations", [])
            token_usage = response_data.get("token_usage")

            # Add assistant message to history
            assistant_msg = await self.memory_service.add_message(
                session_id=session_id,
                role=MessageRole.assistant,
                content=response_content,
                citations=citations_data if include_sources else [],
                token_count=token_usage.get("total_tokens") if token_usage else None,
            )

            processing_time_ms = (time.time() - start_time) * 1000

            # Convert citations to schema format
            sources = []
            if include_sources:
                sources = self._convert_citations(citations_data)

            return {
                "session_id": session_id,
                "message": assistant_msg,
                "sources": sources,
                "token_usage": token_usage,
                "processing_time_ms": processing_time_ms,
            }

        except asyncio.TimeoutError:
            logger.error(f"Chat request timed out after {self.timeout_seconds}s")
            # Add error message
            error_content = "I apologize, but the request timed out. Please try again."
            error_msg = await self.memory_service.add_message(
                session_id=session_id,
                role=MessageRole.assistant,
                content=error_content,
            )

            return {
                "session_id": session_id,
                "message": error_msg,
                "sources": [],
                "token_usage": None,
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

        except Exception as e:
            logger.error(f"Error processing chat: {e}", exc_info=True)
            # Add error message
            error_content = f"An error occurred while processing your request: {str(e)}"
            error_msg = await self.memory_service.add_message(
                session_id=session_id,
                role=MessageRole.assistant,
                content=error_content,
            )

            return {
                "session_id": session_id,
                "message": error_msg,
                "sources": [],
                "token_usage": None,
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

    async def chat_stream(
        self,
        message: str,
        session_id: UUID | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int = 5,
        include_sources: bool = True,
    ) -> AsyncIterator[StreamChunk]:
        """Process a chat message with streaming response.

        Args:
            message: User message
            session_id: Existing session ID or None for new session
            max_tokens: Max tokens for LLM response
            temperature: LLM temperature
            top_k: Number of documents to retrieve
            include_sources: Whether to include source citations

        Yields:
            StreamChunk objects
        """
        start_time = time.time()

        try:
            # Get or create session
            if session_id:
                session = await self.memory_service.get_session(
                    session_id, self.user_id
                )
                if not session:
                    yield StreamChunk(
                        type="error", error=f"Session {session_id} not found"
                    )
                    return
            else:
                session = await self.memory_service.create_session(
                    user_id=self.user_id,
                    title=message[:50] + "..." if len(message) > 50 else message,
                )
                session_id = session.id

            # Yield session metadata
            yield StreamChunk(
                type="metadata",
                metadata={"session_id": str(session_id), "status": "started"},
            )

            # Add user message
            await self.memory_service.add_message(
                session_id=session_id,
                role=MessageRole.user,
                content=message,
            )

            # Get conversation context
            context_messages = await self.memory_service.get_context_with_token_limit(
                session_id
            )
            llm_messages = self.memory_service.messages_to_llm_format(context_messages)

            # Stream RAG pipeline response
            full_response = ""
            citations_data = []
            token_usage = None

            async for chunk in self._run_rag_pipeline_stream(
                message=message,
                history=llm_messages[:-1],
                top_k=top_k,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                if chunk.get("type") == "content":
                    content = chunk.get("content", "")
                    full_response += content
                    yield StreamChunk(type="content", content=content)

                elif chunk.get("type") == "citation":
                    citation_data = chunk.get("citation")
                    if citation_data and include_sources:
                        citations_data.append(citation_data)
                        citation = self._convert_citation(citation_data)
                        yield StreamChunk(type="citation", citation=citation)

                elif chunk.get("type") == "metadata":
                    token_usage = chunk.get("metadata", {}).get("token_usage")

            # Save assistant message
            await self.memory_service.add_message(
                session_id=session_id,
                role=MessageRole.assistant,
                content=full_response,
                citations=citations_data if include_sources else [],
                token_count=token_usage.get("total_tokens") if token_usage else None,
            )

            # Yield completion
            processing_time_ms = (time.time() - start_time) * 1000
            yield StreamChunk(
                type="done",
                metadata={
                    "processing_time_ms": processing_time_ms,
                    "token_usage": token_usage,
                },
            )

        except asyncio.TimeoutError:
            logger.error(f"Chat stream timed out after {self.timeout_seconds}s")
            yield StreamChunk(type="error", error="Request timed out")

        except Exception as e:
            logger.error(f"Error in chat stream: {e}", exc_info=True)
            yield StreamChunk(type="error", error=str(e))

    async def _run_rag_pipeline(
        self,
        message: str,
        history: list[dict[str, str]],
        top_k: int = 5,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Run RAG pipeline to generate response.

        Args:
            message: Current user message
            history: Conversation history
            top_k: Number of documents to retrieve
            max_tokens: Max tokens for response
            temperature: LLM temperature

        Returns:
            Dict with response, citations, and token_usage
        """
        pipeline = self._get_rag_pipeline()

        # Build RAG state
        state = {
            "question": message,
            "metadata": {
                "top_k": top_k,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        }

        # Run pipeline
        result = await asyncio.to_thread(pipeline.graph.invoke, state)

        # Extract response
        response = result.get("response", "")

        # Extract citations from reranked docs
        citations = []
        for doc in result.get("reranked_docs", [])[:top_k]:
            content, metadata, score = doc
            citations.append(
                {
                    "source_id": metadata.get("source_id"),
                    "source_type": metadata.get("source_type", "unknown"),
                    "content": content[:200],  # Limit snippet length
                    "score": float(score) if score else 0.0,
                    "metadata": metadata,
                }
            )

        # Extract token usage if available
        token_usage = result.get("metadata", {}).get("token_usage")

        return {
            "response": response,
            "citations": citations,
            "token_usage": token_usage,
        }

    async def _run_rag_pipeline_stream(
        self,
        message: str,
        history: list[dict[str, str]],
        top_k: int = 5,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run RAG pipeline with streaming.

        Args:
            message: Current user message
            history: Conversation history
            top_k: Number of documents to retrieve
            max_tokens: Max tokens
            temperature: LLM temperature

        Yields:
            Chunks with type: content, citation, or metadata
        """
        pipeline = self._get_rag_pipeline()

        # For now, simulate streaming by running full pipeline and chunking response
        # TODO: Implement true streaming with LLM provider stream methods
        result = await self._run_rag_pipeline(
            message=message,
            history=history,
            top_k=top_k,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Stream response content in chunks
        response = result.get("response", "")
        chunk_size = 20  # Characters per chunk

        for i in range(0, len(response), chunk_size):
            chunk = response[i : i + chunk_size]
            yield {"type": "content", "content": chunk}
            await asyncio.sleep(0.01)  # Small delay for realistic streaming

        # Yield citations
        for citation in result.get("citations", []):
            yield {"type": "citation", "citation": citation}

        # Yield metadata
        yield {
            "type": "metadata",
            "metadata": {"token_usage": result.get("token_usage")},
        }

    def _convert_citation(self, citation_data: dict[str, Any]) -> Citation:
        """Convert citation data to Citation schema.

        Args:
            citation_data: Raw citation data

        Returns:
            Citation schema object
        """
        return Citation(
            source_id=citation_data.get("source_id"),
            source_type=citation_data.get("source_type", "unknown"),
            content=citation_data.get("content", ""),
            score=citation_data.get("score", 0.0),
            metadata=citation_data.get("metadata", {}),
        )

    def _convert_citations(
        self, citations_data: list[dict[str, Any]]
    ) -> list[Citation]:
        """Convert list of citation data to Citation schemas.

        Args:
            citations_data: List of raw citation data

        Returns:
            List of Citation schema objects
        """
        return [self._convert_citation(c) for c in citations_data]
