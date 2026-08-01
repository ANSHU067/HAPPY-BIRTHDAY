"""Test configuration and fixtures for RAG tests."""

import os

import pytest


@pytest.fixture
def sample_documents():
    """Sample documents for RAG testing."""
    return [
        "LangChain is a framework for developing applications powered by language models.",
        "RAG (Retrieval Augmented Generation) combines retrieval with generation.",
        "Vector databases store embeddings and enable similarity search.",
        "LangGraph extends LangChain with graph-based workflows.",
        "Reranking improves retrieval quality using cross-encoder models.",
    ]


@pytest.fixture
def sample_metadatas():
    """Sample metadata for documents."""
    return [
        {"source": "langchain_docs", "topic": "framework"},
        {"source": "rag_guide", "topic": "rag"},
        {"source": "vector_db_intro", "topic": "storage"},
        {"source": "langgraph_docs", "topic": "framework"},
        {"source": "reranking_paper", "topic": "retrieval"},
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        "What is RAG?",
        "How does LangChain work?",
        "What are vector databases?",
        "Explain reranking",
    ]


@pytest.fixture
def test_collection_name():
    """Unique collection name for tests."""
    return f"test_collection_{os.getpid()}"


@pytest.fixture
def sample_transcript():
    """Sample transcript data for testing."""
    return [
        {"text": "Welcome to this video tutorial.", "start": 0.0, "duration": 2.5},
        {
            "text": "Today we will learn about machine learning.",
            "start": 2.5,
            "duration": 3.0,
        },
        {
            "text": "Machine learning is a subset of artificial intelligence.",
            "start": 5.5,
            "duration": 3.5,
        },
        {
            "text": "It focuses on training algorithms to make predictions.",
            "start": 9.0,
            "duration": 3.0,
        },
        {
            "text": "There are three main types of machine learning.",
            "start": 12.0,
            "duration": 3.0,
        },
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "success": True,
        "video_id": "test_video_123",
        "title": "Introduction to Machine Learning",
        "author": "Test Channel",
        "length": 600,
        "views": 10000,
        "description": "A comprehensive introduction to machine learning concepts.",
        "keywords": ["machine learning", "AI", "tutorial"],
    }


@pytest.fixture
def long_text():
    """Generate long text for chunking tests."""
    return " ".join([f"This is sentence number {i}." for i in range(100)])


@pytest.fixture
def mock_openai_key(monkeypatch):
    """Mock OpenAI API key for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-testing")
    return "sk-test-key-for-testing"
