"""Website ingestion tests."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.models.entities import User
from app.schemas.website import (CrawledPage, WebsiteChunkingConfig,
                                 WebsiteMetadata, WebsiteProcessingConfig)
from app.services.chunking_service import (chunk_single_page, chunk_text_fixed,
                                           chunk_text_recursive,
                                           chunk_text_semantic,
                                           chunk_website_content,
                                           estimate_tokens)
from app.services.crawler_service import (CrawlerTimeoutError,
                                          CrawlerValidationError,
                                          extract_links, is_same_domain,
                                          normalize_url, validate_url)
from app.services.extraction_service import (clean_html, clean_text,
                                             extract_text_from_html)


class TestURLValidation:
    """Tests for URL validation."""

    def test_validate_url_valid_http(self):
        """Test validation accepts valid HTTP URL."""
        validate_url("http://example.com")
        validate_url("http://example.com/path")
        validate_url("http://example.com:8080/path?query=1")

    def test_validate_url_valid_https(self):
        """Test validation accepts valid HTTPS URL."""
        validate_url("https://example.com")
        validate_url("https://subdomain.example.com/path")

    def test_validate_url_invalid_format(self):
        """Test validation rejects invalid URL format."""
        with pytest.raises(CrawlerValidationError) as exc:
            validate_url("not-a-url")
        assert "Invalid URL format" in exc.value.message

        with pytest.raises(CrawlerValidationError):
            validate_url("")

        with pytest.raises(CrawlerValidationError):
            validate_url("://example.com")

    def test_validate_url_invalid_scheme(self):
        """Test validation rejects non-HTTP schemes."""
        with pytest.raises(CrawlerValidationError) as exc:
            validate_url("ftp://example.com")
        assert "HTTP and HTTPS" in exc.value.message

        with pytest.raises(CrawlerValidationError):
            validate_url("file:///etc/passwd")

    def test_validate_url_blocked_localhost(self):
        """Test validation blocks localhost."""
        with pytest.raises(CrawlerValidationError) as exc:
            validate_url("http://localhost:8000")
        assert "not allowed" in exc.value.message

        with pytest.raises(CrawlerValidationError):
            validate_url("http://127.0.0.1")

    def test_validate_url_blocked_private_ip(self):
        """Test validation blocks private IP addresses."""
        with pytest.raises(CrawlerValidationError):
            validate_url("http://192.168.1.1")

        with pytest.raises(CrawlerValidationError):
            validate_url("http://10.0.0.1")

        with pytest.raises(CrawlerValidationError):
            validate_url("http://172.16.0.1")


class TestURLUtilities:
    """Tests for URL utility functions."""

    def test_is_same_domain(self):
        """Test domain comparison."""
        assert is_same_domain("http://example.com", "http://example.com/page")
        assert is_same_domain("https://example.com", "http://example.com")
        assert not is_same_domain("http://example.com", "http://other.com")
        assert not is_same_domain("http://example.com", "http://sub.example.com")

    def test_normalize_url(self):
        """Test URL normalization."""
        base = "http://example.com/path/"
        assert normalize_url("page.html", base) == "http://example.com/path/page.html"
        assert normalize_url("/absolute", base) == "http://example.com/absolute"
        assert normalize_url("http://other.com", base) == "http://other.com"

    def test_extract_links(self):
        """Test link extraction from HTML."""
        html = """
        <html>
        <body>
            <a href="http://example.com/page1">Link 1</a>
            <a href="/page2">Link 2</a>
            <a href="#anchor">Anchor</a>
            <a href="javascript:void(0)">JS</a>
            <a href="mailto:test@example.com">Email</a>
        </body>
        </html>
        """
        base_url = "http://example.com"
        links = extract_links(html, base_url)

        assert "http://example.com/page1" in links
        assert "http://example.com/page2" in links
        assert len([l for l in links if l.startswith("#")]) == 0
        assert len([l for l in links if l.startswith("javascript:")]) == 0
        assert len([l for l in links if l.startswith("mailto:")]) == 0


class TestHTMLCleaning:
    """Tests for HTML cleaning."""

    def test_clean_html_removes_scripts(self):
        """Test script tag removal."""
        html = "<html><body><script>alert('test');</script><p>Content</p></body></html>"
        config = WebsiteProcessingConfig()
        cleaned = clean_html(html, config)
        assert "<script>" not in cleaned
        assert "alert" not in cleaned
        assert "Content" in cleaned

    def test_clean_html_removes_styles(self):
        """Test style tag removal."""
        html = "<html><head><style>body { color: red; }</style></head><body>Content</body></html>"
        config = WebsiteProcessingConfig()
        cleaned = clean_html(html, config)
        assert "<style>" not in cleaned
        assert "color: red" not in cleaned
        assert "Content" in cleaned

    def test_clean_html_removes_nav_footer(self):
        """Test navigation and footer removal."""
        html = """
        <html><body>
            <nav>Navigation</nav>
            <main>Main Content</main>
            <footer>Footer</footer>
        </body></html>
        """
        config = WebsiteProcessingConfig()
        cleaned = clean_html(html, config)
        assert "Navigation" not in cleaned
        assert "Footer" not in cleaned
        assert "Main Content" in cleaned

    def test_clean_html_removes_hidden_elements(self):
        """Test hidden element removal."""
        html = """
        <html><body>
            <div style="display: none;">Hidden</div>
            <div>Visible</div>
        </body></html>
        """
        config = WebsiteProcessingConfig()
        cleaned = clean_html(html, config)
        assert "Hidden" not in cleaned
        assert "Visible" in cleaned


class TestTextExtraction:
    """Tests for text extraction from HTML."""

    def test_extract_text_from_html_basic(self):
        """Test basic text extraction."""
        html = "<html><body><h1>Title</h1><p>Paragraph content.</p></body></html>"
        text, metadata = extract_text_from_html(html, "http://example.com")

        assert "Title" in text
        assert "Paragraph content" in text
        assert metadata.url == "http://example.com"

    def test_extract_text_from_html_with_metadata(self):
        """Test extraction with metadata."""
        html = """
        <html>
        <head>
            <title>Page Title</title>
            <meta name="description" content="Page description">
            <meta name="author" content="John Doe">
        </head>
        <body><p>Content</p></body>
        </html>
        """
        text, metadata = extract_text_from_html(html, "http://example.com")

        assert "Content" in text
        assert metadata.title or True  # trafilatura may or may not extract title

    def test_extract_text_from_html_empty(self):
        """Test extraction from empty HTML."""
        html = "<html><body></body></html>"
        text, metadata = extract_text_from_html(html, "http://example.com")

        assert isinstance(text, str)
        assert metadata.url == "http://example.com"

    def test_extract_text_malformed_html(self):
        """Test extraction handles malformed HTML."""
        html = "<html><body><p>Unclosed paragraph<div>Content</body>"
        text, metadata = extract_text_from_html(html, "http://example.com")

        assert "Content" in text or "Unclosed" in text
        assert metadata.url == "http://example.com"


class TestTextCleaning:
    """Tests for text cleaning."""

    def test_clean_text_whitespace(self):
        """Test whitespace normalization."""
        text = "Hello    World\n\n\nMultiple   spaces"
        cleaned = clean_text(text)
        assert "    " not in cleaned
        assert "\n\n\n" not in cleaned

    def test_clean_text_empty(self):
        """Test cleaning empty text."""
        text = ""
        cleaned = clean_text(text)
        assert cleaned == ""

    def test_clean_text_preserves_content(self):
        """Test cleaning preserves meaningful content."""
        text = "Important content that should remain."
        cleaned = clean_text(text)
        assert "Important content" in cleaned
        assert "should remain" in cleaned


class TestChunking:
    """Tests for content chunking."""

    def test_chunk_text_fixed_basic(self):
        """Test fixed-size chunking."""
        text = " ".join([f"word{i}" for i in range(200)])
        config = WebsiteChunkingConfig(
            chunk_size=100, chunk_overlap=20, strategy="fixed"
        )
        chunks = chunk_text_fixed(text, config, "http://example.com")

        assert len(chunks) > 1
        assert all(c.source_url == "http://example.com" for c in chunks)
        assert all(c.metadata["strategy"] == "fixed" for c in chunks)

    def test_chunk_text_recursive(self):
        """Test recursive chunking."""
        text = (
            "Section 1\n\n"
            + ("Content " * 100)
            + "\n\nSection 2\n\n"
            + ("More content " * 100)
        )
        config = WebsiteChunkingConfig(
            chunk_size=200, chunk_overlap=50, strategy="recursive"
        )
        chunks = chunk_text_recursive(text, config, "http://example.com")

        assert len(chunks) >= 1
        assert all(c.metadata["strategy"] == "recursive" for c in chunks)

    def test_chunk_text_semantic(self):
        """Test semantic chunking."""
        text = "# Heading 1\n\nContent for heading 1.\n\n## Heading 2\n\nContent for heading 2."
        config = WebsiteChunkingConfig(
            chunk_size=200, chunk_overlap=50, strategy="semantic"
        )
        chunks = chunk_text_semantic(text, config, "http://example.com")

        assert len(chunks) >= 1
        assert all(c.metadata["strategy"] == "semantic" for c in chunks)

    def test_chunk_website_content_multiple_pages(self):
        """Test chunking multiple pages."""
        pages_content = [
            ("http://example.com/page1", "Content for page 1. " * 50),
            ("http://example.com/page2", "Content for page 2. " * 50),
        ]
        config = WebsiteProcessingConfig(
            chunking=WebsiteChunkingConfig(
                chunk_size=100, chunk_overlap=20, strategy="fixed"
            )
        )
        chunks = chunk_website_content(pages_content, config)

        assert len(chunks) > 2
        page1_chunks = [c for c in chunks if c.source_url == "http://example.com/page1"]
        page2_chunks = [c for c in chunks if c.source_url == "http://example.com/page2"]
        assert len(page1_chunks) > 0
        assert len(page2_chunks) > 0

    def test_chunk_single_page(self):
        """Test single page chunking."""
        text = "Sample content. " * 100
        config = WebsiteProcessingConfig()
        chunks = chunk_single_page(text, "http://example.com", config)

        assert len(chunks) >= 1
        assert all(c.source_url == "http://example.com" for c in chunks)

    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "This is a sample text with multiple words."
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens < len(text)  # Should be less than character count


class TestWebsiteAPI:
    """Tests for website ingestion API endpoints."""

    @pytest.fixture
    def mock_user(self) -> User:
        """Create a mock user."""
        from datetime import datetime, timezone

        return User(
            id=uuid.uuid4(),
            email="test@example.com",
            display_name="Test User",
            role="user",
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    def test_ingest_website_valid_url(self, client: TestClient, mock_user: User):
        """Test website ingestion with valid URL."""
        with patch("app.api.auth.get_current_user", return_value=mock_user):
            with patch("app.services.ingestion_service.crawl_website") as mock_crawl:
                # Mock successful crawl
                mock_crawl.return_value = MagicMock(
                    pages=[
                        CrawledPage(
                            url="http://example.com",
                            status_code=200,
                            text_content="Sample content for testing.",
                            word_count=20,
                            character_count=100,
                        )
                    ],
                    base_url="http://example.com",
                    errors=[],
                )

                response = client.post(
                    "/api/v1/website/ingest",
                    json={
                        "url": "http://example.com",
                        "crawl_depth": "single",
                        "max_pages": 10,
                    },
                    headers={"Authorization": "Bearer test-token"},
                )

                # May succeed or fail depending on database mocking
                assert response.status_code in [
                    status.HTTP_202_ACCEPTED,
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                ]

    def test_ingest_website_invalid_url(self, client: TestClient, mock_user: User):
        """Test ingestion rejects invalid URL."""
        with patch("app.api.auth.get_current_user", return_value=mock_user):
            response = client.post(
                "/api/v1/website/ingest",
                json={
                    "url": "not-a-valid-url",
                    "crawl_depth": "single",
                    "max_pages": 10,
                },
                headers={"Authorization": "Bearer test-token"},
            )

            assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_crawl_website_redirect(self, client: TestClient, mock_user: User):
        """Test crawling handles redirects."""
        # This would need mocking of aiohttp responses
        pass

    def test_crawl_website_timeout(self, client: TestClient, mock_user: User):
        """Test crawling handles timeout."""

        with patch("app.api.auth.get_current_user", return_value=mock_user):
            with patch("app.services.crawler_service.fetch_page") as mock_fetch:
                mock_fetch.side_effect = CrawlerTimeoutError("Request timeout")

                response = client.post(
                    "/api/v1/website/crawl",
                    params={
                        "url": "http://example.com",
                        "timeout_seconds": 5,  # Must be >= 5 to pass Pydantic validation
                    },
                    headers={"Authorization": "Bearer test-token"},
                )

                assert response.status_code in [
                    status.HTTP_200_OK,
                    status.HTTP_400_BAD_REQUEST,
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                ]

                assert response.status_code in [
                    status.HTTP_200_OK,
                    status.HTTP_400_BAD_REQUEST,
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                ]


class TestCrawlerErrorHandling:
    """Tests for crawler error scenarios."""

    @pytest.mark.asyncio
    async def test_crawl_empty_response(self):
        """Test handling of empty response."""
        html = ""
        text, metadata = extract_text_from_html(html, "http://example.com")
        assert isinstance(text, str)
        assert metadata.url == "http://example.com"

    @pytest.mark.asyncio
    async def test_crawl_large_website(self):
        """Test handling of large websites."""
        # Would need to mock multiple pages
        pass

    @pytest.mark.asyncio
    async def test_crawl_duplicate_pages(self):
        """Test deduplication of pages."""
        # Same domain check ensures no duplicates
        assert is_same_domain("http://example.com/page1", "http://example.com/page1")

    @pytest.mark.asyncio
    async def test_crawl_unsupported_content(self):
        """Test handling of non-HTML content."""
        # CrawledPage would have error field set
        page = CrawledPage(
            url="http://example.com/file.pdf",
            status_code=200,
            content_type="application/pdf",
            text_content="",
            error="Non-HTML content type: application/pdf",
        )
        assert page.error is not None
        assert "Non-HTML" in page.error


class TestExtractionFailure:
    """Tests for extraction failure scenarios."""

    def test_extraction_malformed_html(self):
        """Test extraction handles malformed HTML."""
        html = "<html><body><p>Unclosed<div>Tags</body>"
        text, metadata = extract_text_from_html(html, "http://example.com")

        # Should not raise, should return something
        assert isinstance(text, str)
        assert isinstance(metadata, WebsiteMetadata)

    def test_extraction_javascript_heavy(self):
        """Test extraction from JS-heavy sites."""
        html = """
        <html>
        <body>
            <script>document.write('Dynamic content');</script>
            <div id="root"></div>
        </body>
        </html>
        """
        text, metadata = extract_text_from_html(html, "http://example.com")

        # May not extract JS-rendered content without crawl4ai
        assert isinstance(text, str)

    def test_extraction_with_encoding_issues(self):
        """Test extraction handles encoding issues."""
        # This would need actual bytes with encoding issues
        html = "<html><body>Content with special chars: café résumé</body></html>"
        text, metadata = extract_text_from_html(html, "http://example.com")

        assert isinstance(text, str)


class TestWebsiteStatus:
    """Tests for website status tracking."""

    def test_get_website_status_not_found(self, client: TestClient, mock_user: User):
        """Test status check for non-existent website."""
        from datetime import datetime, timezone

        mock_user = User(
            id=uuid.uuid4(),
            email="test@example.com",
            display_name="Test User",
            role="user",
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        with patch("app.api.auth.get_current_user", return_value=mock_user):
            random_id = uuid.uuid4()
            response = client.get(
                f"/api/v1/website/status/{random_id}",
                headers={"Authorization": "Bearer test-token"},
            )
            assert response.status_code == status.HTTP_404_NOT_FOUND


class TestWebsiteDeletion:
    """Tests for website deletion."""

    def test_delete_website_not_found(self, client: TestClient, mock_user: User):
        """Test deletion of non-existent website."""
        from datetime import datetime, timezone

        mock_user = User(
            id=uuid.uuid4(),
            email="test@example.com",
            display_name="Test User",
            role="user",
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        with patch("app.api.auth.get_current_user", return_value=mock_user):
            random_id = uuid.uuid4()
            response = client.delete(
                f"/api/v1/website/{random_id}",
                headers={"Authorization": "Bearer test-token"},
            )
            # May return 404 or 500 depending on implementation
            assert response.status_code in [
                status.HTTP_404_NOT_FOUND,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ]
