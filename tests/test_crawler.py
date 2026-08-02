import pytest

from app.crawler import _is_probably_binary, _normalize_url, _same_domain


class TestNormalizeUrl:
    def test_resolves_relative_path(self):
        result = _normalize_url("https://example.com/blog/", "../about")
        assert result == "https://example.com/about"

    def test_strips_fragment(self):
        result = _normalize_url("https://example.com/", "/page#section")
        assert result == "https://example.com/page"

    def test_keeps_https(self):
        result = _normalize_url("https://example.com/", "https://example.com/page")
        assert result == "https://example.com/page"

    def test_rejects_mailto(self):
        assert _normalize_url("https://example.com/", "mailto:user@example.com") is None

    def test_rejects_javascript(self):
        assert _normalize_url("https://example.com/", "javascript:void(0)") is None

    def test_rejects_empty_href(self):
        assert _normalize_url("https://example.com/", "") is None

    def test_strips_default_https_port(self):
        result = _normalize_url("https://example.com/", "https://example.com:443/page")
        assert result == "https://example.com/page"

    def test_strips_default_http_port(self):
        result = _normalize_url("http://example.com/", "http://example.com:80/page")
        assert result == "http://example.com/page"

    def test_keeps_non_default_port(self):
        result = _normalize_url("https://example.com/", "https://example.com:8080/api")
        assert "8080" in result


class TestSameDomain:
    def test_exact_match(self):
        assert _same_domain("https://example.com/page", "example.com", False) is True

    def test_different_domain(self):
        assert _same_domain("https://other.com/page", "example.com", False) is False

    def test_subdomain_excluded_by_default(self):
        assert _same_domain("https://blog.example.com/post", "example.com", False) is False

    def test_subdomain_included_when_flag_set(self):
        assert _same_domain("https://blog.example.com/post", "example.com", True) is True

    def test_unrelated_subdomain_excluded(self):
        assert _same_domain("https://notexample.com/", "example.com", True) is False


class TestIsProbablyBinary:
    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/file.pdf",
            "https://example.com/image.jpg",
            "https://example.com/photo.PNG",
            "https://example.com/archive.zip",
            "https://example.com/video.mp4",
            "https://example.com/app.exe",
        ],
    )
    def test_detects_binary_extensions(self, url):
        assert _is_probably_binary(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/",
            "https://example.com/about",
            "https://example.com/blog/post-1",
            "https://example.com/page.html",
            "https://example.com/page?ref=pdf",  # query param, not extension
        ],
    )
    def test_passes_html_urls(self, url):
        assert _is_probably_binary(url) is False
