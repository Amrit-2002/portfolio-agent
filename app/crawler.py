import logging
import re
import time
import urllib.robotparser as robotparser
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _normalize_url(base_url: str, href: str) -> str | None:
    """Resolve relative links, strip fragments, and keep http(s) only."""
    if not href:
        return None
    # Resolve relative → absolute
    url = urljoin(base_url, href.strip())
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return None

    # Drop fragments ( #... ) to avoid duplicates
    parsed = parsed._replace(fragment="")
    # Optionally normalize default ports (80/443)
    if (parsed.scheme == "http" and parsed.port == 80) or (
        parsed.scheme == "https" and parsed.port == 443
    ):
        netloc = parsed.hostname or ""
        parsed = parsed._replace(netloc=netloc)

    return urlunparse(parsed)


def _same_domain(url: str, root_netloc: str, include_subdomains: bool) -> bool:
    """Return True if url is inside the same domain (optionally including subdomains)."""
    netloc = urlparse(url).netloc
    if include_subdomains:
        # Match exact domain or subdomains (e.g., blog.example.com)
        return netloc == root_netloc or netloc.endswith("." + root_netloc)
    return netloc == root_netloc


def _is_probably_binary(url: str) -> bool:
    """Fast filter by extension to skip non-HTML assets."""
    return bool(
        re.search(
            r"\.(pdf|jpg|jpeg|png|gif|svg|ico|webp|zip|tar|gz|rar|7z|mp3|mp4|mov|avi|wmv|mkv|exe|dmg|apk)$",
            url,
            re.I,
        )
    )


def discover_site_urls(
    start_url: str,
    max_pages: int = 200,
    include_subdomains: bool = False,
    user_agent: str = "Mozilla/5.0 (compatible; MiniCrawler/1.0)",
    request_timeout: int = 15,
    delay_seconds: float = 0.5,
    allow_paths: list[str] | None = None,
    deny_paths: list[str] | None = None,
) -> list[str]:
    """
    BFS crawl from start_url, staying on-domain, respecting robots.txt, and returning HTML page URLs.
    """
    start_parsed = urlparse(start_url)
    if start_parsed.scheme not in {"http", "https"}:
        raise ValueError("start_url must be http(s)")

    root = f"{start_parsed.scheme}://{start_parsed.netloc}"
    root_netloc = start_parsed.netloc

    rp = robotparser.RobotFileParser()
    rp.set_url(urljoin(root, "/robots.txt"))
    try:
        rp.read()
    except Exception:
        logger.debug("Could not fetch robots.txt for %s — proceeding without it.", root)

    headers = {"User-Agent": user_agent}
    seen: set[str] = set()
    queue = deque([start_url])
    results: list[str] = []

    while queue and len(results) < max_pages:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)

        try:
            if not rp.can_fetch(user_agent, url):
                continue
        except Exception:
            logger.debug("robots.txt check failed for %s — allowing.", url)

        path = urlparse(url).path or "/"
        if allow_paths and not any(path.startswith(p) for p in allow_paths):
            continue
        if deny_paths and any(path.startswith(p) for p in deny_paths):
            continue
        if not _same_domain(url, root_netloc, include_subdomains):
            continue
        if _is_probably_binary(url):
            continue

        # Fetch
        try:
            resp = requests.get(url, headers=headers, timeout=request_timeout)
        except Exception:
            continue

        # Only HTML pages
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" not in ctype.lower():
            # Not HTML - don't parse links, but also don't add to results
            time.sleep(delay_seconds)
            continue

        results.append(url)

        # Parse links and enqueue
        try:
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception:
            time.sleep(delay_seconds)
            continue

        for a in soup.find_all("a", href=True):
            nxt = _normalize_url(url, a["href"])
            if not nxt:
                continue
            if nxt in seen:
                continue
            if _is_probably_binary(nxt):
                continue
            if not _same_domain(nxt, root_netloc, include_subdomains):
                continue
            # Optional quick deny of mailto/tel etc. (already filtered by scheme in _normalize_url)
            queue.append(nxt)

        time.sleep(delay_seconds)

    return results


if __name__ == "__main__":
    START = "https://amrit-2002.github.io/Portfolio/"
    urls = discover_site_urls(
        START,
        max_pages=150,
        include_subdomains=False,
        delay_seconds=0.5,
        allow_paths=None,
        deny_paths=["/admin", "/login"],
    )
    print(f"Discovered {len(urls)} URLs")
    for u in urls:
        print(u)
