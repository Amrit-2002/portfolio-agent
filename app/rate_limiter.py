import logging
import threading
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


def get_client_ip(headers: dict | None = None, remote_addr: str | None = None) -> str:
    """
    Extract the real client IP address, taking into account reverse proxy headers
    (such as X-Forwarded-For from Render, Cloudflare, Nginx, or AWS).
    """
    if headers:
        # Case-insensitive header check
        xff = None
        for key, value in headers.items():
            if key.lower() == "x-forwarded-for" and value:
                xff = value
                break

        if xff:
            # X-Forwarded-For can be a comma-separated list: "client, proxy1, proxy2"
            parts = [p.strip() for p in xff.split(",") if p.strip()]
            if parts:
                return parts[0]

        # Alternative real-ip headers
        for h in ("x-real-ip", "cf-connecting-ip"):
            for key, value in headers.items():
                if key.lower() == h and value:
                    return value.strip()

    return remote_addr or "127.0.0.1"


class InMemoryRateLimiter:
    """
    Thread-safe in-memory sliding-window rate limiter per client IP.
    """

    def __init__(self, requests_per_minute: int = 10, window_seconds: int = 60):
        self.requests_per_minute = max(1, requests_per_minute)
        self.window_seconds = max(1, window_seconds)
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Check whether client_id is allowed to make a request.
        Returns:
            (allowed: bool, retry_after_seconds: int)
        """
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            history = self._requests[client_id]

            # Evict timestamps older than the sliding window
            while history and history[0] <= cutoff:
                history.popleft()

            if len(history) < self.requests_per_minute:
                history.append(now)
                return True, 0

            # Exceeded rate limit: calculate seconds until the oldest request expires
            oldest = history[0]
            retry_after = max(1, int(oldest + self.window_seconds - now + 0.999))
            return False, retry_after

    def reset(self, client_id: str | None = None) -> None:
        """Clear rate limit history for a specific client or all clients."""
        with self._lock:
            if client_id is not None:
                self._requests.pop(client_id, None)
            else:
                self._requests.clear()
