import time

from app.rate_limiter import InMemoryRateLimiter, get_client_ip


class TestGetClientIp:
    def test_extracts_first_ip_from_x_forwarded_for(self):
        headers = {"X-Forwarded-For": "203.0.113.195, 70.41.3.18, 150.172.238.178"}
        assert get_client_ip(headers, "127.0.0.1") == "203.0.113.195"

    def test_handles_single_ip_in_x_forwarded_for(self):
        headers = {"X-Forwarded-For": "198.51.100.1"}
        assert get_client_ip(headers, "127.0.0.1") == "198.51.100.1"

    def test_case_insensitive_header_lookup(self):
        headers = {"x-forwarded-for": "198.51.100.2"}
        assert get_client_ip(headers, "127.0.0.1") == "198.51.100.2"

    def test_falls_back_to_x_real_ip(self):
        headers = {"X-Real-IP": "198.51.100.99"}
        assert get_client_ip(headers, "127.0.0.1") == "198.51.100.99"

    def test_falls_back_to_cf_connecting_ip(self):
        headers = {"CF-Connecting-IP": "198.51.100.77"}
        assert get_client_ip(headers, "127.0.0.1") == "198.51.100.77"

    def test_falls_back_to_remote_addr(self):
        assert get_client_ip({}, "10.0.0.5") == "10.0.0.5"

    def test_defaults_to_localhost_when_none(self):
        assert get_client_ip(None, None) == "127.0.0.1"


class TestInMemoryRateLimiter:
    def test_allows_requests_under_limit(self):
        limiter = InMemoryRateLimiter(requests_per_minute=3, window_seconds=60)
        client = "1.2.3.4"

        for _ in range(3):
            allowed, retry_after = limiter.is_allowed(client)
            assert allowed is True
            assert retry_after == 0

    def test_blocks_requests_over_limit(self):
        limiter = InMemoryRateLimiter(requests_per_minute=2, window_seconds=60)
        client = "1.2.3.4"

        assert limiter.is_allowed(client)[0] is True
        assert limiter.is_allowed(client)[0] is True

        allowed, retry_after = limiter.is_allowed(client)
        assert allowed is False
        assert 0 < retry_after <= 60

    def test_isolates_different_clients(self):
        limiter = InMemoryRateLimiter(requests_per_minute=1, window_seconds=60)
        client_a = "10.0.0.1"
        client_b = "10.0.0.2"

        assert limiter.is_allowed(client_a)[0] is True
        assert limiter.is_allowed(client_a)[0] is False  # A is blocked

        assert limiter.is_allowed(client_b)[0] is True  # B is still allowed

    def test_sliding_window_eviction(self):
        limiter = InMemoryRateLimiter(requests_per_minute=1, window_seconds=1)
        client = "1.2.3.4"

        assert limiter.is_allowed(client)[0] is True
        assert limiter.is_allowed(client)[0] is False

        # Sleep past the 1-second window
        time.sleep(1.1)

        allowed, _ = limiter.is_allowed(client)
        assert allowed is True

    def test_reset_clears_history(self):
        limiter = InMemoryRateLimiter(requests_per_minute=1, window_seconds=60)
        client = "1.2.3.4"

        limiter.is_allowed(client)
        assert limiter.is_allowed(client)[0] is False

        limiter.reset()
        assert limiter.is_allowed(client)[0] is True
