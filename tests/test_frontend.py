from unittest.mock import MagicMock, patch

import pytest

from web.frontend_app import app as flask_app
from web.frontend_app import limiter


@pytest.fixture()
def client():
    flask_app.config["TESTING"] = True
    flask_app.config["ENFORCE_RATE_LIMIT_IN_TESTS"] = False
    limiter.reset()
    with flask_app.test_client() as c:
        yield c
    limiter.reset()


class TestIndexRoute:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200


class TestAskRoute:
    def test_missing_question_returns_400(self, client):
        response = client.post("/ask", json={})
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_blank_question_returns_400(self, client):
        response = client.post("/ask", json={"question": "   "})
        assert response.status_code == 400

    def test_returns_answer_on_success(self, client):
        mock_body = {"data": {"askAQuestion": "Amrit built a Flutter token manager."}}
        with patch("web.frontend_app.requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.raise_for_status = lambda: None
            mock_post.return_value.json.return_value = mock_body

            response = client.post("/ask", json={"question": "What did Amrit build?"})

        assert response.status_code == 200
        assert response.get_json()["answer"] == "Amrit built a Flutter token manager."

    def test_graphql_errors_return_502(self, client):
        mock_body = {"errors": [{"message": "Something went wrong"}]}
        with patch("web.frontend_app.requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.raise_for_status = lambda: None
            mock_post.return_value.json.return_value = mock_body

            response = client.post("/ask", json={"question": "What is RAG?"})

        assert response.status_code == 502

    def test_backend_unreachable_returns_502(self, client):
        import requests as req

        with patch("web.frontend_app.requests.post", side_effect=req.ConnectionError):
            response = client.post("/ask", json={"question": "What is RAG?"})

        assert response.status_code == 502

    def test_rate_limit_exceeded_returns_429(self, client):
        flask_app.config["ENFORCE_RATE_LIMIT_IN_TESTS"] = True
        limiter.requests_per_minute = 2

        mock_body = {"data": {"askAQuestion": "Answer"}}
        with patch("web.frontend_app.requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.raise_for_status = lambda: None
            mock_post.return_value.json.return_value = mock_body

            # First 2 should succeed
            r1 = client.post("/ask", json={"question": "Q1?"})
            assert r1.status_code == 200
            r2 = client.post("/ask", json={"question": "Q2?"})
            assert r2.status_code == 200

            # 3rd should be rate limited
            r3 = client.post("/ask", json={"question": "Q3?"})
            assert r3.status_code == 429
            assert "Rate limit exceeded" in r3.get_json()["error"]
            assert "Retry-After" in r3.headers


class TestAskStreamRoute:
    def test_missing_question_returns_400(self, client):
        response = client.post("/ask-stream", json={})
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_blank_question_returns_400(self, client):
        response = client.post("/ask-stream", json={"question": "   "})
        assert response.status_code == 400

    def test_proxies_sse_chunks_from_backend(self, client):
        sse_chunks = [
            'data: {"token": "Hello"}\n\n',
            'data: {"token": " world"}\n\n',
            "data: [DONE]\n\n",
        ]

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_content.return_value = iter(sse_chunks)

        with patch("web.frontend_app.requests.post", return_value=mock_resp):
            response = client.post("/ask-stream", json={"question": "Q?"})
            body = response.get_data(as_text=True)

        assert response.status_code == 200
        assert response.headers["Content-Type"].startswith("text/event-stream")
        assert "Hello" in body
        assert " world" in body
        assert "[DONE]" in body

    def test_backend_unreachable_yields_error_then_done(self, client):
        import requests as req

        with patch(
            "web.frontend_app.requests.post",
            side_effect=req.ConnectionError("nope"),
        ):
            response = client.post("/ask-stream", json={"question": "Q?"})
            body = response.get_data(as_text=True)

        assert response.status_code == 200
        assert response.headers["Content-Type"].startswith("text/event-stream")
        assert "error" in body
        assert "nope" in body
        assert "[DONE]" in body

    def test_rate_limit_exceeded_on_stream_returns_429(self, client):
        flask_app.config["ENFORCE_RATE_LIMIT_IN_TESTS"] = True
        limiter.requests_per_minute = 1

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_content.return_value = iter(['data: {"token": "Hi"}\n\n'])

        with patch("web.frontend_app.requests.post", return_value=mock_resp):
            r1 = client.post("/ask-stream", json={"question": "Q1?"})
            assert r1.status_code == 200

            r2 = client.post("/ask-stream", json={"question": "Q2?"})
            assert r2.status_code == 429
            assert "Rate limit exceeded" in r2.get_json()["error"]
            assert "Retry-After" in r2.headers
