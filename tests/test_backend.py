import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from app.backend import app


@pytest.fixture()
def client():
    return TestClient(app)


def _parse_sse(body: str) -> list[str]:
    """Extract the list of `data:` payloads from a raw SSE body."""
    payloads = []
    for event in body.split("\n\n"):
        for line in event.split("\n"):
            if line.startswith("data: "):
                payloads.append(line[len("data: ") :])
    return payloads


class TestStreamEndpoint:
    def test_returns_sse_content_type(self, client):
        with patch("app.backend.retrieve_generate_stream", return_value=iter(["hi"])):
            with client.stream("POST", "/v1/stream", json={"question": "Q?"}) as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                # consume so the generator finishes cleanly
                for _ in resp.iter_text():
                    pass

    def test_yields_token_events_then_done(self, client):
        with patch(
            "app.backend.retrieve_generate_stream",
            return_value=iter(["Hello", " world"]),
        ):
            with client.stream("POST", "/v1/stream", json={"question": "Q?"}) as resp:
                body = "".join(resp.iter_text())

        payloads = _parse_sse(body)
        assert payloads[-1] == "[DONE]"

        tokens = [json.loads(p)["token"] for p in payloads if p != "[DONE]"]
        assert tokens == ["Hello", " world"]

    def test_passes_question_and_history_to_stream_function(self, client):
        captured = {}

        def fake_stream(question, history):
            captured["question"] = question
            captured["history"] = history
            yield "ok"

        with patch("app.backend.retrieve_generate_stream", side_effect=fake_stream):
            with client.stream(
                "POST",
                "/v1/stream",
                json={
                    "question": "Follow-up?",
                    "chatHistory": [
                        {"role": "human", "content": "Hello"},
                        {"role": "ai", "content": "Hi there"},
                    ],
                },
            ) as resp:
                for _ in resp.iter_text():
                    pass

        assert captured["question"] == "Follow-up?"
        assert len(captured["history"]) == 2
        assert isinstance(captured["history"][0], HumanMessage)
        assert captured["history"][0].content == "Hello"
        assert isinstance(captured["history"][1], AIMessage)
        assert captured["history"][1].content == "Hi there"

    def test_defaults_to_empty_history_when_omitted(self, client):
        captured = {}

        def fake_stream(question, history):
            captured["history"] = history
            yield "ok"

        with patch("app.backend.retrieve_generate_stream", side_effect=fake_stream):
            with client.stream("POST", "/v1/stream", json={"question": "Q?"}) as resp:
                for _ in resp.iter_text():
                    pass

        assert captured["history"] == []

    def test_handles_exception_with_error_event(self, client):
        def boom(*args, **kwargs):
            yield "partial"
            raise RuntimeError("explosion")

        with patch("app.backend.retrieve_generate_stream", side_effect=boom):
            with client.stream("POST", "/v1/stream", json={"question": "Q?"}) as resp:
                body = "".join(resp.iter_text())

        payloads = _parse_sse(body)
        assert payloads[-1] == "[DONE]"

        # Should have one token event before the error event.
        non_done = [p for p in payloads if p != "[DONE]"]
        events = [json.loads(p) for p in non_done]
        assert events[0] == {"token": "partial"}
        assert "error" in events[1]
        assert "explosion" in events[1]["error"]

    def test_missing_question_returns_422(self, client):
        # Pydantic validation on StreamRequest rejects empty body.
        resp = client.post("/v1/stream", json={})
        assert resp.status_code == 422
