import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when running `python web/frontend_app.py` directly
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import requests  # noqa: E402
from flask import Flask, Response, jsonify, render_template, request  # noqa: E402

from app.rate_limiter import InMemoryRateLimiter, get_client_ip  # noqa: E402

FASTAPI_GRAPHQL_URL = os.getenv("FASTAPI_GRAPHQL_URL", "http://localhost:8000/v1/graphql")
FASTAPI_STREAM_URL = os.getenv("FASTAPI_STREAM_URL", "http://localhost:8000/v1/stream")

# Rate limiting configuration (default: 10 requests per minute per IP)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
limiter = InMemoryRateLimiter(requests_per_minute=RATE_LIMIT_PER_MINUTE, window_seconds=60)

app = Flask(__name__, static_folder="static", template_folder="templates")


def _check_rate_limit():
    """
    Check if the current request exceeds the rate limit.
    Returns None if allowed, or a 429 Response tuple if exceeded.
    """
    if app.config.get("TESTING") and not app.config.get("ENFORCE_RATE_LIMIT_IN_TESTS"):
        return None

    client_ip = get_client_ip(request.headers, request.remote_addr)
    allowed, retry_after = limiter.is_allowed(client_ip)
    if not allowed:
        msg = f"Rate limit exceeded (max {limiter.requests_per_minute} questions/min). Please try again in {retry_after} seconds."
        response = jsonify({"error": msg})
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response
    return None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.post("/ask")
def ask():
    rate_limit_resp = _check_rate_limit()
    if rate_limit_resp is not None:
        return rate_limit_resp

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    chat_history = data.get("chatHistory") or []

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    payload = {
        "query": """query($q: String!, $h: [ChatMessageInput!]) {
            askAQuestion(question: $q, chatHistory: $h)
        }""",
        "variables": {
            "q": question,
            "h": [{"role": m["role"], "content": m["content"]} for m in chat_history],
        },
    }

    try:
        resp = requests.post(
            FASTAPI_GRAPHQL_URL,
            json=payload,
            timeout=120,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "flask-frontend/1.0",
            },
        )
        resp.raise_for_status()
        body = resp.json()

        if "errors" in body:
            return jsonify({"error": body["errors"]}), 502

        answer = body.get("data", {}).get("askAQuestion")
        return jsonify({"answer": answer})
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


@app.post("/ask-stream")
def ask_stream():
    rate_limit_resp = _check_rate_limit()
    if rate_limit_resp is not None:
        return rate_limit_resp

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    chat_history = data.get("chatHistory") or []

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    payload = {"question": question, "chatHistory": chat_history}

    def relay():
        try:
            with requests.post(
                FASTAPI_STREAM_URL,
                json=payload,
                stream=True,
                timeout=300,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "User-Agent": "flask-frontend/1.0",
                },
            ) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        yield chunk
        except requests.RequestException as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        relay(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5070")), debug=True)
