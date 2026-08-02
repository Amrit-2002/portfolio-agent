import json
import os

import requests
from flask import Flask, Response, jsonify, render_template, request

FASTAPI_GRAPHQL_URL = os.getenv("FASTAPI_GRAPHQL_URL", "http://localhost:8000/v1/graphql")
FASTAPI_STREAM_URL = os.getenv("FASTAPI_STREAM_URL", "http://localhost:8000/v1/stream")

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.post("/ask")
def ask():
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
