#!/bin/bash

# Start the FastAPI backend in the background (binds to localhost:8000)
python -m uvicorn app.backend:app --host 127.0.0.1 --port 8000 --timeout-keep-alive 200 &

# Start the Flask frontend in the foreground (binds to Render's public PORT)
python -m gunicorn -w 4 -b 0.0.0.0:$PORT "web.frontend_app:app" --timeout 300
