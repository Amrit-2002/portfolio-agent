import json

import strawberry
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from strawberry.fastapi import GraphQLRouter

from .rag_application import retrieve_generate, retrieve_generate_stream


@strawberry.input
class ChatMessageInput:
    role: str
    content: str


@strawberry.type
class Query:
    @strawberry.field
    def askAQuestion(
        self,
        question: str,
        chat_history: list[ChatMessageInput] | None = None,
    ) -> str:
        history = _to_lc_messages(chat_history)
        return retrieve_generate(question, history)


schema = strawberry.Schema(Query)

app = FastAPI(title="RAG Backend")
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/v1/graphql")


class ChatMessage(BaseModel):
    role: str
    content: str


class StreamRequest(BaseModel):
    question: str
    chatHistory: list[ChatMessage] | None = None


def _to_lc_messages(history):
    """Convert dict/Strawberry chat messages to LangChain HumanMessage/AIMessage."""
    out = []
    if not history:
        return out
    for msg in history:
        role = msg.role if hasattr(msg, "role") else msg["role"]
        content = msg.content if hasattr(msg, "content") else msg["content"]
        if role == "human":
            out.append(HumanMessage(content=content))
        else:
            out.append(AIMessage(content=content))
    return out


@app.post("/v1/stream")
def stream(payload: StreamRequest):
    history = _to_lc_messages(payload.chatHistory)

    def gen():
        try:
            for token in retrieve_generate_stream(payload.question, history):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
