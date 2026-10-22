import hmac
import json
import logging
import os

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field, model_validator

import backend as backend
from guardrails import check_message
from rag.ingestion import MAX_BYTES, IngestError, ingest

log = logging.getLogger(__name__)


def require_key(x_api_key: str | None = Header(None)):
    # auth is switched on by setting API_KEY; leave it unset for local development
    key = os.getenv("API_KEY")
    if key and not hmac.compare_digest(x_api_key or "", key):
        raise HTTPException(401, "Invalid or missing API key.")


app = FastAPI(title="Agentic RAG Chatbot")
secured = APIRouter(dependencies=[Depends(require_key)])


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    message: str | None = Field(None, min_length=1, max_length=4000)
    approve: bool | None = None  # answer to a pending approval request

    @model_validator(mode="after")
    def message_or_approval(self):
        if (self.message is None) == (self.approve is None):
            raise ValueError("Send either 'message' or 'approve', not both.")
        return self


def thread_config(thread_id):
    return {"configurable": {"thread_id": thread_id}, "metadata": {"thread_id": thread_id}}


def pending(state):
    return [i.value for task in state.tasks for i in task.interrupts]


def prepare(req, graph, config):
    """Turn a request into graph input. Returns (input, refusal_text)."""
    waiting = pending(graph.get_state(config))
    if req.approve is not None:
        if not waiting:
            raise HTTPException(409, "Nothing is waiting for approval on this thread.")
        return Command(resume=req.approve), None
    if waiting:
        raise HTTPException(409, "Approve or reject the pending action first.")
    verdict = check_message(req.message, backend.get_llm())
    if not verdict.allowed:
        return None, verdict.reason
    return {"messages": [HumanMessage(req.message)]}, None


@app.get("/health")
def health():
    return {"status": "ok"}


@secured.post("/chat")
def chat(req: ChatRequest):
    graph, config = backend.get_graph(), thread_config(req.thread_id)
    graph_input, refusal = prepare(req, graph, config)
    if refusal:
        return {"reply": refusal, "interrupt": None}
    result = graph.invoke(graph_input, config)
    waiting = pending(graph.get_state(config))
    return {"reply": str(result["messages"][-1].text), "interrupt": waiting[0] if waiting else None}


@secured.post("/chat/stream")
def chat_stream(req: ChatRequest):
    graph, config = backend.get_graph(), thread_config(req.thread_id)
    graph_input, refusal = prepare(req, graph, config)

    def event(**data):
        return json.dumps(data) + "\n"

    def events():
        if refusal:
            yield event(type="token", text=refusal)
        else:
            try:
                for chunk, meta in graph.stream(graph_input, config, stream_mode="messages"):
                    if meta["langgraph_node"] == "chat" and chunk.text:
                        yield event(type="token", text=str(chunk.text))
                for request in pending(graph.get_state(config)):
                    yield event(type="interrupt", data=request)
            except Exception:
                log.exception("stream failed")
                yield event(type="error", detail="The response was interrupted. Please try again.")
        yield event(type="done")

    return StreamingResponse(events(), media_type="application/x-ndjson")


@secured.get("/threads")
def list_threads():
    titles = {}
    for item in backend.get_graph().checkpointer.list(None):  # newest checkpoint first
        tid = item.config["configurable"]["thread_id"]
        if tid not in titles:
            first = next((m for m in item.checkpoint["channel_values"].get("messages", []) if m.type == "human"), None)
            titles[tid] = str(first.text)[:50] if first else "New chat"
    return [{"thread_id": tid, "title": title} for tid, title in titles.items()]


@secured.get("/threads/{thread_id}")
def get_thread(thread_id: str):
    state = backend.get_graph().get_state(thread_config(thread_id))
    if not state.values:
        raise HTTPException(404, "Thread not found.")
    waiting = pending(state)
    messages = [{"role": m.type, "text": str(m.text)} for m in state.values["messages"] if m.type in ("human", "ai") and m.text]
    return {"thread_id": thread_id, "messages": messages, "interrupt": waiting[0] if waiting else None}


@secured.post("/documents/ingest")
def ingest_document(file: UploadFile):
    data = file.file.read(MAX_BYTES + 1)
    try:
        return ingest(backend.get_retriever(), file.filename or "", data)
    except IngestError as exc:
        raise HTTPException(400, str(exc))


app.include_router(secured)
