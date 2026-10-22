import json
import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
HEADERS = {"X-API-Key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}

st.set_page_config(page_title="Agentic RAG Chatbot")
state = st.session_state
state.setdefault("thread_id", str(uuid.uuid4()))
state.setdefault("messages", [])
state.setdefault("pending", None)  # approval request waiting for the user


def error_text(response):
    try:
        return response.json()["detail"]
    except (ValueError, KeyError):
        return response.text


def call(method, path, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{path}", headers=HEADERS, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Can't reach the API: {exc}")
        return None
    if not response.ok:
        st.error(error_text(response))
        return None
    return response.json()


def stream_reply(payload):
    """Yield text as it arrives; remember an approval request if the graph pauses for one."""
    try:
        with requests.post(f"{API_URL}/chat/stream", json=payload, headers=HEADERS, stream=True, timeout=120) as response:
            if not response.ok:
                yield f"Error: {error_text(response)}"
                return
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "token":
                    yield event["text"]
                elif event["type"] == "interrupt":
                    state.pending = event["data"]
                elif event["type"] == "error":
                    yield f"\n\nError: {event['detail']}"
    except requests.RequestException as exc:
        yield f"Error: lost connection to the API ({exc})"


def respond(payload):
    with st.chat_message("assistant"):
        text = st.write_stream(stream_reply(payload))
    if text:
        state.messages.append({"role": "ai", "text": text})
    st.rerun()


with st.sidebar:
    if st.button("New chat", use_container_width=True):
        state.update(thread_id=str(uuid.uuid4()), messages=[], pending=None)

    upload = st.file_uploader("Add a document", type=["pdf", "txt", "md"])
    if upload and st.button("Ingest", use_container_width=True):
        result = call("POST", "/documents/ingest", files={"file": (upload.name, upload.getvalue())})
        if result:
            st.success(f"Added {result['filename']} ({result['chunks']} chunks)")

    st.subheader("Conversations")
    for thread in call("GET", "/threads") or []:
        if st.button(thread["title"], key=thread["thread_id"], use_container_width=True):
            data = call("GET", f"/threads/{thread['thread_id']}")
            if data:
                state.update(thread_id=thread["thread_id"], messages=data["messages"], pending=data["interrupt"])

st.title("Agentic RAG Chatbot")

for message in state.messages:
    with st.chat_message("user" if message["role"] == "human" else "assistant"):
        st.markdown(message["text"])

if state.pending:
    st.info(f"The assistant wants to run **{state.pending['tool']}**")
    st.json(state.pending["args"])
    approve, reject = st.columns(2)
    choice = True if approve.button("Approve", use_container_width=True) else (
        False if reject.button("Reject", use_container_width=True) else None
    )
    if choice is not None:
        state.pending = None
        respond({"thread_id": state.thread_id, "approve": choice})

if prompt := st.chat_input("Ask about code, AI/ML, data or your documents", disabled=bool(state.pending)):
    state.messages.append({"role": "human", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    respond({"thread_id": state.thread_id, "message": prompt})
