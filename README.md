# Agentic RAG Chatbot

A technical-focused chatbot built around LangGraph and Retrieval-Augmented Generation (RAG). The target system combines document ingestion, hybrid search, source-grounded answers, human-in-the-loop (HITL) approvals, persistent conversations, a Streamlit interface, a FastAPI service, PostgreSQL checkpoints, LangSmith observability, and automated evaluations.

> **Project status:** Code for the API, guardrails, ingestion, hybrid retrieval, HITL graph, Streamlit UI, tests and evals is written, but none of it has been run yet (the build sandbox couldn't install dependencies). Run `pytest` first and fix whatever breaks. The Docker files are also untested. Not done: OCR, rate limiting, reranker implementation (the hook exists), and LangSmith dataset uploads. No evaluation results exist yet.

## Table of Contents

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Functionality](#functionality)
- [RAG and Ingestion Pipeline](#rag-and-ingestion-pipeline)
- [Architecture](#architecture)
- [Evaluation and Testing](#evaluation-and-testing)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Local Development](#local-development)
- [Security and Reliability](#security-and-reliability)
- [Implementation Notes](#implementation-notes)

## Overview

The application provides a conversational interface for technical questions and questions about uploaded documents. LangGraph coordinates the assistant's workflow and tools. The RAG pipeline retrieves relevant evidence from the document collection, while guardrails enforce the application's technical-use policy. PostgreSQL stores LangGraph checkpoint state by conversation thread, and LangSmith helps inspect workflow execution and evaluation results.

The project is designed to preserve the existing `app_hitl.py` Streamlit UI and `agentic_chatbot_hitl_backend.py` workflow wherever practical, adding focused modules for new features rather than keeping multiple alternative tutorial apps.

## Technology Stack

| Layer | Technology | Responsibility |
|---|---|---|
| Frontend | **Streamlit** | Chat UI, streaming display, thread sidebar, document upload, and HITL controls |
| API layer | **FastAPI** + **Uvicorn** | HTTP endpoints for chat, streaming, thread history, health checks, and document ingestion |
| Agent orchestration | **LangGraph** | Stateful graph, model/tool routing, checkpointing, and interrupt/resume flow for HITL |
| LLM integration | **LangChain model integrations** | Connect the graph to a supported chat model through environment configuration |
| Document loading | **LangChain document loaders**, **pypdf** | Extract text and metadata from supported documents, starting with PDFs and text files |
| Text splitting | **RecursiveCharacterTextSplitter** | Divide text into overlapping chunks while preserving useful source metadata |
| Embeddings | **Configured embedding model** | Convert document chunks and queries into vectors |
| Vector search | **FAISS** | Semantic similarity retrieval |
| Keyword search | **BM25** (for example, `rank-bm25`) | Retrieve exact terms, identifiers, keywords, and lexical matches |
| Hybrid fusion | **Reciprocal Rank Fusion (RRF)** | Combine semantic and keyword rankings into one candidate list |
| Reranking | Optional cross-encoder or provider-backed reranker | Reorder retrieved candidates by query relevance when configured |
| Persistent workflow state | **PostgreSQL** + **LangGraph PostgreSQL checkpointer** | Save graph checkpoints and conversation state associated with thread IDs |
| Index storage | **FAISS index files** and document metadata | Persist vector index data; maintain a matching BM25 corpus/index |
| Observability | **LangSmith** | Trace graph runs, model calls, tools, retrieval stages, errors, and latency; organize experiments |
| Evaluation | **Python evaluation scripts**, **LangSmith datasets/experiments** | Evaluate retrieval, answer quality, guardrails, and regressions |
| Testing | **pytest** | Unit and integration tests for guardrails, retrieval, API endpoints, ingestion, and persistence |
| Configuration | Environment variables and `.env` for local development | Configure API keys, model names, database connection, and tracing |
| Packaging | **Docker** | Reproducible local setup and deployment packaging |
| Version control | **Git / GitHub** | Source control and project collaboration |

### Model and provider choice

The LLM and embedding providers should be configurable rather than hard-coded. Choose provider/model IDs that are available to your account and verify current pricing, quotas, and rate limits. Do not assume a provider's free tier will remain available indefinitely.

## Functionality

### 1. Technical-topic guardrails

- Classify or validate incoming requests against a documented technical-use policy.
- Reject clearly unrelated requests with a short, consistent explanation.
- Allow legitimate software engineering, programming, AI/ML, data, systems, and other configured technical questions.
- Include test cases for ambiguous requests, prompt injection, and attempts to bypass the policy.
- Keep topic filtering separate from RAG grounding: a question can be technical but still have no answer in the uploaded documents.

Guardrails reduce unwanted behavior but cannot guarantee that every malicious or ambiguous prompt will be handled perfectly. The application should still treat user input and retrieved document content as untrusted.

### 2. Document ingestion

- Accept supported file uploads through Streamlit and/or the API.
- Validate file type and size before processing.
- Extract text and retain source metadata such as document name and page number.
- Clean extracted text where appropriate.
- Split text into overlapping chunks and assign stable identifiers.
- Generate embeddings and add chunks to the FAISS index.
- Add the same chunks to the BM25 corpus so both retrieval methods search the same document collection.
- Return ingestion status and useful error messages.
- Support index rebuilds and re-ingestion without duplicate chunks where stable document identifiers are available.

**Scanned PDFs:** ordinary text extraction does not reliably read image-only pages. OCR should be added if scanned documents need to be supported.

### 3. Hybrid RAG retrieval

1. Embed the user query and run semantic search against FAISS.
2. Run BM25 keyword search over the indexed chunks.
3. Merge the ranked lists using Reciprocal Rank Fusion (RRF).
4. Deduplicate overlapping results and retain source metadata.
5. Optionally rerank candidates using a configured reranker.
6. Pass the selected context to the LLM to generate a grounded answer.
7. Return available citations, including document names and page numbers.
8. For document-specific questions with insufficient evidence, say that the available documents do not establish an answer instead of inventing one.

### 4. Chat, streaming, and conversation history

- Provide a Streamlit chat interface.
- Stream generated responses where supported by the model and API implementation.
- Assign a thread ID to each conversation.
- List and reopen previous threads.
- Resume graph state from PostgreSQL checkpoints.
- Keep conversation state separate from document storage: PostgreSQL checkpoints do not automatically replace FAISS or BM25 indexes.

### 5. Human-in-the-loop (HITL)

- Pause selected tool actions for approval before execution or continuation.
- Allow the user to approve or reject the interrupted action.
- Resume the LangGraph run using the persisted thread/checkpoint state.
- Keep approval requirements explicit for each action that can cause a consequential side effect.

HITL is not automatically applied to every tool. Each protected action must be wired into the graph's interrupt/resume workflow and tested.

### 6. FastAPI service

Planned endpoints include:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Submit a message and receive a response |
| `POST` | `/chat/stream` | Stream response events |
| `GET` | `/threads` | List conversation threads available to the current user/session |
| `GET` | `/threads/{thread_id}` | Retrieve available history or state for a thread |
| `POST` | `/documents/ingest` | Upload and ingest a document |

Exact schemas, authentication, and authorization must be defined in the implementation. Do not expose a production API publicly without appropriate access controls and request limits.

### 7. PostgreSQL persistence

- Use PostgreSQL with LangGraph's supported PostgreSQL checkpointer for graph state.
- Associate saved state with stable thread IDs to support conversation resumption.
- Initialize checkpoint tables using the installed checkpointer's supported setup process.
- Use connection pooling and appropriate database credentials for deployment.

This scope covers persistent workflow/conversation checkpoints. A separate relational schema may be needed for users, document metadata, ingestion jobs, or evaluation results. PostgreSQL checkpointing alone does not provide those features automatically.

### 8. LangSmith observability

- Trace LangGraph executions and supported LLM/tool calls.
- Inspect execution paths, errors, and latency.
- Add explicit tracing around custom retrieval, fusion, reranking, and guardrail functions where automatic instrumentation is insufficient.
- Attach useful run metadata, such as project name and thread identifier, without logging secrets.
- Use traces to investigate failed runs and poor-quality answers.

LangSmith tracing is observability, not proof of correctness. Evaluation datasets and test assertions are needed to measure quality. Traces can contain prompts, retrieved text, and model outputs, so configure data capture deliberately.

## RAG and Ingestion Pipeline

```text
Document upload
    |
    v
Validate file -> Extract text -> Clean and chunk
                                  |
                                  v
                             Create embeddings
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
               FAISS index                 BM25 corpus/index
                    |                           |
                    +-------------+-------------+
                                  |
User question -> Guardrails -> Semantic search + Keyword search
                                  |
                                  v
                         Reciprocal Rank Fusion
                                  |
                           Optional reranker
                                  |
                                  v
                       Chunks + source metadata
                                  |
                                  v
                         LLM answer generation
                                  |
                                  v
                       Answer with citations, when available
                                  |
                                  v
                       LangSmith traces and evaluations
```

## Architecture

```text
                   +----------------------+
                   | Streamlit UI         |
                   +----------+-----------+
                              |
                   +----------v-----------+
                   | FastAPI service      |
                   +----------+-----------+
                              |
                   +----------v-----------+
                   | Input guardrails     |
                   +----------+-----------+
                              |
                   +----------v-----------+
                   | LangGraph workflow   |
                   | - model calls        |
                   | - tool routing       |
                   | - RAG                |
                   | - HITL interrupts    |
                   +------+----------+----+
                          |          |
                 +--------v----+  +--v----------------+
                 | Hybrid RAG  |  | PostgreSQL        |
                 | FAISS+BM25  |  | graph checkpoints |
                 +-------------+  +-------------------+
                          |
                 +--------v-----------+
                 | LangSmith          |
                 | traces/evaluations |
                 +--------------------+
```

## Evaluation and Testing

The evaluation suite should use a curated, versioned dataset. Do not publish accuracy or performance claims until the tests have been run and results recorded.

### Retrieval metrics

- **Recall@K:** fraction of known relevant chunks found in the top K results.
- **Precision@K:** fraction of top-K results that are relevant.
- **MRR (Mean Reciprocal Rank):** rewards ranking the first relevant result highly.
- Compare vector-only, keyword-only, and hybrid retrieval on the same dataset to check whether fusion actually helps.

### Answer-quality checks

`evals/evaluate_rag.py` scores answers with RAGAS (Faithfulness, ResponseRelevancy, LLMContextRecall, FactualCorrectness) and adds two plain checks RAGAS doesn't cover: citation validity and insufficient-evidence behavior. It uses RAGAS's LangChain wrappers and `evaluate()`, which RAGAS 0.4 marks as deprecated but still supports, so `ragas` is pinned below 1.0.

- **Correctness:** compare answers with reference answers or reviewed expected facts.
- **Faithfulness / groundedness:** check whether answer claims are supported by retrieved context.
- **Relevance:** check whether the answer addresses the question.
- **Citation validity:** confirm cited document/page references exist and support the associated claims.
- **Insufficient-evidence behavior:** verify that the system avoids fabricating document-based answers when retrieval fails.

### Guardrail and workflow tests

- Legitimate technical requests should be allowed.
- Clearly off-topic requests should be rejected according to policy.
- Test ambiguous requests and prompt-injection/bypass attempts.
- Test HITL pause, approval, rejection, and resume paths.
- Test thread persistence across application restarts.
- Test empty retrieval results, malformed files, API errors, and streaming interruptions.

### Suggested evaluation files

```text
evals/
├── dataset.jsonl
├── evaluate_retrieval.py
├── evaluate_rag.py
├── evaluate_guardrails.py
└── run_evals.py
```

The golden dataset is built from the "Agentic AI using LangGraph" notes PDF. Put the PDF at `evals/data/agentic_ai_langgraph.pdf` (or set `EVAL_PDF`). The first run indexes it into `evals/.index/` and later runs reuse that index. Each `qa` row has a reference answer and the PDF pages that hold it, and retrieval counts a chunk as relevant when it comes from one of those pages. `python -m evals.run_evals --rag-limit 10` runs RAGAS on only the first 10 questions, which is useful on free-tier rate limits. The PDF is larger than the default 20 MB upload limit, so set `MAX_UPLOAD_MB=120` in `.env` if you also want to upload it through the app.

Use LangSmith datasets/experiments to compare runs and inspect failures. LLM-as-judge scores may help evaluate relevance or groundedness, but should be cross-checked against human-reviewed examples and deterministic tests where possible.

## Project Structure

The following is the intended cleaned-up structure. Preserve the existing UI and backend files where practical. Remove redundant tutorial variants only after checking imports and backing up local data.

```text
Agentic-Chatbot-using-LangGraph/
├── app.py                       # Existing Streamlit UI and HITL flow
├── backend.py   # Existing LangGraph workflow; updated as needed
├── api.py                            # FastAPI application
├── guardrails/
│   ├── __init__.py
│   └── checks.py                     # Technical-topic policy and validation
├── rag/
│   ├── __init__.py
│   ├── hybrid_retriever.py           # FAISS + BM25 + RRF retrieval
│   └── ingestion.py                  # Validation, text extraction, chunking
├── requirements.txt                  # Project dependencies
├── .env.example                      # Example configuration without secrets
├── .gitignore
├── README.md
├── faiss_db/                         # Persisted vector index, if retained
├── evals/
    ├── common.py
    ├── dataset.jsonl
    ├── evaluate_retrieval.py
    ├── evaluate_rag.py
    ├── evaluate_guardrails.py
    └── run_evals.py

```

Additional modules may be appropriate for ingestion, document metadata, database helpers, or schemas as the implementation grows. Keep modules focused; do not duplicate the same application in multiple files.

## Configuration

Create a local `.env` file based on `.env.example`. Add only the settings required by the chosen providers and actual code. Example keys:

```env
# Configure the model and embedding providers selected by the project.
# Add their required API keys and model identifiers here.

DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DATABASE

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=YOUR_LANGSMITH_API_KEY
LANGSMITH_PROJECT=agentic-rag-chatbot
```

The implementation must use a consistent database configuration variable and document any provider-specific model or embedding variables. Never commit real API keys, passwords, or production database URLs. Use environment-based secrets or a managed secret store in deployment.

## Local Development

These are the intended local-development steps; confirm module names and command options against the completed implementation.

1. Install a supported Python version and create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Configure model, embedding, PostgreSQL, and LangSmith settings in `.env`.
4. Start PostgreSQL and create the configured database.
5. Initialize the LangGraph PostgreSQL checkpoint schema with the supported checkpointer setup procedure.
6. Start the API service.
7. Start the Streamlit interface.
8. Upload a test document and verify ingestion, hybrid retrieval, source metadata, and grounded answers.
9. Run the evaluation suite and inspect LangSmith traces.

Typical commands, once the corresponding files and entry points exist:

```bash
pip install -r requirements.txt
uvicorn api:app --reload
streamlit run app_hitl.py
python -m evals.run_evals
```

To run everything in Docker (API, Streamlit and PostgreSQL), fill in `.env` including `POSTGRES_PASSWORD`, then run `docker compose up --build`. The API is on port 8000 and the UI on 8501. Compose builds `DATABASE_URL` itself and keeps the database, the FAISS index and saved notes in named volumes (`pgdata`, `faiss_data`, `notes_data`), so they survive restarts.

## Security and Reliability

- Keep credentials out of source code and Git history.
- Validate upload types and sizes; consider malware scanning for untrusted uploads.
- Treat uploaded files and retrieved content as untrusted input.
- Use least-privilege PostgreSQL credentials and back up persistent state.
- Authenticate and authorize public API access; add rate limits where appropriate.
- Do not allow unrestricted tool execution through public endpoints.
- Require HITL approval for actions with meaningful side effects.
- Avoid logging secrets and unnecessary sensitive text to LangSmith.
- Restrict FAISS deserialization to index files from trusted sources.
- Pin dependencies and run automated tests before releases.
- Test recovery, re-ingestion, deletion, and index consistency.

## Implementation Notes

- The current tutorial's combined HITL files provide a starting point, not a guarantee that every feature above is already available.
- Reuse the existing `app_hitl.py` and `agentic_chatbot_hitl_backend.py` wherever practical, making only the changes needed to connect new modules.
- Hybrid retrieval requires the vector and keyword indexes to be built from the same version of the chunk collection.
- PostgreSQL checkpoints persist graph state; they do not, by themselves, implement user management, a document registry, or vector storage.
- LangSmith traces show what happened during a run; the evaluation suite measures how well it performed.
- Label project status and benchmark results accurately. Do not claim production readiness until the implementation and tests support that claim.

## License

Add a license file if you intend to distribute the project publicly. Respect the licenses of the original repository and all third-party dependencies.
