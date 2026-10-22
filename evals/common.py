import json
import os
from pathlib import Path

from rag import ingestion
from rag.hybrid_retriever import HybridRetriever
from rag.ingestion import ingest

HERE = Path(__file__).parent
DATASET = HERE / "dataset.jsonl"
PDF = Path(os.getenv("EVAL_PDF", HERE / "data" / "agentic_ai_langgraph.pdf"))
INDEX_DIR = HERE / ".index"


def load(kind):
    rows = (json.loads(line) for line in DATASET.read_text().splitlines() if line.strip())
    return [row for row in rows if row["kind"] == kind]


def corpus_retriever(embeddings):
    """Index the eval PDF once. Later runs reuse evals/.index, so delete it to rebuild."""
    retriever = HybridRetriever(embeddings, INDEX_DIR)
    if not retriever.chunks:
        if not PDF.exists():
            raise SystemExit(f"Eval PDF not found. Put it at {PDF} or set EVAL_PDF.")
        data = PDF.read_bytes()
        ingestion.MAX_BYTES = max(ingestion.MAX_BYTES, len(data))  # this PDF is bigger than the upload limit
        ingest(retriever, PDF.name, data)
    return retriever
