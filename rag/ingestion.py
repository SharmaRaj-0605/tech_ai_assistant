import hashlib
import io
import os
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

ALLOWED_TYPES = {".pdf", ".txt", ".md"}
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "120")) * 1024 * 1024

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


class IngestError(ValueError):
    pass


def read_pages(name, data):
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_TYPES:
        raise IngestError(f"Unsupported file type '{suffix}'. Use PDF, TXT or MD.")
    if len(data) > MAX_BYTES:
        raise IngestError(f"File is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    if suffix != ".pdf":
        return [(1, data.decode("utf-8", errors="replace"))]
    try:
        reader = PdfReader(io.BytesIO(data))
        return [(n, page.extract_text() or "") for n, page in enumerate(reader.pages, 1)]
    except Exception as exc:
        raise IngestError("Couldn't read this PDF. It may be corrupted or encrypted.") from exc


def clean(text):
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_chunks(name, data):
    name = Path(name).name
    # the id comes from the filename, so re-uploading an edited file replaces the old chunks
    doc_id = hashlib.sha1(name.encode()).hexdigest()[:12]
    chunks = []
    for page, text in read_pages(name, data):
        for piece in splitter.split_text(clean(text)):
            meta = {"source": name, "page": page, "doc_id": doc_id, "chunk_id": f"{doc_id}:{len(chunks)}"}
            chunks.append(Document(page_content=piece, metadata=meta))
    if not chunks:
        raise IngestError("No text found. Scanned PDFs need OCR, which isn't supported yet.")
    return doc_id, chunks


def ingest(retriever, name, data):
    doc_id, chunks = build_chunks(name, data)
    retriever.add(doc_id, chunks)
    return {"doc_id": doc_id, "filename": Path(name).name, "chunks": len(chunks)}
