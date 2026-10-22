import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langsmith import traceable
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def rrf(rankings, k=60):
    """Merge ranked chunk lists. A chunk scores higher the closer to the top it sits in any list."""
    scores, chunks = {}, {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            cid = chunk.metadata["chunk_id"]
            chunks[cid] = chunk
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    return [chunks[cid] for cid in sorted(scores, key=scores.get, reverse=True)]


class HybridRetriever:
    """FAISS for meaning, BM25 for exact terms. The FAISS docstore is the single
    source of chunks, so both indexes always cover the same collection."""

    def __init__(self, embeddings, index_dir="faiss_db", reranker=None):
        self.embeddings = embeddings
        self.dir = Path(index_dir)
        self.reranker = reranker  # optional callable: (query, chunks) -> chunks
        self.store = None
        self.chunks = {}
        self.bm25 = None
        if (self.dir / "index.faiss").exists():
            # safe here: we only load index files this app wrote itself
            self.store = FAISS.load_local(str(self.dir), embeddings, allow_dangerous_deserialization=True)
            self._refresh()

    def add(self, doc_id, chunks):
        """Index a document's chunks, replacing anything stored earlier under the same doc_id."""
        self.remove(doc_id)
        ids = [c.metadata["chunk_id"] for c in chunks]
        if self.store is None:
            self.store = FAISS.from_documents(chunks, self.embeddings, ids=ids)
        else:
            self.store.add_documents(chunks, ids=ids)
        self.store.save_local(str(self.dir))
        self._refresh()

    def remove(self, doc_id):
        stale = [cid for cid, c in self.chunks.items() if c.metadata["doc_id"] == doc_id]
        if stale:
            self.store.delete(stale)
            self.store.save_local(str(self.dir))
            self._refresh()

    def _refresh(self):
        self.chunks = {i: self.store.docstore.search(i) for i in self.store.index_to_docstore_id.values()}
        corpus = [tokenize(c.page_content) for c in self.chunks.values()]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def semantic(self, query, k):
        return self.store.similarity_search(query, k=k) if self.store else []

    def keyword(self, query, k):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ids = list(self.chunks)
        best = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[ids[i]] for i in best if scores[i] > 0]

    @traceable(name="hybrid_search", run_type="retriever")
    def search(self, query, k=4, pool=20):
        fused = rrf([self.semantic(query, pool), self.keyword(query, pool)])
        if self.reranker:
            fused = self.reranker(query, fused)
        return fused[:k]
