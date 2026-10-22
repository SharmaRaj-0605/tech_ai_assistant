from evals.common import load


def score(search, rows, k):
    """A chunk counts as relevant when it comes from a page that holds the answer."""
    recall = precision = mrr = 0
    for row in rows:
        relevant = set(row["pages"])
        ranked = [c.metadata["page"] for c in search(row["question"], k)]
        hits = [page in relevant for page in ranked]
        recall += len(relevant & set(ranked)) / len(relevant)
        precision += sum(hits) / k
        mrr += 1 / (hits.index(True) + 1) if True in hits else 0
    n = len(rows)
    return {f"recall@{k}": recall / n, f"precision@{k}": precision / n, "mrr": mrr / n}


def run(retriever, k=4):
    rows = load("qa")
    modes = {"vector": retriever.semantic, "keyword": retriever.keyword, "hybrid": retriever.search}
    return {name: score(search, rows, k) for name, search in modes.items()}
