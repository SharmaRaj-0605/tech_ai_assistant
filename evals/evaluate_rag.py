import re
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall, ResponseRelevancy
from ragas.run_config import RunConfig

import backend as backend
from evals.common import load


def ask(graph, question):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    messages = graph.invoke({"messages": [("human", question)]}, config)["messages"]
    contexts = []
    for m in messages:
        if m.type == "tool" and m.name == "search_documents":
            # the tool joins hits with a blank line, and each hit starts with "[file, p.N]"
            contexts += re.split(r"\n\n(?=\[[^\]]+, p\.\d+\]\n)", str(m.text))
    return str(messages[-1].text), [c for c in contexts if c.startswith("[")]


def citations_exist(answer, contexts):
    cited = re.findall(r"\(([^,()]+), p\.(\d+)\)", answer)
    return all(any(c.startswith(f"[{name}, p.{page}]") for c in contexts) for name, page in cited)


def admits_missing_evidence(llm, answer):
    prompt = f"Answer:\n{answer}\n\nDoes this answer say the documents don't contain the information, rather than stating a fact? Reply YES or NO."
    return llm.invoke([("human", prompt)]).text.strip().upper().startswith("YES")


def run(retriever, llm, embeddings, limit=None):
    backend.get_retriever = lambda: retriever  # point the search tool at the eval corpus
    graph = backend.build_graph(InMemorySaver())

    samples, cited_ok = [], []
    for row in load("qa")[:limit]:
        answer, contexts = ask(graph, row["question"])
        samples.append({
            "user_input": row["question"],
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": row["reference"],
        })
        cited_ok.append(citations_exist(answer, contexts))

    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextRecall(), FactualCorrectness()]
    result = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=metrics,
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        run_config=RunConfig(max_workers=4),  # keeps free-tier rate limits happy
    )

    missing = [admits_missing_evidence(llm, ask(graph, row["question"])[0]) for row in load("unanswerable")]
    return {
        **{m.name: float(result[m.name]) for m in metrics},
        "citations_valid": sum(cited_ok) / len(cited_ok),
        "admits_missing_evidence": sum(missing) / len(missing),
    }
