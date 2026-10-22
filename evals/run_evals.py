import argparse
import json

import backend as backend
from evals import evaluate_guardrails, evaluate_rag, evaluate_retrieval
from evals.common import corpus_retriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-limit", type=int, help="run RAGAS on only the first N questions (it makes many LLM calls)")
    args = parser.parse_args()

    llm, embeddings = backend.get_llm(), backend.get_embeddings()
    retriever = corpus_retriever(embeddings)
    report = {
        "retrieval": evaluate_retrieval.run(retriever),
        "guardrails": evaluate_guardrails.run(llm),
        "rag": evaluate_rag.run(retriever, llm, embeddings, args.rag_limit),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
