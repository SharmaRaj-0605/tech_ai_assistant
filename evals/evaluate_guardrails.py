from evals.common import load
from guardrails import check_message


def run(llm):
    rows = load("guardrail")
    wrong = [r["text"] for r in rows if check_message(r["text"], llm).allowed != (r["expected"] == "allow")]
    return {"accuracy": 1 - len(wrong) / len(rows), "wrong": wrong}
