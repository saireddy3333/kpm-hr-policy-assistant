"""Evaluate retrieval and intent quality for the KPM HR Policy Assistant."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag import config
from rag.embeddings import ensure_index
from rag.generator import HRPolicyAssistant
from rag.intents import classify_intent
from rag.memory import ConversationMemory
from rag.retriever import PolicyRetriever


TEST_SET_PATH = Path(__file__).with_name("test_set.json")
RESULTS_PATH = Path(__file__).with_name("results.json")


def load_test_set() -> list[dict]:
    return json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))


def macro_f1(expected: list[str], predicted: list[str]) -> float:
    labels = sorted(set(expected) | set(predicted))
    scores = []
    for label in labels:
        tp = sum(1 for e, p in zip(expected, predicted) if e == label and p == label)
        fp = sum(1 for e, p in zip(expected, predicted) if e != label and p == label)
        fn = sum(1 for e, p in zip(expected, predicted) if e == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return mean(scores) if scores else 0.0


def evaluate() -> dict:
    ensure_index()
    test_set = load_test_set()
    retriever = PolicyRetriever()
    assistant = HRPolicyAssistant(retriever=retriever)

    hits_at_1 = []
    hits_at_3 = []
    expected_intents = []
    predicted_intents = []
    rows = []

    for item in test_set:
        question = item["question"]
        expected_sections = set(item["expected_sections"])
        intent = classify_intent(question)
        results = retriever.retrieve(question, intent=intent, top_k=5)
        retrieved_sections = [result.chunk["subsection"] for result in results]
        hit1 = bool(retrieved_sections[:1] and retrieved_sections[0] in expected_sections)
        hit3 = bool(expected_sections & set(retrieved_sections[:3]))
        answer = assistant.answer(question, name="Alex", memory=ConversationMemory())

        hits_at_1.append(hit1)
        hits_at_3.append(hit3)
        expected_intents.append(item["intent"])
        predicted_intents.append(intent.intent)
        rows.append(
            {
                "question": question,
                "expected_sections": sorted(expected_sections),
                "retrieved_sections": retrieved_sections,
                "hits_at_1": hit1,
                "hits_at_3": hit3,
                "expected_intent": item["intent"],
                "predicted_intent": intent.intent,
                "answer": answer.answer,
                "sources": answer.sources,
            }
        )

    intent_accuracy = sum(e == p for e, p in zip(expected_intents, predicted_intents)) / len(test_set)
    confusion = Counter(zip(expected_intents, predicted_intents))
    results = {
        "hits_at_1": sum(hits_at_1) / len(hits_at_1),
        "hits_at_3": sum(hits_at_3) / len(hits_at_3),
        "intent_accuracy": intent_accuracy,
        "intent_macro_f1": macro_f1(expected_intents, predicted_intents),
        "intent_confusion": {f"{expected} -> {predicted}": count for (expected, predicted), count in confusion.items()},
        "rows": rows,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(results)
    return results


def write_report(results: dict) -> None:
    lines = [
        "# TEST REPORT",
        "",
        "## Evaluation Results",
        f"- HITS@1: {results['hits_at_1']:.3f}",
        f"- HITS@3: {results['hits_at_3']:.3f}",
        f"- Intent accuracy: {results['intent_accuracy']:.3f}",
        f"- Intent macro F1: {results['intent_macro_f1']:.3f}",
        "",
        "## Retrieval Rows",
        "",
    ]
    for row in results["rows"]:
        lines.extend(
            [
                f"### {row['question']}",
                f"- Expected sections: {', '.join(row['expected_sections'])}",
                f"- Retrieved sections: {', '.join(row['retrieved_sections'])}",
                f"- HITS@1: {row['hits_at_1']}",
                f"- HITS@3: {row['hits_at_3']}",
                f"- Intent: expected {row['expected_intent']}, predicted {row['predicted_intent']}",
                "",
                "Answer:",
                "",
                row["answer"],
                "",
            ]
        )
    (config.PROJECT_ROOT / "TEST_REPORT.md").write_text(
        "\n".join(lines).strip() + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    metrics = evaluate()
    print(json.dumps({k: v for k, v in metrics.items() if k != "rows"}, indent=2, default=str))
