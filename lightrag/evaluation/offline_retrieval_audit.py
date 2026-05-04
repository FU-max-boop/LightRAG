#!/usr/bin/env python3
"""
Offline retrieval sanity audit for LightRAG evaluation samples.

This script does not start LightRAG, call a model endpoint, compute embeddings,
or use RAGAS. It checks whether the sample evaluation questions have enough
lexical signal to retrieve their expected sample documents under a deterministic
BM25-style baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "lightrag" / "evaluation"
DEFAULT_DATASET = DEFAULT_EVAL_DIR / "sample_dataset.json"
DEFAULT_DOCS_DIR = DEFAULT_EVAL_DIR / "sample_documents"
DEFAULT_ORACLE = DEFAULT_EVAL_DIR / "sample_retrieval_oracle.json"
DEFAULT_OUTPUT_JSON = DEFAULT_EVAL_DIR / "results" / "offline_retrieval_audit.json"
DEFAULT_OUTPUT_MD = DEFAULT_EVAL_DIR / "results" / "offline_retrieval_audit.md"
DEFAULT_OUTPUT_CSV = DEFAULT_EVAL_DIR / "results" / "offline_retrieval_audit.csv"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "what",
    "with",
}


@dataclass
class Document:
    name: str
    path: str
    text: str
    tokens: list[str]
    term_counts: Counter[str]
    length: int


@dataclass
class QueryAudit:
    index: int
    question: str
    expected_documents: list[str]
    ranked_documents: list[dict[str, Any]]
    recall_at_k: float
    reciprocal_rank: float
    first_expected_rank: int | None

    @property
    def hit_at_k(self) -> bool:
        return self.recall_at_k > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "question": self.question,
            "expected_documents": self.expected_documents,
            "ranked_documents": self.ranked_documents,
            "recall_at_k": self.recall_at_k,
            "hit_at_k": self.hit_at_k,
            "reciprocal_rank": self.reciprocal_rank,
            "first_expected_rank": self.first_expected_rank,
        }


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def load_dataset(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("test_cases")
    if not isinstance(cases, list):
        raise ValueError(f"{path} must contain a test_cases list")
    return cases


def load_oracle(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("oracle")
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain an oracle list")

    oracle: dict[str, list[str]] = {}
    for entry in entries:
        question = str(entry.get("question", "")).strip()
        documents = entry.get("expected_documents", [])
        if not question or not isinstance(documents, list) or not documents:
            raise ValueError("Each oracle entry needs question and expected_documents")
        oracle[question] = [str(document) for document in documents]
    return oracle


def load_documents(docs_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        text = path.read_text(encoding="utf-8")
        tokens = tokenize(text)
        documents.append(
            Document(
                name=path.name,
                path=str(path),
                text=text,
                tokens=tokens,
                term_counts=Counter(tokens),
                length=len(tokens),
            )
        )
    if not documents:
        raise ValueError(f"No markdown sample documents found in {docs_dir}")
    return documents


def build_idf(documents: list[Document]) -> dict[str, float]:
    doc_count = len(documents)
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document.tokens))
    return {
        token: math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
        for token, df in document_frequency.items()
    }


def bm25_score(
    query_tokens: list[str],
    document: Document,
    idf: dict[str, float],
    average_doc_length: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not query_tokens or document.length == 0:
        return 0.0
    score = 0.0
    for token in query_tokens:
        frequency = document.term_counts.get(token, 0)
        if not frequency:
            continue
        denominator = frequency + k1 * (
            1 - b + b * (document.length / average_doc_length)
        )
        score += idf.get(token, 0.0) * (frequency * (k1 + 1) / denominator)
    return score


def audit_retrieval(
    cases: list[dict[str, Any]],
    oracle: dict[str, list[str]],
    documents: list[Document],
    top_k: int,
) -> list[QueryAudit]:
    idf = build_idf(documents)
    average_doc_length = sum(document.length for document in documents) / len(documents)
    audits: list[QueryAudit] = []

    for index, case in enumerate(cases, start=1):
        question = str(case.get("question", "")).strip()
        if question not in oracle:
            raise ValueError(f"No retrieval oracle entry found for question: {question}")

        query_tokens = tokenize(question)
        scored = [
            {
                "document": document.name,
                "score": bm25_score(query_tokens, document, idf, average_doc_length),
            }
            for document in documents
        ]
        ranked = sorted(scored, key=lambda item: (-item["score"], item["document"]))
        expected = oracle[question]
        top_documents = [item["document"] for item in ranked[:top_k]]
        matched = [document for document in expected if document in top_documents]
        first_expected_rank = None
        for rank, item in enumerate(ranked, start=1):
            if item["document"] in expected:
                first_expected_rank = rank
                break

        audits.append(
            QueryAudit(
                index=index,
                question=question,
                expected_documents=expected,
                ranked_documents=ranked,
                recall_at_k=len(matched) / len(expected),
                reciprocal_rank=1 / first_expected_rank if first_expected_rank else 0.0,
                first_expected_rank=first_expected_rank,
            )
        )
    return audits


def build_report(
    audits: list[QueryAudit],
    dataset: Path,
    docs_dir: Path,
    oracle: Path,
    top_k: int,
) -> dict[str, Any]:
    average_recall = sum(audit.recall_at_k for audit in audits) / len(audits)
    mean_reciprocal_rank = (
        sum(audit.reciprocal_rank for audit in audits) / len(audits)
    )
    full_recall_count = len([audit for audit in audits if audit.recall_at_k == 1.0])
    no_hit_count = len([audit for audit in audits if not audit.hit_at_k])
    multi_doc_cases = len([audit for audit in audits if len(audit.expected_documents) > 1])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "documents_dir": str(docs_dir),
        "oracle": str(oracle),
        "top_k": top_k,
        "summary": {
            "queries": len(audits),
            "average_recall_at_k": average_recall,
            "mean_reciprocal_rank": mean_reciprocal_rank,
            "full_recall_count": full_recall_count,
            "no_hit_count": no_hit_count,
            "multi_doc_cases": multi_doc_cases,
        },
        "queries": [audit.as_dict() for audit in audits],
    }


def report_to_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# LightRAG Offline Retrieval Audit",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Dataset: `{report['dataset']}`",
        f"- Documents: `{report['documents_dir']}`",
        f"- Oracle: `{report['oracle']}`",
        f"- Top-k: {report['top_k']}",
        f"- Queries: {summary['queries']}",
        f"- Average recall@k: {summary['average_recall_at_k']:.3f}",
        f"- Mean reciprocal rank: {summary['mean_reciprocal_rank']:.3f}",
        f"- Full-recall queries: {summary['full_recall_count']}/{summary['queries']}",
        f"- No-hit queries: {summary['no_hit_count']}",
        f"- Multi-document queries: {summary['multi_doc_cases']}",
        "",
        "## Query Results",
        "",
        "| # | Recall@k | MRR | Expected | Top Documents | Question |",
        "| ---: | ---: | ---: | --- | --- | --- |",
    ]
    for query in report["queries"]:
        top_documents = ", ".join(
            f"{item['document']} ({item['score']:.2f})"
            for item in query["ranked_documents"][: report["top_k"]]
        )
        expected = ", ".join(query["expected_documents"])
        question = query["question"].replace("|", "\\|")
        lines.append(
            f"| {query['index']} | {query['recall_at_k']:.3f} | "
            f"{query['reciprocal_rank']:.3f} | `{expected}` | "
            f"{top_documents} | {question} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(report: dict[str, Any], output_json: Path, output_md: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(report_to_markdown(report), encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "question",
                "expected_documents",
                "recall_at_k",
                "hit_at_k",
                "reciprocal_rank",
                "first_expected_rank",
                "top_documents",
            ],
        )
        writer.writeheader()
        for query in report["queries"]:
            writer.writerow(
                {
                    "index": query["index"],
                    "question": query["question"],
                    "expected_documents": ";".join(query["expected_documents"]),
                    "recall_at_k": query["recall_at_k"],
                    "hit_at_k": query["hit_at_k"],
                    "reciprocal_rank": query["reciprocal_rank"],
                    "first_expected_rank": query["first_expected_rank"],
                    "top_documents": ";".join(
                        item["document"] for item in query["ranked_documents"][: report["top_k"]]
                    ),
                }
            )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an offline retrieval sanity audit for LightRAG evaluation samples."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR))
    parser.add_argument("--oracle", default=str(DEFAULT_ORACLE))
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any query has zero recall@k.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.top_k <= 0:
        print("--top-k must be positive", file=sys.stderr)
        return 2

    dataset = Path(args.dataset).expanduser().resolve()
    docs_dir = Path(args.docs_dir).expanduser().resolve()
    oracle = Path(args.oracle).expanduser().resolve()
    try:
        cases = load_dataset(dataset)
        oracle_map = load_oracle(oracle)
        documents = load_documents(docs_dir)
        audits = audit_retrieval(cases, oracle_map, documents, args.top_k)
        report = build_report(audits, dataset, docs_dir, oracle, args.top_k)
        write_outputs(
            report,
            Path(args.output_json).expanduser(),
            Path(args.output_md).expanduser(),
            Path(args.output_csv).expanduser(),
        )
    except (OSError, ValueError) as exc:
        print(f"Offline retrieval audit failed: {exc}", file=sys.stderr)
        return 2

    summary = report["summary"]
    print("LightRAG offline retrieval audit")
    print(f"Queries: {summary['queries']}")
    print(f"Top-k: {report['top_k']}")
    print(f"Average recall@k: {summary['average_recall_at_k']:.3f}")
    print(f"Mean reciprocal rank: {summary['mean_reciprocal_rank']:.3f}")
    print(f"No-hit queries: {summary['no_hit_count']}")
    if args.strict and summary["no_hit_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
