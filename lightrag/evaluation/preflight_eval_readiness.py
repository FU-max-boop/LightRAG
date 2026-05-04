#!/usr/bin/env python3
"""
Preflight checks for the LightRAG RAGAS evaluation workflow.

This utility performs static readiness checks before running
``eval_rag_quality.py``. It does not start LightRAG, call the LightRAG API,
or make LLM/embedding requests.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


REQUIRED_PACKAGES = {
    "datasets": "datasets",
    "ragas": "ragas",
    "langchain_openai": "langchain-openai",
    "httpx": "httpx",
    "dotenv": "python-dotenv",
    "tqdm": "tqdm",
}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


@dataclass
class PreflightReport:
    dataset: str
    documents_dir: str
    rag_endpoint: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(CheckResult(name=name, status=status, detail=detail))

    @property
    def blockers(self) -> list[CheckResult]:
        return [check for check in self.checks if check.status == "fail"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [check for check in self.checks if check.status == "warn"]

    @property
    def ready(self) -> bool:
        return not self.blockers

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "ready": self.ready,
            "dataset": self.dataset,
            "documents_dir": self.documents_dir,
            "rag_endpoint": self.rag_endpoint,
            "summary": {
                "passed": len([c for c in self.checks if c.status == "pass"]),
                "warnings": len(self.warnings),
                "blockers": len(self.blockers),
            },
            "checks": [check.__dict__ for check in self.checks],
        }

    def to_markdown(self) -> str:
        status = "ready" if self.ready else "blocked"
        lines = [
            "# LightRAG Evaluation Preflight Report",
            "",
            f"- Generated at: `{self.generated_at}`",
            f"- Status: `{status}`",
            f"- Dataset: `{self.dataset}`",
            f"- Documents directory: `{self.documents_dir}`",
            f"- LightRAG endpoint: `{self.rag_endpoint}`",
            "",
            "## Summary",
            "",
            f"- Passed checks: {len([c for c in self.checks if c.status == 'pass'])}",
            f"- Warnings: {len(self.warnings)}",
            f"- Blockers: {len(self.blockers)}",
            "",
            "## Checks",
            "",
            "| Status | Check | Detail |",
            "| --- | --- | --- |",
        ]
        for check in self.checks:
            detail = check.detail.replace("|", "\\|")
            lines.append(f"| `{check.status}` | {check.name} | {detail} |")
        lines.append("")
        return "\n".join(lines)

    def to_text(self) -> str:
        status = "READY" if self.ready else "BLOCKED"
        lines = [
            f"LightRAG evaluation preflight: {status}",
            f"Dataset: {self.dataset}",
            f"Documents directory: {self.documents_dir}",
            f"LightRAG endpoint: {self.rag_endpoint}",
            "",
        ]
        for check in self.checks:
            lines.append(f"[{check.status.upper()}] {check.name}: {check.detail}")
        return "\n".join(lines)


def resolve_path(path_text: str, base_dir: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_dataset(path: Path, report: PreflightReport) -> list[dict[str, Any]]:
    if not path.exists():
        report.add("dataset_exists", "fail", f"Dataset file does not exist: {path}")
        return []

    if not path.is_file():
        report.add("dataset_exists", "fail", f"Dataset path is not a file: {path}")
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.add("dataset_json", "fail", f"Invalid JSON: {exc}")
        return []

    if not isinstance(data, dict):
        report.add("dataset_schema", "fail", "Root JSON value must be an object")
        return []

    test_cases = data.get("test_cases")
    if not isinstance(test_cases, list):
        report.add("dataset_schema", "fail", "Expected a list at key `test_cases`")
        return []

    if not test_cases:
        report.add("dataset_cases", "fail", "`test_cases` is empty")
        return []

    report.add(
        "dataset_cases",
        "pass",
        f"Found {len(test_cases)} test case(s) in {path.name}",
    )
    return test_cases


def check_dataset_cases(test_cases: list[dict[str, Any]], report: PreflightReport) -> None:
    required_fields = ("question", "ground_truth")
    missing_fields: list[str] = []
    empty_fields: list[str] = []
    duplicate_questions: set[str] = set()
    seen_questions: set[str] = set()
    placeholder_cases: list[str] = []
    missing_project_count = 0

    for idx, test_case in enumerate(test_cases, 1):
        if not isinstance(test_case, dict):
            missing_fields.append(f"case {idx}: not an object")
            continue

        for field_name in required_fields:
            if field_name not in test_case:
                missing_fields.append(f"case {idx}: missing `{field_name}`")
            elif not str(test_case.get(field_name, "")).strip():
                empty_fields.append(f"case {idx}: empty `{field_name}`")

        question = str(test_case.get("question", "")).strip()
        if question:
            if question in seen_questions:
                duplicate_questions.add(question)
            seen_questions.add(question)

        combined_text = " ".join(
            str(test_case.get(field_name, "")) for field_name in required_fields
        ).lower()
        if "your question here" in combined_text or "expected answer" in combined_text:
            placeholder_cases.append(f"case {idx}")

        if not str(test_case.get("project", "")).strip():
            missing_project_count += 1

    if missing_fields:
        report.add("dataset_required_fields", "fail", "; ".join(missing_fields[:8]))
    else:
        report.add(
            "dataset_required_fields",
            "pass",
            "Every test case has `question` and `ground_truth` fields",
        )

    if empty_fields:
        report.add("dataset_empty_fields", "fail", "; ".join(empty_fields[:8]))
    else:
        report.add("dataset_empty_fields", "pass", "No required fields are empty")

    if duplicate_questions:
        report.add(
            "dataset_duplicate_questions",
            "warn",
            f"Found {len(duplicate_questions)} duplicate question(s)",
        )
    else:
        report.add("dataset_duplicate_questions", "pass", "No duplicate questions")

    if placeholder_cases:
        report.add(
            "dataset_placeholder_content",
            "warn",
            f"Template placeholder text appears in {', '.join(placeholder_cases[:8])}",
        )
    else:
        report.add("dataset_placeholder_content", "pass", "No template placeholders")

    if missing_project_count:
        report.add(
            "dataset_project_labels",
            "warn",
            f"{missing_project_count} case(s) do not set the optional `project` label",
        )
    else:
        report.add("dataset_project_labels", "pass", "All cases include project labels")


def check_documents_dir(path: Path, report: PreflightReport) -> None:
    if not path.exists():
        report.add(
            "documents_dir",
            "warn",
            f"Documents directory does not exist: {path}. This is okay for custom indexed data.",
        )
        return

    if not path.is_dir():
        report.add("documents_dir", "warn", f"Documents path is not a directory: {path}")
        return

    files = [entry for entry in path.iterdir() if entry.is_file()]
    if not files:
        report.add(
            "documents_dir",
            "warn",
            f"No files found in {path}. Make sure your LightRAG instance is indexed.",
        )
        return

    empty_files = [entry.name for entry in files if entry.stat().st_size == 0]
    if empty_files:
        report.add(
            "documents_non_empty",
            "warn",
            f"{len(empty_files)} empty file(s): {', '.join(empty_files[:8])}",
        )
    else:
        report.add(
            "documents_non_empty",
            "pass",
            f"Found {len(files)} non-empty document file(s)",
        )


def check_dependencies(report: PreflightReport) -> None:
    missing: list[str] = []
    available: list[str] = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
        else:
            available.append(package_name)

    if missing:
        report.add(
            "evaluation_dependencies",
            "fail",
            "Missing package(s): "
            + ", ".join(missing)
            + '. Install with `pip install -e ".[evaluation]"`.',
        )
    else:
        report.add(
            "evaluation_dependencies",
            "pass",
            "All required evaluation packages are importable: "
            + ", ".join(available),
        )


def check_environment(report: PreflightReport) -> None:
    llm_key = os.getenv("EVAL_LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    embedding_key = (
        os.getenv("EVAL_EMBEDDING_BINDING_API_KEY")
        or os.getenv("EVAL_LLM_BINDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    if llm_key:
        report.add(
            "llm_api_key",
            "pass",
            "LLM evaluation API key is configured via environment variables",
        )
    else:
        report.add(
            "llm_api_key",
            "fail",
            "Set EVAL_LLM_BINDING_API_KEY or OPENAI_API_KEY before running RAGAS",
        )

    if embedding_key:
        report.add(
            "embedding_api_key",
            "pass",
            "Embedding evaluation API key is configured via environment variables",
        )
    else:
        report.add(
            "embedding_api_key",
            "fail",
            "Set EVAL_EMBEDDING_BINDING_API_KEY, EVAL_LLM_BINDING_API_KEY, or OPENAI_API_KEY",
        )

    concurrency = os.getenv("EVAL_MAX_CONCURRENT", "2")
    query_top_k = os.getenv("EVAL_QUERY_TOP_K", "10")
    numeric_errors = []
    for name, value in (
        ("EVAL_MAX_CONCURRENT", concurrency),
        ("EVAL_QUERY_TOP_K", query_top_k),
    ):
        try:
            parsed = int(value)
            if parsed < 1:
                numeric_errors.append(f"{name} must be >= 1")
        except ValueError:
            numeric_errors.append(f"{name} must be an integer")

    if numeric_errors:
        report.add("evaluation_numeric_env", "fail", "; ".join(numeric_errors))
    else:
        report.add(
            "evaluation_numeric_env",
            "pass",
            f"EVAL_MAX_CONCURRENT={concurrency}, EVAL_QUERY_TOP_K={query_top_k}",
        )


def check_endpoint(endpoint: str, report: PreflightReport) -> None:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        report.add(
            "rag_endpoint_format",
            "fail",
            "Endpoint must be an absolute http(s) URL, for example http://localhost:9621",
        )
        return

    report.add(
        "rag_endpoint_format",
        "pass",
        "Endpoint has a valid http(s) URL format. Connectivity is not checked.",
    )


def write_outputs(report: PreflightReport, json_path: Path | None, md_path: Path | None) -> None:
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(report.to_markdown(), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run static readiness checks before LightRAG RAGAS evaluation."
    )
    parser.add_argument(
        "--dataset",
        default=str(SCRIPT_DIR / "sample_dataset.json"),
        help="Path to a RAGAS evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--documents-dir",
        default=str(SCRIPT_DIR / "sample_documents"),
        help="Directory containing the documents expected to be indexed for sample evaluation.",
    )
    parser.add_argument(
        "--ragendpoint",
        default=os.getenv("LIGHTRAG_API_URL", "http://localhost:9621"),
        help="LightRAG API endpoint URL to validate syntactically.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path for a machine-readable JSON report.",
    )
    parser.add_argument(
        "--output-md",
        help="Optional path for a Markdown report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when any blocking check fails.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    dataset_path = resolve_path(args.dataset)
    documents_dir = resolve_path(args.documents_dir)
    endpoint = args.ragendpoint

    report = PreflightReport(
        dataset=str(dataset_path),
        documents_dir=str(documents_dir),
        rag_endpoint=endpoint,
    )

    test_cases = load_dataset(dataset_path, report)
    if test_cases:
        check_dataset_cases(test_cases, report)
    check_documents_dir(documents_dir, report)
    check_dependencies(report)
    check_environment(report)
    check_endpoint(endpoint, report)

    write_outputs(
        report,
        resolve_path(args.output_json) if args.output_json else None,
        resolve_path(args.output_md) if args.output_md else None,
    )

    print(report.to_text())

    if args.strict and not report.ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
