import tempfile
import unittest
from pathlib import Path

from offline_retrieval_audit import (
    audit_retrieval,
    build_report,
    load_dataset,
    load_documents,
    load_oracle,
)


class OfflineRetrievalAuditTests(unittest.TestCase):
    def test_audit_retrieves_expected_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "alpha.md").write_text(
                "Alpha database supports vector search and filtering.",
                encoding="utf-8",
            )
            (docs_dir / "beta.md").write_text(
                "Beta explains deployment metrics and observability.",
                encoding="utf-8",
            )
            dataset = root / "dataset.json"
            dataset.write_text(
                '{"test_cases":[{"question":"Which document covers vector search?"}]}',
                encoding="utf-8",
            )
            oracle = root / "oracle.json"
            oracle.write_text(
                '{"oracle":[{"question":"Which document covers vector search?",'
                '"expected_documents":["alpha.md"]}]}',
                encoding="utf-8",
            )

            cases = load_dataset(dataset)
            oracle_map = load_oracle(oracle)
            documents = load_documents(docs_dir)
            audits = audit_retrieval(cases, oracle_map, documents, top_k=1)
            report = build_report(audits, dataset, docs_dir, oracle, top_k=1)

        self.assertEqual(report["summary"]["queries"], 1)
        self.assertEqual(report["summary"]["average_recall_at_k"], 1.0)
        self.assertEqual(audits[0].ranked_documents[0]["document"], "alpha.md")


if __name__ == "__main__":
    unittest.main()
