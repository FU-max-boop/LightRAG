# LightRAG Evaluation Preflight

`preflight_eval_readiness.py` runs static checks before the RAGAS evaluator is
started. It does not call the LightRAG API, start a server, or make LLM and
embedding requests.

Use it when you want to catch common setup problems before spending time or API
budget on `eval_rag_quality.py`.

## Quick Start

From the project root:

```bash
python lightrag/evaluation/preflight_eval_readiness.py
```

Write reports for CI logs or debugging:

```bash
python lightrag/evaluation/preflight_eval_readiness.py \
  --output-json lightrag/evaluation/results/preflight_report.json \
  --output-md lightrag/evaluation/results/preflight_report.md
```

Fail the command when a blocking readiness issue is found:

```bash
python lightrag/evaluation/preflight_eval_readiness.py --strict
```

## What It Checks

- Dataset JSON exists and has a `test_cases` list.
- Each case has non-empty `question` and `ground_truth` fields.
- Duplicate questions and template placeholder text are surfaced as warnings.
- Sample document files are present and non-empty.
- RAGAS evaluation dependencies are importable.
- Required LLM and embedding API keys are configured.
- `EVAL_MAX_CONCURRENT`, `EVAL_QUERY_TOP_K`, and `--ragendpoint` are valid.

## Example Output

```text
LightRAG evaluation preflight: BLOCKED
Dataset: /path/to/LightRAG/lightrag/evaluation/sample_dataset.json
Documents directory: /path/to/LightRAG/lightrag/evaluation/sample_documents
LightRAG endpoint: http://localhost:9621

[PASS] dataset_cases: Found 6 test case(s) in sample_dataset.json
[PASS] dataset_required_fields: Every test case has `question` and `ground_truth` fields
[PASS] documents_non_empty: Found 6 non-empty document file(s)
[FAIL] evaluation_dependencies: Missing package(s): ragas, datasets. Install with `pip install -e ".[evaluation]"`.
[FAIL] llm_api_key: Set EVAL_LLM_BINDING_API_KEY or OPENAI_API_KEY before running RAGAS
```

## Relationship To The Evaluator

The preflight tool is intentionally conservative. A passing preflight means the
local configuration is structurally ready for `eval_rag_quality.py`; it does not
prove that the LightRAG server is running, that documents have already been
indexed, or that the remote model endpoints will accept requests.
