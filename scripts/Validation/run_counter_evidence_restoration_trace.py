import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness


EXP2_RESULTS_PATH = Path(
    "benchmark_results/Exp2/formal/experiment_2_generation.jsonl"
)
EXTERNAL_RESULTS_PATH = Path(
    "benchmark_results/external_metric_validation/"
    "external_metric_validation_e2.jsonl"
)
AVAILABLE_CORPUS = Path("benchmarks/Exp1/corpora/formal_v1/available")

OUTPUT_ROOT = Path("benchmark_results/counter_evidence_restoration_trace")
CHECKPOINT_PATH = OUTPUT_ROOT / "restoration_trace_checkpoint.jsonl"
DETAIL_PATH = OUTPUT_ROOT / "restoration_trace_e2.jsonl"
SUMMARY_PATH = OUTPUT_ROOT / "restoration_trace_summary.json"

EVALUATOR_MODEL = "openai/gpt-oss-120b"
MAX_OUTPUT_TOKENS = 4096
INTER_CASE_SECONDS = 12


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_response(query: str, verdict: str) -> str:
    return f"For the factual question '{query}', the answer is {verdict}."


def serialize(value) -> dict | list | str:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return str(value)


def fingerprint(value) -> str:
    prompt_type = type(value)
    payload = (
        f"{prompt_type.__module__}.{prompt_type.__qualname__}\n"
        f"{inspect.getsource(prompt_type)}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exposed_contexts(row: dict) -> list[str]:
    if row["condition"] != "exposed" or row["corpus_state"] != "available":
        raise RuntimeError("Trace context must come from the matched EXPOSED row.")
    texts: list[str] = []
    for filename in row["context_source_files"]:
        path = AVAILABLE_CORPUS / filename
        if not path.exists():
            raise FileNotFoundError(f"Frozen context file missing: {path}")
        texts.append(path.read_text(encoding="utf-8").strip())
    return texts


def paired_rows(rows: list[dict]) -> list[tuple[dict, dict]]:
    by_key = {(row["condition"], row["claim_id"]): row for row in rows}
    claim_ids = sorted(
        row["claim_id"]
        for row in rows
        if row["condition"] == "available_not_exposed"
    )
    if len(claim_ids) != 6:
        raise RuntimeError(f"Expected six hidden rows, got {len(claim_ids)}.")

    pairs: list[tuple[dict, dict]] = []
    for claim_id in claim_ids:
        hidden = by_key[("available_not_exposed", claim_id)]
        exposed = by_key[("exposed", claim_id)]
        if hidden["query"] != exposed["query"]:
            raise RuntimeError(f"Query mismatch for {claim_id}.")
        if hidden["correct"] is not False:
            raise RuntimeError(f"Frozen hidden answer is not wrong for {claim_id}.")
        pairs.append((hidden, exposed))
    return pairs


async def trace_faithfulness(
    metric: Faithfulness,
    *,
    query: str,
    response: str,
    contexts: list[str],
) -> tuple[float, list[str], list[dict]]:
    # RAGAS 0.4.3 exposes the two prompt stages through these internal
    # Faithfulness methods; the public ascore() intentionally returns only
    # the final scalar MetricResult.
    statements = list(await metric._create_statements(query, response))
    if not statements:
        raise RuntimeError("RAGAS generated no statements.")

    verdict_output = await metric._create_verdicts(
        statements,
        "\n".join(contexts),
    )
    verdicts = [serialize(item) for item in verdict_output.statements]
    score = float(metric._compute_score(verdict_output))
    return score, statements, verdicts


async def main_async() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    exp2_rows = load_jsonl(EXP2_RESULTS_PATH)
    if len(exp2_rows) != 18:
        raise RuntimeError(f"Expected 18 frozen E2 rows, got {len(exp2_rows)}.")
    pairs = paired_rows(exp2_rows)

    external_rows = load_jsonl(EXTERNAL_RESULTS_PATH)
    baseline_rows = {
        row["claim_id"]: row
        for row in external_rows
        if row["condition"] == "available_not_exposed"
    }
    if len(baseline_rows) != 6:
        raise RuntimeError(f"Expected six baseline rows, got {len(baseline_rows)}.")
    if any(float(row["ragas_faithfulness"]) != 1.0 for row in baseline_rows.values()):
        raise RuntimeError("Not every frozen baseline faithfulness score is 1.0.")
    if any(row.get("evaluator_model") != EVALUATOR_MODEL for row in baseline_rows.values()):
        raise RuntimeError("Baseline evaluator model does not match the trace model.")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    evaluator_llm = llm_factory(
        EVALUATOR_MODEL,
        provider="openai",
        client=client,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    metric = Faithfulness(llm=evaluator_llm)

    ragas_version = importlib.metadata.version("ragas")
    statement_prompt_hash = fingerprint(metric.statement_generator_prompt)
    nli_prompt_hash = fingerprint(metric.nli_statement_prompt)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict] = {}
    if CHECKPOINT_PATH.exists():
        completed = {
            row["claim_id"]: row for row in load_jsonl(CHECKPOINT_PATH)
        }
        print(f"Resuming from checkpoint: {len(completed)} completed cases")

    print()
    print("COUNTER-EVIDENCE RESTORATION — STATEMENT TRACE")
    print("=" * 92)
    print(f"Evaluator model: {EVALUATOR_MODEL}")
    print(f"RAGAS version:   {ragas_version}")
    print(f"Statement prompt SHA-256: {statement_prompt_hash}")
    print(f"NLI prompt SHA-256:       {nli_prompt_hash}")
    print()

    for hidden, exposed in pairs:
        claim_id = hidden["claim_id"]
        if claim_id in completed:
            saved = completed[claim_id]
            print(
                f"{claim_id:<10} score={saved['restored_faithfulness']:.3f} "
                f"statements={saved['statement_count']} [checkpoint]"
            )
            continue

        query = hidden["query"]
        response = render_response(query, hidden["verdict"])
        contexts = exposed_contexts(exposed)
        score, statements, verdicts = await trace_faithfulness(
            metric,
            query=query,
            response=response,
            contexts=contexts,
        )

        result = {
            "analysis": "counter_evidence_restoration_statement_trace",
            "formal_experiment": False,
            "analysis_of_frozen_formal_outputs": True,
            "source_experiment": "E2",
            "claim_id": claim_id,
            "truth_label": bool(hidden["truth_label"]),
            "query": query,
            "frozen_generator_verdict": hidden["verdict"],
            "frozen_answer_correct": bool(hidden["correct"]),
            "rendered_response_for_ragas": response,
            "baseline_faithfulness": 1.0,
            "restored_faithfulness": score,
            "statement_count": len(statements),
            "generated_statements": statements,
            "statement_verdicts": verdicts,
            "restored_evaluation_context_files": exposed["context_source_files"],
            "evaluator_model": EVALUATOR_MODEL,
            "temperature": 0,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "ragas_version": ragas_version,
            "statement_prompt_sha256": statement_prompt_hash,
            "nli_prompt_sha256": nli_prompt_hash,
            "baseline_recorded_evaluator_model": baseline_rows[claim_id].get(
                "evaluator_model"
            ),
            "baseline_recorded_ragas_transport": baseline_rows[claim_id].get(
                "ragas_transport"
            ),
        }
        completed[claim_id] = result
        append_jsonl(CHECKPOINT_PATH, result)

        print(f"{claim_id:<10} score={score:.3f} statements={len(statements)}")
        for item in verdicts:
            print(f"  [{item['verdict']}] {item['statement']}")
            print(f"      {item['reason']}")
        await asyncio.sleep(INTER_CASE_SECONDS)

    results = [completed[hidden["claim_id"]] for hidden, _ in pairs]
    with DETAIL_PATH.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_statements = sum(row["statement_count"] for row in results)
    supported_statements = sum(
        int(item["verdict"])
        for row in results
        for item in row["statement_verdicts"]
    )
    summary = {
        "analysis": "counter_evidence_restoration_statement_trace",
        "formal_experiment": False,
        "analysis_of_frozen_formal_outputs": True,
        "source_experiment": "E2",
        "n_cases": len(results),
        "baseline_mean_faithfulness": 1.0,
        "restored_mean_faithfulness": (
            sum(row["restored_faithfulness"] for row in results) / len(results)
        ),
        "total_generated_statements": total_statements,
        "supported_statements": supported_statements,
        "unsupported_statements": total_statements - supported_statements,
        "statements_per_case": {
            row["claim_id"]: row["statement_count"] for row in results
        },
        "ragas_version": ragas_version,
        "evaluator_model": EVALUATOR_MODEL,
        "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "statement_prompt_sha256": statement_prompt_hash,
        "nli_prompt_sha256": nli_prompt_hash,
        "provenance_limit": (
            "The original baseline result records evaluator model and transport "
            "but not the RAGAS package version or default-prompt fingerprints."
        ),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 92)
    print(f"Cases:                  {len(results)}")
    print(f"Generated statements:   {total_statements}")
    print(f"Verdict 1 (supported):  {supported_statements}")
    print(f"Verdict 0:              {total_statements - supported_statements}")
    print(f"Detailed trace:         {DETAIL_PATH}")
    print(f"Summary:                {SUMMARY_PATH}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
