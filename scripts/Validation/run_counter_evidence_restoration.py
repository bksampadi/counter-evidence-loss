import asyncio
import json
import os
import time
from pathlib import Path

from openai import AsyncOpenAI
from ragas.llms import llm_factory

try:
    from ragas.metrics.collections import Faithfulness
except ImportError:
    from ragas.metrics import Faithfulness  # type: ignore


EXP2_RESULTS_PATH = Path(
    "benchmark_results/Exp2/formal/experiment_2_generation.jsonl"
)
EXTERNAL_RESULTS_PATH = Path(
    "benchmark_results/external_metric_validation/"
    "external_metric_validation_e2.jsonl"
)
AVAILABLE_CORPUS = Path("benchmarks/Exp1/corpora/formal_v1/available")

OUTPUT_ROOT = Path("benchmark_results/counter_evidence_restoration")
CHECKPOINT_PATH = OUTPUT_ROOT / "counter_evidence_restoration_checkpoint.jsonl"
DETAIL_PATH = OUTPUT_ROOT / "counter_evidence_restoration_e2.jsonl"
SUMMARY_PATH = OUTPUT_ROOT / "counter_evidence_restoration_summary.json"

EVALUATOR_MODEL = "openai/gpt-oss-120b"
MAX_OUTPUT_TOKENS = 4096
INTER_CALL_SECONDS = 12
RATE_LIMIT_RETRY_SECONDS = 12
MAX_RATE_LIMIT_RETRIES = 6


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
    # Identical deterministic wrapper used in external-metric validation v3.
    return f"For the factual question '{query}', the answer is {verdict}."


def context_texts(row: dict) -> list[str]:
    if row["condition"] != "exposed" or row["corpus_state"] != "available":
        raise RuntimeError("Restoration context must be the matched EXPOSED row.")

    texts: list[str] = []
    for filename in row["context_source_files"]:
        path = AVAILABLE_CORPUS / filename
        if not path.exists():
            raise FileNotFoundError(f"Frozen context file missing: {path}")
        texts.append(path.read_text(encoding="utf-8").strip())
    return texts


async def ragas_faithfulness_score(
    metric,
    *,
    query: str,
    response: str,
    contexts: list[str],
) -> float:
    if hasattr(metric, "ascore"):
        result = await metric.ascore(
            user_input=query,
            response=response,
            retrieved_contexts=contexts,
        )
        return float(getattr(result, "value", result))

    from ragas import SingleTurnSample

    sample = SingleTurnSample(
        user_input=query,
        response=response,
        retrieved_contexts=contexts,
    )
    return float(await metric.single_turn_ascore(sample))


async def retry_async_call(label: str, fn):
    last_error: Exception | None = None
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return await fn()
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            is_rate_limit = (
                "429" in message
                or "rate limit" in message
                or "rate_limit" in message
            )
            if not is_rate_limit or attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            print(
                f"{label}: rate limit; retrying "
                f"({attempt + 1}/{MAX_RATE_LIMIT_RETRIES})..."
            )
            await asyncio.sleep(RATE_LIMIT_RETRY_SECONDS)
    raise RuntimeError(f"{label} retry loop exhausted") from last_error


def paired_rows(rows: list[dict]) -> list[tuple[dict, dict]]:
    by_key = {(row["condition"], row["claim_id"]): row for row in rows}
    claim_ids = sorted(
        row["claim_id"]
        for row in rows
        if row["condition"] == "available_not_exposed"
    )
    if len(claim_ids) != 6:
        raise RuntimeError(
            f"Expected six AVAILABLE_NOT_EXPOSED rows, got {len(claim_ids)}."
        )

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


async def main_async() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    exp2_rows = load_jsonl(EXP2_RESULTS_PATH)
    if len(exp2_rows) != 18:
        raise RuntimeError(f"Expected 18 frozen E2 rows, got {len(exp2_rows)}.")

    external_rows = load_jsonl(EXTERNAL_RESULTS_PATH)
    baselines = {
        row["claim_id"]: float(row["ragas_faithfulness"])
        for row in external_rows
        if row["condition"] == "available_not_exposed"
    }
    if len(baselines) != 6:
        raise RuntimeError(f"Expected six baseline scores, got {len(baselines)}.")
    nonmaximal = {key: value for key, value in baselines.items() if value != 1.0}
    if nonmaximal:
        raise RuntimeError(f"Expected all frozen baselines to equal 1.0: {nonmaximal}")

    pairs = paired_rows(exp2_rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    completed: dict[str, dict] = {}
    if CHECKPOINT_PATH.exists():
        completed = {
            row["claim_id"]: row for row in load_jsonl(CHECKPOINT_PATH)
        }
        print(f"Resuming from checkpoint: {len(completed)} completed calls")

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
    faithfulness = Faithfulness(llm=evaluator_llm)

    print()
    print("COUNTER-EVIDENCE RESTORATION TEST — FROZEN E2 ANSWERS")
    print("=" * 88)
    print(f"Evaluator model: {EVALUATOR_MODEL}")
    print("Answers:         six frozen, incorrect AVAILABLE_NOT_EXPOSED answers")
    print("Contexts:        matched EXPOSED contexts")
    print("New calls:       six RAGAS Faithfulness evaluations only")
    print()

    for hidden, exposed in pairs:
        claim_id = hidden["claim_id"]
        if claim_id in completed:
            saved = completed[claim_id]
            print(
                f"{claim_id:<10} baseline={saved['baseline_faithfulness']:.3f} "
                f"restored={saved['restored_faithfulness']:.3f} [checkpoint]"
            )
            continue

        query = hidden["query"]
        verdict = hidden["verdict"]
        response = render_response(query, verdict)
        contexts = context_texts(exposed)

        score = await retry_async_call(
            f"RAGAS restoration {claim_id}",
            lambda: ragas_faithfulness_score(
                faithfulness,
                query=query,
                response=response,
                contexts=contexts,
            ),
        )

        result = {
            "analysis": "counter_evidence_restoration",
            "formal_experiment": False,
            "analysis_of_frozen_formal_outputs": True,
            "source_experiment": "E2",
            "evaluator_model": EVALUATOR_MODEL,
            "claim_id": claim_id,
            "truth_label": bool(hidden["truth_label"]),
            "query": query,
            "frozen_source_condition": "available_not_exposed",
            "frozen_generator_verdict": verdict,
            "frozen_answer_correct": bool(hidden["correct"]),
            "evaluation_context_condition": "exposed",
            "rendered_response_for_ragas": response,
            "baseline_faithfulness": baselines[claim_id],
            "restored_faithfulness": score,
            "delta_restored_minus_baseline": score - baselines[claim_id],
            "source_answer_context_files": hidden["context_source_files"],
            "restored_evaluation_context_files": exposed["context_source_files"],
        }
        completed[claim_id] = result
        append_jsonl(CHECKPOINT_PATH, result)
        print(f"{claim_id:<10} baseline=1.000 restored={score:.3f}")
        await asyncio.sleep(INTER_CALL_SECONDS)

    results = [completed[hidden["claim_id"]] for hidden, _ in pairs]
    with DETAIL_PATH.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    restored_scores = [row["restored_faithfulness"] for row in results]
    mean_restored = sum(restored_scores) / len(restored_scores)
    summary = {
        "analysis": "counter_evidence_restoration",
        "formal_experiment": False,
        "analysis_of_frozen_formal_outputs": True,
        "source_experiment": "E2",
        "n": len(results),
        "baseline_mean_faithfulness": 1.0,
        "restored_mean_faithfulness": mean_restored,
        "mean_change": mean_restored - 1.0,
        "restored_scores": restored_scores,
        "all_restored_scores_maximal": all(score == 1.0 for score in restored_scores),
        "interpretation_guardrail": (
            "This diagnostic tests faithfulness after restoring counter-evidence "
            "to the evaluator context while holding the six generated answers fixed."
        ),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print(f"Baseline mean: {1.0:.3f}")
    print(f"Restored mean: {mean_restored:.3f}")
    print(f"Mean change:   {mean_restored - 1.0:+.3f}")
    print(f"Detailed:      {DETAIL_PATH}")
    print(f"Summary:       {SUMMARY_PATH}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
