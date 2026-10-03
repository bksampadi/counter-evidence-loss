import asyncio
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ragas.llms import llm_factory

try:
    from ragas.metrics.collections import Faithfulness
except ImportError:
    from ragas.metrics import Faithfulness  # type: ignore

from signalrank.components.llm.providers.groq import build_groq_provider
from signalrank.services.llm_service import LLMService

EXP1_SUMMARY_PATH = Path(
    "benchmark_results/Exp1/formal/experiment_1_summary.json"
)
EXP2_RESULTS_PATH = Path(
    "benchmark_results/Exp2/formal/experiment_2_generation.jsonl"
)

AVAILABLE_CORPUS = Path("benchmarks/Exp1/corpora/formal_v1/available")
ABSENT_CORPUS = Path("benchmarks/Exp1/corpora/formal_v1/absent")

OUTPUT_ROOT = Path(
    "benchmark_results/external_metric_validation"
)
CHECKPOINT_PATH = OUTPUT_ROOT / "external_metric_validation_checkpoint.jsonl"

# External evaluator/judge model. This is NOT the 20B generator used in the
# formal experiments.
EVALUATOR_MODEL = "openai/gpt-oss-120b"
MAX_OUTPUT_TOKENS = 4096
INTER_CALL_SECONDS = 12
BETWEEN_METRICS_SECONDS = 10
RATE_LIMIT_RETRY_SECONDS = 12
MAX_RATE_LIMIT_RETRIES = 6

SUFFICIENCY_SYSTEM_PROMPT = """You are a blind context-sufficiency judge.

You receive only a question and the context supplied to a RAG generator.
You do NOT know the reference answer and must not use outside knowledge.

Return exactly ONE word:
SUFFICIENT
or
INSUFFICIENT

Use SUFFICIENT if the supplied context appears to contain enough direct
evidence to give a definite answer to the question from the context alone.

Use INSUFFICIENT if the context does not appear to contain enough evidence
to give a definite answer.

Important:
- Judge apparent answerability from the supplied context only.
- Do not ask whether hidden evidence might exist elsewhere.
- Do not compare against a reference answer.
"""

Sufficiency = Literal["SUFFICIENT", "INSUFFICIENT"]

SUFFICIENCY_PATTERN = re.compile(
    r"^\s*(SUFFICIENT|INSUFFICIENT)[.!]?\s*$",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ai_message_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def parse_sufficiency(text: str) -> Sufficiency:
    match = SUFFICIENCY_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(
            f"Sufficiency judge returned non-canonical output: {text!r}"
        )
    return match.group(1).upper()  # type: ignore[return-value]


def context_texts(row: dict) -> list[str]:
    state = row["corpus_state"]
    root = AVAILABLE_CORPUS if state == "available" else ABSENT_CORPUS

    texts = []
    for filename in row["context_source_files"]:
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Frozen context file missing: {path}")
        texts.append(path.read_text(encoding="utf-8").strip())

    return texts


def render_response(query: str, verdict: str) -> str:
    # The formal generator emitted a binary verdict only. This deterministic
    # wrapper makes the proposition explicit for the external faithfulness
    # metric without changing its content.
    return (
        f"For the factual question '{query}', "
        f"the answer is {verdict}."
    )


async def ragas_faithfulness_score(
    metric,
    *,
    query: str,
    response: str,
    contexts: list[str],
) -> float:
    # RAGAS v0.4+ collections API.
    if hasattr(metric, "ascore"):
        result = await metric.ascore(
            user_input=query,
            response=response,
            retrieved_contexts=contexts,
        )
        value = getattr(result, "value", result)
        return float(value)

    # Legacy compatibility, if a pre-v0.4 RAGAS happens to be loaded.
    from ragas import SingleTurnSample

    sample = SingleTurnSample(
        user_input=query,
        response=response,
        retrieved_contexts=contexts,
    )
    result = await metric.single_turn_ascore(sample)
    return float(result)



async def retry_async_call(label: str, fn):
    last_error: Exception | None = None

    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return await fn()
        except Exception as exc:  # intentionally transport-level
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


def retry_sync_call(label: str, fn):
    last_error: Exception | None = None

    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # intentionally transport-level
            last_error = exc
            message = str(exc).lower()
            cause = getattr(exc, "__cause__", None)
            if cause is not None:
                message += " " + str(cause).lower()

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
            time.sleep(RATE_LIMIT_RETRY_SECONDS)

    raise RuntimeError(f"{label} retry loop exhausted") from last_error


def load_checkpoint() -> dict[tuple[str, str], dict]:
    if not CHECKPOINT_PATH.exists():
        return {}

    completed: dict[tuple[str, str], dict] = {}
    with CHECKPOINT_PATH.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            completed[(row["condition"], row["claim_id"])] = row

    return completed


def append_checkpoint(row: dict) -> None:
    with CHECKPOINT_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


async def main_async() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    exp1 = load_json(EXP1_SUMMARY_PATH)
    exp2 = load_jsonl(EXP2_RESULTS_PATH)

    if exp1.get("formal_acceptance_pass") is not True:
        raise RuntimeError("Formal E1 acceptance is not True.")

    if len(exp2) != 18:
        raise RuntimeError(f"Expected 18 frozen E2 rows, got {len(exp2)}.")

    # RAGAS external faithfulness evaluator.
    # Use Groq through its official OpenAI-compatible endpoint.
    # This deliberately presents RAGAS with an OpenAI client so RAGAS uses
    # its stable OpenAI Instructor adapter rather than the broken native
    # Groq adapter path.
    ragas_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    evaluator_llm = llm_factory(
        EVALUATOR_MODEL,
        provider="openai",
        client=ragas_client,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    faithfulness = Faithfulness(llm=evaluator_llm)

    # Blind sufficiency judge uses the same independent 120B evaluator model,
    # but plain one-word output avoids structured-output transport issues.
    judge_provider = build_groq_provider(
        model_name=EVALUATOR_MODEL,
        max_retries=0,
        max_output_tokens=1024,
    )
    judge_llm = LLMService([judge_provider])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    completed = load_checkpoint()
    rows = list(completed.values())
    if completed:
        print(f"Resuming from checkpoint: {len(completed)} completed cases")

    print()
    print("EXTERNAL METRIC VALIDATION — FROZEN FORMAL E2 CONTEXTS")
    print("=" * 108)
    print(f"Evaluator model:       {EVALUATOR_MODEL}")
    print("RAGAS transport:       Groq via AsyncOpenAI-compatible endpoint")
    print("RAGAS metric:          Faithfulness")
    print("Blind judge:           Context sufficiency")
    print("Reference shown to judge: NO")
    print("Contexts:              18 frozen E2 cases")
    print(f"Pacing:                {BETWEEN_METRICS_SECONDS}s between metric/judge, "
          f"{INTER_CALL_SECONDS}s between cases")
    print(f"Checkpoint:            {CHECKPOINT_PATH}")
    print()

    for row in exp2:
        checkpoint_key = (row["condition"], row["claim_id"])
        if checkpoint_key in completed:
            saved = completed[checkpoint_key]
            print(
                f"{row['condition']:<24}"
                f"{row['claim_id']:<10}"
                f"correct={str(row['correct']):<6}"
                f"faith={saved['ragas_faithfulness']:.3f} "
                f"suff={saved['blind_context_sufficiency']} [checkpoint]"
            )
            continue

        query = row["query"]
        verdict = row["verdict"]
        contexts = context_texts(row)
        response = render_response(query, verdict)

        faith_score = await retry_async_call(
            f"RAGAS faithfulness {row['condition']} / {row['claim_id']}",
            lambda: ragas_faithfulness_score(
                faithfulness,
                query=query,
                response=response,
                contexts=contexts,
            ),
        )

        # RAGAS Faithfulness may itself use multiple evaluator-model calls.
        # Give the shared GPT-OSS-120B TPM window time to replenish before
        # invoking the blind sufficiency judge.
        await asyncio.sleep(BETWEEN_METRICS_SECONDS)

        suff_prompt = (
            f"Question:\n{query}\n\n"
            "Context:\n"
            + "\n\n".join(
                f"[Evidence {i}]\n{text}"
                for i, text in enumerate(contexts, start=1)
            )
            + "\n\nIs this context sufficient to answer the question?"
        )

        message = retry_sync_call(
            f"Blind sufficiency {row['condition']} / {row['claim_id']}",
            lambda: judge_llm.invoke(
                [
                    SystemMessage(content=SUFFICIENCY_SYSTEM_PROMPT),
                    HumanMessage(content=suff_prompt),
                ]
            ),
        )

        raw_sufficiency = ai_message_text(message)
        sufficiency = parse_sufficiency(raw_sufficiency)

        result = {
            "analysis": "external_metric_validation",
            "source_experiment": "E2",
            "evaluator_model": EVALUATOR_MODEL,
"ragas_transport": "groq_via_openai_compatible_async_client",
            "claim_id": row["claim_id"],
            "truth_label": bool(row["truth_label"]),
            "condition": row["condition"],
            "query": query,
            "generator_verdict": verdict,
            "reference_correct": bool(row["correct"]),
            "rendered_response_for_ragas": response,
            "ragas_faithfulness": faith_score,
            "blind_context_sufficiency": sufficiency,
            "blind_context_sufficient": sufficiency == "SUFFICIENT",
            "raw_sufficiency_output": raw_sufficiency,
            "context_source_files": row["context_source_files"],
        }
        rows.append(result)
        completed[checkpoint_key] = result
        append_checkpoint(result)

        print(
            f"{row['condition']:<24}"
            f"{row['claim_id']:<10}"
            f"correct={str(row['correct']):<6}"
            f"faith={faith_score:.3f} "
            f"suff={sufficiency}"
        )

        await asyncio.sleep(INTER_CALL_SECONDS)

    rows = list({(r['condition'], r['claim_id']): r for r in rows}.values())

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    condition_order = (
        "exposed",
        "available_not_exposed",
        "absent",
    )

    summary_by_condition = {}

    for condition in condition_order:
        items = grouped[condition]
        summary_by_condition[condition] = {
            "n": len(items),
            "reference_accuracy": (
                sum(item["reference_correct"] for item in items) / len(items)
            ),
            "mean_ragas_faithfulness": (
                sum(item["ragas_faithfulness"] for item in items) / len(items)
            ),
            "blind_sufficiency_rate": (
                sum(item["blind_context_sufficient"] for item in items)
                / len(items)
            ),
        }

    hidden = summary_by_condition["available_not_exposed"]

    pathology_externalized = (
        hidden["reference_accuracy"] == 0.0
        and hidden["mean_ragas_faithfulness"] >= 0.8
        and hidden["blind_sufficiency_rate"] >= 5 / 6
    )

    detail_path = OUTPUT_ROOT / "external_metric_validation_e2.jsonl"
    with detail_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "analysis": "external_metric_validation",
        "formal_experiment": False,
        "analysis_of_frozen_formal_outputs": True,
        "stimulus_version": exp1["stimulus_version"],
        "evaluator_model": EVALUATOR_MODEL,
"ragas_transport": "groq_via_openai_compatible_async_client",
        "ragas_metric": "Faithfulness",
        "blind_sufficiency_reference_visible": False,
        "conditions": summary_by_condition,
        "external_pathology_pattern_present": pathology_externalized,
        "interpretation_guardrail": (
            "RAGAS Faithfulness is treated as an external context-anchored "
            "validation metric, not as ground-truth correctness. The blind "
            "sufficiency judge sees no reference answer."
        ),
    }

    summary_path = OUTPUT_ROOT / "external_metric_validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 108)
    print("CONDITION SUMMARY")
    print("-" * 108)

    for condition in condition_order:
        item = summary_by_condition[condition]
        print(
            f"{condition:<24}"
            f"n={item['n']} "
            f"| ref_acc={item['reference_accuracy']:.3f} "
            f"| ragas_faith={item['mean_ragas_faithfulness']:.3f} "
            f"| blind_suff={item['blind_sufficiency_rate']:.3f}"
        )

    print()
    print(
        "External pathology pattern present: "
        f"{pathology_externalized}"
    )
    print(f"Detailed results: {detail_path}")
    print(f"Summary:          {summary_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
