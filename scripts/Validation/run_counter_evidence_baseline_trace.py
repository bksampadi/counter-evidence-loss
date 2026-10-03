import asyncio
import importlib.metadata
import json
import os
from pathlib import Path

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

import run_counter_evidence_restoration_trace as restoration


RESTORATION_TRACE_PATH = Path(
    "benchmark_results/counter_evidence_restoration_trace/restoration_trace_e2.jsonl"
)
OUTPUT_ROOT = Path("benchmark_results/counter_evidence_paired_trace")
CHECKPOINT_PATH = OUTPUT_ROOT / "baseline_trace_checkpoint.jsonl"
DETAIL_PATH = OUTPUT_ROOT / "paired_trace_e2.jsonl"
SUMMARY_PATH = OUTPUT_ROOT / "paired_trace_summary.json"


def hidden_contexts(row: dict) -> list[str]:
    if (
        row["condition"] != "available_not_exposed"
        or row["corpus_state"] != "available"
    ):
        raise RuntimeError("Baseline context must be AVAILABLE_NOT_EXPOSED.")

    texts: list[str] = []
    for filename in row["context_source_files"]:
        path = restoration.AVAILABLE_CORPUS / filename
        if not path.exists():
            raise FileNotFoundError(f"Frozen context file missing: {path}")
        texts.append(path.read_text(encoding="utf-8").strip())
    return texts


async def main_async() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    exp2_rows = restoration.load_jsonl(restoration.EXP2_RESULTS_PATH)
    pairs = restoration.paired_rows(exp2_rows)
    restoration_rows = {
        row["claim_id"]: row
        for row in restoration.load_jsonl(RESTORATION_TRACE_PATH)
    }
    if len(restoration_rows) != 6:
        raise RuntimeError(
            f"Expected six completed restoration traces, got {len(restoration_rows)}."
        )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    evaluator_llm = llm_factory(
        restoration.EVALUATOR_MODEL,
        provider="openai",
        client=client,
        temperature=0,
        max_tokens=restoration.MAX_OUTPUT_TOKENS,
    )
    metric = Faithfulness(llm=evaluator_llm)

    ragas_version = importlib.metadata.version("ragas")
    statement_prompt_hash = restoration.fingerprint(
        metric.statement_generator_prompt
    )
    nli_prompt_hash = restoration.fingerprint(metric.nli_statement_prompt)

    for row in restoration_rows.values():
        if row["ragas_version"] != ragas_version:
            raise RuntimeError("RAGAS version differs from restoration trace.")
        if row["statement_prompt_sha256"] != statement_prompt_hash:
            raise RuntimeError("Statement prompt differs from restoration trace.")
        if row["nli_prompt_sha256"] != nli_prompt_hash:
            raise RuntimeError("NLI prompt differs from restoration trace.")
        if row["evaluator_model"] != restoration.EVALUATOR_MODEL:
            raise RuntimeError("Evaluator model differs from restoration trace.")
        if row["temperature"] != 0:
            raise RuntimeError("Temperature differs from restoration trace.")
        if row["max_output_tokens"] != restoration.MAX_OUTPUT_TOKENS:
            raise RuntimeError("Token limit differs from restoration trace.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict] = {}
    if CHECKPOINT_PATH.exists():
        completed = {
            row["claim_id"]: row
            for row in restoration.load_jsonl(CHECKPOINT_PATH)
        }
        print(f"Resuming from checkpoint: {len(completed)} completed cases")

    print()
    print("COUNTER-EVIDENCE RESTORATION — MATCHED BASELINE TRACE")
    print("=" * 92)
    print(f"Evaluator model: {restoration.EVALUATOR_MODEL}")
    print(f"RAGAS version:   {ragas_version}")
    print(f"Statement prompt SHA-256: {statement_prompt_hash}")
    print(f"NLI prompt SHA-256:       {nli_prompt_hash}")
    print("Provenance match to restoration trace: PASS")
    print()

    for hidden, _ in pairs:
        claim_id = hidden["claim_id"]
        if claim_id in completed:
            saved = completed[claim_id]
            print(
                f"{claim_id:<10} baseline={saved['baseline_faithfulness']:.3f} "
                f"restored={saved['restored_faithfulness']:.3f} [checkpoint]"
            )
            continue

        query = hidden["query"]
        response = restoration.render_response(query, hidden["verdict"])
        score, statements, verdicts = await restoration.trace_faithfulness(
            metric,
            query=query,
            response=response,
            contexts=hidden_contexts(hidden),
        )
        restored = restoration_rows[claim_id]

        result = {
            "analysis": "counter_evidence_restoration_paired_trace",
            "formal_experiment": False,
            "analysis_of_frozen_formal_outputs": True,
            "source_experiment": "E2",
            "claim_id": claim_id,
            "truth_label": bool(hidden["truth_label"]),
            "query": query,
            "frozen_generator_verdict": hidden["verdict"],
            "frozen_answer_correct": bool(hidden["correct"]),
            "rendered_response_for_ragas": response,
            "baseline_context_condition": "available_not_exposed",
            "baseline_context_files": hidden["context_source_files"],
            "baseline_faithfulness": score,
            "baseline_statement_count": len(statements),
            "baseline_generated_statements": statements,
            "baseline_statement_verdicts": verdicts,
            "restored_context_condition": "exposed",
            "restored_context_files": restored[
                "restored_evaluation_context_files"
            ],
            "restored_faithfulness": restored["restored_faithfulness"],
            "restored_statement_count": restored["statement_count"],
            "restored_generated_statements": restored["generated_statements"],
            "restored_statement_verdicts": restored["statement_verdicts"],
            "paired_change": restored["restored_faithfulness"] - score,
            "evaluator_model": restoration.EVALUATOR_MODEL,
            "temperature": 0,
            "max_output_tokens": restoration.MAX_OUTPUT_TOKENS,
            "ragas_version": ragas_version,
            "statement_prompt_sha256": statement_prompt_hash,
            "nli_prompt_sha256": nli_prompt_hash,
            "provenance_match": True,
        }
        completed[claim_id] = result
        restoration.append_jsonl(CHECKPOINT_PATH, result)

        print(
            f"{claim_id:<10} baseline={score:.3f} "
            f"restored={restored['restored_faithfulness']:.3f}"
        )
        for item in verdicts:
            print(f"  [{item['verdict']}] {item['statement']}")
            print(f"      {item['reason']}")

        await asyncio.sleep(restoration.INTER_CASE_SECONDS)

    results = [completed[hidden["claim_id"]] for hidden, _ in pairs]
    with DETAIL_PATH.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    baseline_mean = sum(row["baseline_faithfulness"] for row in results) / 6
    restored_mean = sum(row["restored_faithfulness"] for row in results) / 6
    all_paired_decreases = all(row["paired_change"] < 0 for row in results)
    summary = {
        "analysis": "counter_evidence_restoration_paired_trace",
        "formal_experiment": False,
        "analysis_of_frozen_formal_outputs": True,
        "source_experiment": "E2",
        "n_cases": 6,
        "baseline_mean_faithfulness": baseline_mean,
        "restored_mean_faithfulness": restored_mean,
        "mean_change": restored_mean - baseline_mean,
        "all_paired_decreases": all_paired_decreases,
        "evaluator_model": restoration.EVALUATOR_MODEL,
        "temperature": 0,
        "max_output_tokens": restoration.MAX_OUTPUT_TOKENS,
        "ragas_version": ragas_version,
        "statement_prompt_sha256": statement_prompt_hash,
        "nli_prompt_sha256": nli_prompt_hash,
        "provenance_match_to_restoration_trace": True,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 92)
    print(f"Baseline mean:           {baseline_mean:.3f}")
    print(f"Restored mean:           {restored_mean:.3f}")
    print(f"Mean change:             {restored_mean - baseline_mean:+.3f}")
    print(f"All paired decreases:    {all_paired_decreases}")
    print("Provenance match:        PASS")
    print(f"Detailed paired trace:   {DETAIL_PATH}")
    print(f"Summary:                 {SUMMARY_PATH}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
