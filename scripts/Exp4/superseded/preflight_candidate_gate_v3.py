import json
import time
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from signalrank.components.llm.providers.groq import build_groq_provider
from signalrank.services.llm_service import LLMService, LLMUnavailableError

STIMULUS_ROOT = Path("benchmarks/Exp1/stimuli_v1")
CLAIMS_PATH = STIMULUS_ROOT / "claims.json"
OUTPUT_ROOT = Path("benchmark_results/Exp4/preflight_v2")

MODEL_NAME = "openai/gpt-oss-20b"
MAX_RETRIES = 0
MAX_OUTPUT_TOKENS = 1024
INTER_CALL_SECONDS = 10
STRUCTURED_RETRY_ATTEMPTS = 3
STRUCTURED_RETRY_DELAY_SECONDS = 8

SYSTEM_PROMPT = """You are evaluating whether ONE candidate evidence passage establishes a factual claim.

Use ONLY the candidate evidence. Do not use outside knowledge.

Return:
- TRUE if the candidate evidence establishes that the claim is true.
- FALSE if the candidate evidence establishes that the claim is false.
- INSUFFICIENT if the candidate evidence does not establish either verdict.

Important:
- A passage about a different entity, test, condition, endpoint, or sample is INSUFFICIENT.
- A passage that only describes handling, setup, storage, documentation, equipment, or administration is INSUFFICIENT.
- Missing outcome information is INSUFFICIENT, not FALSE.
"""

Verdict = Literal["TRUE", "FALSE", "INSUFFICIENT"]


class EvidenceVerdict(BaseModel):
    verdict: Verdict


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def truth_to_verdict(truth_label: bool) -> Verdict:
    return "TRUE" if truth_label else "FALSE"


def opposite_truth(truth_label: bool) -> Verdict:
    return "FALSE" if truth_label else "TRUE"


def wrong_entity_map(claim_ids: list[str]) -> dict[str, str]:
    return {
        claim_id: claim_ids[(i + 1) % len(claim_ids)]
        for i, claim_id in enumerate(claim_ids)
    }



def invoke_structured_reliably(
    *,
    llm: LLMService,
    messages,
    schema,
):
    """Retry only when no parseable structured result was returned."""
    last_error: Exception | None = None

    for attempt in range(1, STRUCTURED_RETRY_ATTEMPTS + 1):
        try:
            return llm.invoke_structured(messages, schema), attempt
        except LLMUnavailableError as exc:
            last_error = exc
            if attempt == STRUCTURED_RETRY_ATTEMPTS:
                raise
            print(
                f"\nStructured-output plumbing failure; retrying identical call "
                f"({attempt + 1}/{STRUCTURED_RETRY_ATTEMPTS})..."
            )
            time.sleep(STRUCTURED_RETRY_DELAY_SECONDS)

    raise RuntimeError("Structured invocation retry loop exhausted.") from last_error


def main() -> None:
    manifest = load_json(CLAIMS_PATH)
    claims = sorted(manifest["claims"], key=lambda x: x["claim_id"])
    claim_ids = [claim["claim_id"] for claim in claims]
    mapping = wrong_entity_map(claim_ids)

    provider = build_groq_provider(
        model_name=MODEL_NAME,
        max_retries=MAX_RETRIES,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    llm = LLMService([provider])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print()
    print("EXP4 PREFLIGHT V2 — CANDIDATE EVIDENTIAL-VERDICT GATE")
    print("=" * 104)
    print(f"Stimuli:     {manifest['stimulus_version']}")
    print(f"Model:       {MODEL_NAME}")
    print("Temperature: 0")
    print("Fallbacks:   none")
    print("Model-visible filenames/roles: NO")
    print("Checks:      30")
    print()
    print(
        f"{'Claim':<11}"
        f"{'Candidate':<24}"
        f"{'Expected':<16}"
        f"{'Observed':<16}"
        f"{'Admit?':<9}"
        f"{'Pass'}"
    )
    print("-" * 104)

    for claim in claims:
        claim_id = claim["claim_id"]
        truth_label = bool(claim["truth_label"])
        query = claim["query"]
        provisional = opposite_truth(truth_label)
        wrong_id = mapping[claim_id]

        cases = [
            (
                "target_critical",
                STIMULUS_ROOT / f"{claim_id}__critical.txt",
                truth_to_verdict(truth_label),
                True,
            ),
            (
                "wrong_entity_critical",
                STIMULUS_ROOT / f"{wrong_id}__critical.txt",
                "INSUFFICIENT",
                False,
            ),
            (
                "wrong_entity_support",
                STIMULUS_ROOT / f"{wrong_id}__support.txt",
                "INSUFFICIENT",
                False,
            ),
            (
                "target_neutral",
                STIMULUS_ROOT / f"{claim_id}__neutral_1.txt",
                "INSUFFICIENT",
                False,
            ),
            (
                "target_replacement",
                STIMULUS_ROOT / f"{claim_id}__replacement.txt",
                "INSUFFICIENT",
                False,
            ),
        ]

        for case_name, path, expected, expected_admit in cases:
            prompt = (
                f"Claim:\n{query}\n\n"
                f"Candidate evidence:\n{read_text(path)}\n\n"
                "Return the evidential verdict."
            )

            assessment, structured_attempts = invoke_structured_reliably(
                llm=llm,
                messages=[
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ],
                schema=EvidenceVerdict,
            )

            observed = assessment.verdict
            admit = (
                observed in {"TRUE", "FALSE"}
                and observed != provisional
            )

            passed = (
                observed == expected
                and admit == expected_admit
            )

            rows.append(
                {
                    "preflight": "exp4_candidate_evidential_verdict_gate",
                    "formal_experiment": False,
                    "stimulus_version": manifest["stimulus_version"],
                    "model_name": MODEL_NAME,
                    "claim_id": claim_id,
                    "truth_label": truth_label,
                    "query": query,
                    "provisional_verdict": provisional,
                    "candidate_case": case_name,
                    "candidate_file": path.name,
                    "expected_evidence_verdict": expected,
                    "observed_evidence_verdict": observed,
                    "expected_admit": expected_admit,
                    "observed_admit": admit,
                    "structured_attempts": structured_attempts,
                    "pass": passed,
                    "model_visible_file_labels": False,
                }
            )

            print(
                f"{claim_id:<11}"
                f"{case_name:<24}"
                f"{expected:<16}"
                f"{observed:<16}"
                f"{str(admit):<9}"
                f"{str(passed)}"
            )

            time.sleep(INTER_CALL_SECONDS)

    passed_count = sum(row["pass"] for row in rows)
    all_pass = passed_count == len(rows)

    detail_path = OUTPUT_ROOT / "candidate_gate_preflight.jsonl"
    with detail_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "preflight": "exp4_candidate_evidential_verdict_gate",
        "formal_experiment": False,
        "stimulus_version": manifest["stimulus_version"],
        "model_provider": "groq",
        "model_name": MODEL_NAME,
        "temperature": 0,
        "model_visible_file_labels": False,
        "n_claims": len(claims),
        "n_checks": len(rows),
        "passed_checks": passed_count,
        "all_checks_pass": all_pass,
        "wrong_entity_map": mapping,
    }

    summary_path = OUTPUT_ROOT / "candidate_gate_preflight_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 104)
    print(f"PASSED: {passed_count}/{len(rows)}")
    print(f"ALL CANDIDATE-GATE CHECKS PASS: {all_pass}")
    print(f"Summary: {summary_path}")

    if not all_pass:
        raise SystemExit(
            "Exp4 candidate-gate preflight failed. Do NOT run formal E4."
        )


if __name__ == "__main__":
    main()
