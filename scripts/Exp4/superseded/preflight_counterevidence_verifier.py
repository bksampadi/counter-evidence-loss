import json
import time
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from signalrank.components.llm.providers.groq import build_groq_provider
from signalrank.services.llm_service import LLMService

STIMULUS_ROOT = Path("benchmarks/Exp1/stimuli_v1")
CLAIMS_PATH = STIMULUS_ROOT / "claims.json"
OUTPUT_ROOT = Path("benchmark_results/Exp4/preflight")

MODEL_NAME = "openai/gpt-oss-20b"
MAX_RETRIES = 0
MAX_OUTPUT_TOKENS = 1024
INTER_CALL_SECONDS = 10

SYSTEM_PROMPT = """You are verifying candidate counter-evidence for a factual claim.

You are given:
1. a claim,
2. a provisional TRUE/FALSE verdict produced from earlier evidence,
3. one candidate evidence passage.

Classify the candidate as DIRECT_COUNTEREVIDENCE only if it:
- is about the SAME entity/test/condition as the claim, AND
- directly contradicts the provisional verdict in a way that bears on the exact claim.

Otherwise return NOT_COUNTEREVIDENCE.

Do not use outside knowledge.
Do not infer relevance from generic words such as audit, correction, final, or superseded.
"""

Decision = Literal[
    "DIRECT_COUNTEREVIDENCE",
    "NOT_COUNTEREVIDENCE",
]


class CounterEvidenceAssessment(BaseModel):
    decision: Decision


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def truth_to_verdict(truth_label: bool) -> str:
    return "TRUE" if truth_label else "FALSE"


def opposite_truth(truth_label: bool) -> str:
    return "FALSE" if truth_label else "TRUE"


def wrong_entity_map(claim_ids: list[str]) -> dict[str, str]:
    return {
        claim_id: claim_ids[(i + 1) % len(claim_ids)]
        for i, claim_id in enumerate(claim_ids)
    }


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
    print("EXP4 PREFLIGHT — COUNTER-EVIDENCE VERIFIER VALIDATION")
    print("=" * 100)
    print(f"Stimuli:     {manifest['stimulus_version']}")
    print(f"Model:       {MODEL_NAME}")
    print("Temperature: 0")
    print("Fallbacks:   none")
    print("Model-visible filenames/roles: NO")
    print("Checks:      24")
    print()
    print(
        f"{'Claim':<11}"
        f"{'Candidate':<24}"
        f"{'Expected':<24}"
        f"{'Observed':<24}"
        f"{'Pass'}"
    )
    print("-" * 100)

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
                "DIRECT_COUNTEREVIDENCE",
            ),
            (
                "wrong_entity_critical",
                STIMULUS_ROOT / f"{wrong_id}__critical.txt",
                "NOT_COUNTEREVIDENCE",
            ),
            (
                "target_neutral",
                STIMULUS_ROOT / f"{claim_id}__neutral_1.txt",
                "NOT_COUNTEREVIDENCE",
            ),
            (
                "target_replacement",
                STIMULUS_ROOT / f"{claim_id}__replacement.txt",
                "NOT_COUNTEREVIDENCE",
            ),
        ]

        for case_name, path, expected in cases:
            prompt = (
                f"Claim:\n{query}\n\n"
                f"Provisional verdict:\n{provisional}\n\n"
                f"Candidate evidence:\n{read_text(path)}\n\n"
                "Classify the candidate."
            )

            assessment = llm.invoke_structured(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ],
                CounterEvidenceAssessment,
            )

            observed = assessment.decision
            passed = observed == expected

            rows.append(
                {
                    "preflight": "exp4_counterevidence_verifier_validation",
                    "formal_experiment": False,
                    "stimulus_version": manifest["stimulus_version"],
                    "model_name": MODEL_NAME,
                    "claim_id": claim_id,
                    "truth_label": truth_label,
                    "query": query,
                    "provisional_verdict": provisional,
                    "candidate_case": case_name,
                    "candidate_file": path.name,
                    "expected": expected,
                    "observed": observed,
                    "pass": passed,
                    "model_visible_file_labels": False,
                }
            )

            print(
                f"{claim_id:<11}"
                f"{case_name:<24}"
                f"{expected:<24}"
                f"{observed:<24}"
                f"{str(passed)}"
            )

            time.sleep(INTER_CALL_SECONDS)

    passed_count = sum(row["pass"] for row in rows)
    all_pass = passed_count == len(rows)

    detail_path = OUTPUT_ROOT / "counterevidence_verifier_preflight.jsonl"
    with detail_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "preflight": "exp4_counterevidence_verifier_validation",
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

    summary_path = OUTPUT_ROOT / "counterevidence_verifier_preflight_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(f"PASSED: {passed_count}/{len(rows)}")
    print(f"ALL VERIFIER CHECKS PASS: {all_pass}")
    print(f"Summary: {summary_path}")

    if not all_pass:
        raise SystemExit(
            "Exp4 verifier preflight failed. Do NOT run formal E4."
        )


if __name__ == "__main__":
    main()
