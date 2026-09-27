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
OUTPUT_ROOT = Path("benchmark_results/Exp3/preflight")

MODEL_NAME = "openai/gpt-oss-20b"
MAX_RETRIES = 0
MAX_OUTPUT_TOKENS = 1024
INTER_CALL_SECONDS = 10

SYSTEM_PROMPT = """You are evaluating whether supplied evidence establishes a factual claim.

Use ONLY the supplied evidence. Do not use outside knowledge.

Return:
- TRUE if the evidence establishes that the claim is true.
- FALSE if the evidence establishes that the claim is false.
- INSUFFICIENT if the evidence does not establish either verdict.

Do not treat missing evidence as evidence that a claim is false.
"""

Verdict = Literal["TRUE", "FALSE", "INSUFFICIENT"]


class ClaimAssessment(BaseModel):
    verdict: Verdict


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def opposite_truth(truth_label: bool) -> Verdict:
    return "FALSE" if truth_label else "TRUE"


def wrong_entity_map(claim_ids: list[str]) -> dict[str, str]:
    # Deterministic cyclic mapping; never pair a claim with its own critical passage.
    return {
        claim_id: claim_ids[(i + 1) % len(claim_ids)]
        for i, claim_id in enumerate(claim_ids)
    }


def main() -> None:
    manifest = load_json(CLAIMS_PATH)
    claims = sorted(manifest["claims"], key=lambda x: x["claim_id"])
    claim_ids = [c["claim_id"] for c in claims]
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
    print("EXP3 PREFLIGHT — WRONG-ENTITY CORRECTION CONTROL")
    print("=" * 92)
    print(f"Stimuli:     {manifest['stimulus_version']}")
    print(f"Model:       {MODEL_NAME}")
    print("Temperature: 0")
    print("Fallbacks:   none")
    print("Model-visible filenames/roles: NO")
    print("Checks:      12")
    print()
    print(f"{'Claim':<11}{'Check':<28}{'Expected':<14}{'Observed':<14}{'Pass'}")
    print("-" * 92)

    for claim in claims:
        claim_id = claim["claim_id"]
        truth_label = bool(claim["truth_label"])
        query = claim["query"]
        other_id = mapping[claim_id]

        support = STIMULUS_ROOT / f"{claim_id}__support.txt"
        wrong_critical = STIMULUS_ROOT / f"{other_id}__critical.txt"

        checks = [
            (
                "wrong_entity_only",
                [wrong_critical],
                "INSUFFICIENT",
            ),
            (
                "support_plus_wrong_entity",
                [support, wrong_critical],
                opposite_truth(truth_label),
            ),
        ]

        for check_name, paths, expected in checks:
            blocks = [
                f"[Evidence {rank}]\n{read_text(path)}"
                for rank, path in enumerate(paths, start=1)
            ]
            prompt = (
                f"Claim:\n{query}\n\n"
                f"Evidence:\n{chr(10).join(blocks)}\n\n"
                "Return the evidential verdict."
            )

            assessment = llm.invoke_structured(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ],
                ClaimAssessment,
            )

            observed = assessment.verdict
            passed = observed == expected

            rows.append(
                {
                    "preflight": "exp3_wrong_entity_control",
                    "formal_experiment": False,
                    "stimulus_version": manifest["stimulus_version"],
                    "model_name": MODEL_NAME,
                    "claim_id": claim_id,
                    "wrong_entity_claim_id": other_id,
                    "check": check_name,
                    "expected": expected,
                    "observed": observed,
                    "pass": passed,
                    "model_visible_file_labels": False,
                }
            )

            print(
                f"{claim_id:<11}{check_name:<28}"
                f"{expected:<14}{observed:<14}{str(passed)}"
            )
            time.sleep(INTER_CALL_SECONDS)

    passed = sum(r["pass"] for r in rows)
    all_pass = passed == len(rows)

    detail_path = OUTPUT_ROOT / "wrong_entity_preflight.jsonl"
    with detail_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "preflight": "exp3_wrong_entity_control",
        "formal_experiment": False,
        "stimulus_version": manifest["stimulus_version"],
        "model_name": MODEL_NAME,
        "n_checks": len(rows),
        "passed_checks": passed,
        "all_checks_pass": all_pass,
        "model_visible_file_labels": False,
        "wrong_entity_map": mapping,
    }

    summary_path = OUTPUT_ROOT / "wrong_entity_preflight_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 92)
    print(f"PASSED: {passed}/{len(rows)}")
    print(f"ALL WRONG-ENTITY CHECKS PASS: {all_pass}")
    print(f"Summary: {summary_path}")

    if not all_pass:
        raise SystemExit(
            "Exp3 wrong-entity control failed preflight. Do not run formal E3."
        )


if __name__ == "__main__":
    main()
