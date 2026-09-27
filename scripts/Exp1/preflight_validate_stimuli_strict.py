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
OUTPUT_ROOT = Path("benchmark_results/Exp1/preflight_strict")

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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def expected_truth(truth_label: bool) -> Verdict:
    return "TRUE" if truth_label else "FALSE"


def opposite_truth(truth_label: bool) -> Verdict:
    return "FALSE" if truth_label else "TRUE"


def build_checks(
    claim_id: str,
    truth_label: bool,
) -> list[tuple[str, list[Path], Verdict]]:
    support = STIMULUS_ROOT / f"{claim_id}__support.txt"
    critical = STIMULUS_ROOT / f"{claim_id}__critical.txt"
    replacement = STIMULUS_ROOT / f"{claim_id}__replacement.txt"
    neutrals = [
        STIMULUS_ROOT / f"{claim_id}__neutral_{i}.txt"
        for i in range(1, 5)
    ]

    return [
        ("support_only", [support], opposite_truth(truth_label)),
        ("critical_only", [critical], expected_truth(truth_label)),
        ("replacement_only", [replacement], "INSUFFICIENT"),
        ("distractors_only", neutrals, "INSUFFICIENT"),
        (
            "support_plus_distractors",
            [support, *neutrals],
            opposite_truth(truth_label),
        ),
        (
            "critical_plus_distractors",
            [critical, *neutrals],
            expected_truth(truth_label),
        ),
    ]


def main() -> None:
    manifest = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    claims = manifest["claims"]

    provider = build_groq_provider(
        model_name=MODEL_NAME,
        max_retries=MAX_RETRIES,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    llm = LLMService([provider])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print()
    print("STRICT EXP1 PREFLIGHT — NO MODEL-VISIBLE FILE/ROLE LABELS")
    print("=" * 92)
    print(f"Stimuli:     {manifest['stimulus_version']}")
    print(f"Model:       {MODEL_NAME}")
    print("Temperature: 0")
    print("Fallbacks:   none")
    print("Labels shown to model: neutral Evidence 1, Evidence 2, ... only")
    print(f"Checks:      {len(claims) * 6}")
    print()
    print(f"{'Claim':<11}{'Check':<29}{'Expected':<14}{'Observed':<14}{'Pass'}")
    print("-" * 92)

    for claim in claims:
        claim_id = claim["claim_id"]
        truth_label = bool(claim["truth_label"])
        query = claim["query"]

        for check_name, paths, expected in build_checks(
            claim_id,
            truth_label,
        ):
            # CRITICAL ANTI-LEAK RULE:
            # model sees only neutral ordinal labels, never filenames or roles.
            blocks = [
                f"[Evidence {rank}]\n{read_text(path)}"
                for rank, path in enumerate(paths, start=1)
            ]
            context = "\n\n".join(blocks)

            prompt = (
                f"Claim:\n{query}\n\n"
                f"Evidence:\n{context}\n\n"
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
                    "preflight": "exp1_strict_anti_leak_validation",
                    "formal_experiment": False,
                    "stimulus_version": manifest["stimulus_version"],
                    "model_provider": "groq",
                    "model_name": MODEL_NAME,
                    "temperature": 0,
                    "claim_id": claim_id,
                    "truth_label": truth_label,
                    "query": query,
                    "check": check_name,
                    # Filenames are stored for audit only; never sent to model.
                    "evidence_files": [p.name for p in paths],
                    "expected": expected,
                    "observed": observed,
                    "pass": passed,
                    "model_visible_file_labels": False,
                }
            )

            print(
                f"{claim_id:<11}"
                f"{check_name:<29}"
                f"{expected:<14}"
                f"{observed:<14}"
                f"{str(passed)}"
            )
            time.sleep(INTER_CALL_SECONDS)

    passed_count = sum(row["pass"] for row in rows)
    all_pass = passed_count == len(rows)

    detail_path = OUTPUT_ROOT / "stimulus_preflight_strict.jsonl"
    with detail_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "preflight": "exp1_strict_anti_leak_validation",
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
    }

    summary_path = OUTPUT_ROOT / "stimulus_preflight_strict_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 92)
    print(f"PASSED: {passed_count}/{len(rows)}")
    print(f"ALL STRICT CHECKS PASS: {all_pass}")
    print(f"Summary: {summary_path}")

    if not all_pass:
        raise SystemExit(
            "Strict preflight failed. Do NOT run formal Experiment 2."
        )


if __name__ == "__main__":
    main()
