import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from signalrank.components.llm.providers.groq import build_groq_provider
from signalrank.services.llm_service import LLMService

EXP1_SUMMARY_PATH = Path(
    "benchmark_results/Exp1/formal/experiment_1_summary.json"
)
EXP2_SUMMARY_PATH = Path(
    "benchmark_results/Exp2/formal/experiment_2_summary.json"
)
EXP3_PREFLIGHT_PATH = Path(
    "benchmark_results/Exp3/preflight/wrong_entity_preflight_summary.json"
)
STIMULUS_ROOT = Path("benchmarks/Exp1/stimuli_v1")
AVAILABLE_CORPUS = Path("benchmarks/Exp1/corpora/formal_v1/available")
ABSENT_CORPUS = Path("benchmarks/Exp1/corpora/formal_v1/absent")
OUTPUT_ROOT = Path("benchmark_results/Exp3/formal")

CONDITIONS = (
    "corrective_exposed_k2",
    "corrective_below_k1",
    "corrective_absent_k1",
    "wrong_entity_control_k2",
)

MODEL_NAME = "openai/gpt-oss-20b"
MAX_RETRIES = 0
MAX_OUTPUT_TOKENS = 1024
INTER_CALL_SECONDS = 10

SYSTEM_PROMPT = """You are a claim-verification system.

Judge the claim using ONLY the supplied evidence.
Do not use outside knowledge.

Return TRUE if the supplied evidence supports the claim.
Return FALSE if the supplied evidence contradicts the claim.

You must return one binary verdict.
"""

Verdict = Literal["TRUE", "FALSE"]


class ClaimAssessment(BaseModel):
    verdict: Verdict


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def truth_to_verdict(truth_label: bool) -> Verdict:
    return "TRUE" if truth_label else "FALSE"


def opposite_truth(truth_label: bool) -> Verdict:
    return "FALSE" if truth_label else "TRUE"


def accuracy(rows: list[dict]) -> float:
    return sum(r["correct"] for r in rows) / len(rows) if rows else 0.0


def assert_gates() -> tuple[dict, dict, dict]:
    exp1 = load_json(EXP1_SUMMARY_PATH)
    exp2 = load_json(EXP2_SUMMARY_PATH)
    preflight = load_json(EXP3_PREFLIGHT_PATH)

    if exp1.get("formal_acceptance_pass") is not True:
        raise RuntimeError("Formal E1 acceptance is not True.")

    if exp2.get("formal_experiment") is not True:
        raise RuntimeError("Formal E2 summary is missing/not formal.")

    if preflight.get("all_checks_pass") is not True:
        raise RuntimeError("Exp3 wrong-entity preflight did not pass.")

    if preflight.get("passed_checks") != 12:
        raise RuntimeError("Formal E3 requires 12/12 wrong-entity checks.")

    stimulus_version = exp1["stimulus_version"]
    if exp2.get("stimulus_version") != stimulus_version:
        raise RuntimeError("E1/E2 stimulus versions do not match.")
    if preflight.get("stimulus_version") != stimulus_version:
        raise RuntimeError("E1/E3-preflight stimulus versions do not match.")

    # Formal E1 established the exact boundary needed for E3:
    # support rank 1 and corrective rank 2 in all six available-corpus cases.
    for row in exp1["states"]["exposed"]["claims"]:
        if row["support_rank"] != 1 or row["critical_rank"] != 2:
            raise RuntimeError(
                "Formal E3 requires support rank 1 and corrective rank 2 "
                f"for every claim; failed at {row['claim_id']}."
            )

    return exp1, exp2, preflight


def build_conditions(
    claim: dict,
    wrong_entity_id: str,
) -> dict[str, list[Path]]:
    claim_id = claim["claim_id"]

    support_available = AVAILABLE_CORPUS / f"{claim_id}__support.txt"
    critical_available = AVAILABLE_CORPUS / f"{claim_id}__critical.txt"
    support_absent = ABSENT_CORPUS / f"{claim_id}__support.txt"
    wrong_critical = AVAILABLE_CORPUS / f"{wrong_entity_id}__critical.txt"

    return {
        # k=2 crosses the formal E1 boundary and exposes the corrective passage.
        "corrective_exposed_k2": [
            support_available,
            critical_available,
        ],
        # Same available corpus: corrective passage exists at rank 2, but k=1.
        "corrective_below_k1": [
            support_available,
        ],
        # Same k=1 context, but corrective passage is genuinely absent from corpus.
        "corrective_absent_k1": [
            support_absent,
        ],
        # Control for generic "correction-shaped" wording:
        # target support + another entity's corrective passage.
        "wrong_entity_control_k2": [
            support_available,
            wrong_critical,
        ],
    }


def main() -> None:
    exp1, exp2, preflight = assert_gates()

    claims = exp1["states"]["exposed"]["claims"]
    claims = sorted(claims, key=lambda x: x["claim_id"])
    wrong_map = preflight["wrong_entity_map"]

    provider = build_groq_provider(
        model_name=MODEL_NAME,
        max_retries=MAX_RETRIES,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    llm = LLMService([provider])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    by_condition: dict[str, list[dict]] = defaultdict(list)

    print()
    print("FORMAL EXPERIMENT 3 — RETRIEVAL CUTOFF / BOUNDARY ROBUSTNESS")
    print("=" * 96)
    print(f"Stimuli:       {exp1['stimulus_version']}")
    print(f"Model:         {MODEL_NAME}")
    print("Temperature:   0")
    print("Fallbacks:     none")
    print("Formal E1 boundary: support rank 1, corrective rank 2 for all claims")
    print("Model-visible filenames/roles: NO")
    print("Claims:        6")
    print("Conditions:    4")
    print("Cases:         24")
    print()

    for condition in CONDITIONS:
        print(condition.upper())
        print("-" * 96)
        print(
            f"{'Claim':<12}"
            f"{'Truth':<10}"
            f"{'Verdict':<12}"
            f"{'Correct':<10}"
        )

        for claim in claims:
            claim_id = claim["claim_id"]
            truth_label = bool(claim["truth_label"])
            query = claim["query"]
            wrong_entity_id = wrong_map[claim_id]

            paths = build_conditions(claim, wrong_entity_id)[condition]

            for path in paths:
                if not path.exists():
                    raise FileNotFoundError(f"Missing E3 context file: {path}")

            blocks = [
                f"[Evidence {rank}]\n{read_text(path)}"
                for rank, path in enumerate(paths, start=1)
            ]
            context = "\n\n".join(blocks)

            prompt = (
                f"Claim:\n{query}\n\n"
                f"Evidence:\n{context}\n\n"
                "Determine whether the claim is TRUE or FALSE "
                "based only on the supplied evidence."
            )

            assessment = llm.invoke_structured(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ],
                ClaimAssessment,
            )

            verdict = assessment.verdict
            correct = verdict == truth_to_verdict(truth_label)

            row = {
                "experiment": "formal_experiment_3",
                "formal_experiment": True,
                "stimulus_version": exp1["stimulus_version"],
                "model_provider": "groq",
                "model_name": MODEL_NAME,
                "temperature": 0,
                "fallback_enabled": False,
                "model_visible_file_labels": False,
                "claim_id": claim_id,
                "truth_label": truth_label,
                "query": query,
                "condition": condition,
                "context_k": len(paths),
                "context_source_files": [p.name for p in paths],
                "wrong_entity_claim_id": (
                    wrong_entity_id
                    if condition == "wrong_entity_control_k2"
                    else None
                ),
                "verdict": verdict,
                "correct": correct,
            }

            rows.append(row)
            by_condition[condition].append(row)

            print(
                f"{claim_id:<12}"
                f"{str(truth_label):<10}"
                f"{verdict:<12}"
                f"{str(correct):<10}"
            )

            time.sleep(INTER_CALL_SECONDS)

        cond_rows = by_condition[condition]
        print(
            f"\nAccuracy: "
            f"{sum(r['correct'] for r in cond_rows)}/{len(cond_rows)} "
            f"({accuracy(cond_rows):.3f})\n"
        )

    detail_path = OUTPUT_ROOT / "experiment_3_generation.jsonl"
    with detail_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    indexed = {
        (r["claim_id"], r["condition"]): r
        for r in rows
    }
    claim_ids = sorted({r["claim_id"] for r in rows})

    contrasts = {}
    for left, right in (
        ("corrective_exposed_k2", "corrective_below_k1"),
        ("corrective_below_k1", "corrective_absent_k1"),
        ("corrective_exposed_k2", "wrong_entity_control_k2"),
        ("corrective_below_k1", "wrong_entity_control_k2"),
    ):
        switches = 0
        left_correct_right_wrong = 0
        pairs = []

        for claim_id in claim_ids:
            a = indexed[(claim_id, left)]
            b = indexed[(claim_id, right)]
            switched = a["verdict"] != b["verdict"]
            deterioration = a["correct"] and not b["correct"]

            switches += switched
            left_correct_right_wrong += deterioration
            pairs.append(
                {
                    "claim_id": claim_id,
                    "left_verdict": a["verdict"],
                    "right_verdict": b["verdict"],
                    "verdict_switched": switched,
                    "left_correct": a["correct"],
                    "right_correct": b["correct"],
                }
            )

        contrasts[f"{left}_vs_{right}"] = {
            "n_claims": len(claim_ids),
            "verdict_switch_count": switches,
            "verdict_switch_rate": switches / len(claim_ids),
            "left_correct_right_wrong_count": left_correct_right_wrong,
            "pairs": pairs,
        }

    condition_summary = {}
    for condition in CONDITIONS:
        cond_rows = by_condition[condition]
        true_rows = [r for r in cond_rows if r["truth_label"]]
        false_rows = [r for r in cond_rows if not r["truth_label"]]

        condition_summary[condition] = {
            "n": len(cond_rows),
            "correct": sum(r["correct"] for r in cond_rows),
            "accuracy": accuracy(cond_rows),
            "true_claim_accuracy": accuracy(true_rows),
            "false_claim_accuracy": accuracy(false_rows),
        }

    summary = {
        "experiment": "formal_experiment_3",
        "formal_experiment": True,
        "stimulus_version": exp1["stimulus_version"],
        "model_provider": "groq",
        "model_name": MODEL_NAME,
        "temperature": 0,
        "fallback_enabled": False,
        "n_claims": len(claim_ids),
        "n_cases": len(rows),
        "boundary": {
            "support_rank": 1,
            "corrective_rank": 2,
            "below_cutoff_k": 1,
            "exposed_cutoff_k": 2,
        },
        "conditions": condition_summary,
        "contrasts": contrasts,
    }

    summary_path = OUTPUT_ROOT / "experiment_3_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 96)
    print("CONDITION SUMMARY")
    print("-" * 96)
    for condition in CONDITIONS:
        item = condition_summary[condition]
        print(
            f"{condition:<30}"
            f"{item['correct']}/{item['n']} correct"
            f" | accuracy={item['accuracy']:.3f}"
            f" | true={item['true_claim_accuracy']:.3f}"
            f" | false={item['false_claim_accuracy']:.3f}"
        )

    print()
    print("PAIRED CONTRASTS")
    print("-" * 96)
    for name, item in contrasts.items():
        print(
            f"{name:<64}"
            f"switches={item['verdict_switch_count']}/{item['n_claims']} "
            f"({item['verdict_switch_rate']:.3f})"
            f" | correct->wrong={item['left_correct_right_wrong_count']}"
        )

    print()
    print(f"Detailed results: {detail_path}")
    print(f"Summary:          {summary_path}")


if __name__ == "__main__":
    main()
