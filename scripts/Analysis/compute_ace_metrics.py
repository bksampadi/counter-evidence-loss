import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


EXP1_CONTEXTS = (
    ROOT
    / "benchmark_results"
    / "Exp1"
    / "formal"
    / "experiment_1_contexts.jsonl"
)

EXP3_RESULTS = (
    ROOT
    / "benchmark_results"
    / "Exp3"
    / "formal"
    / "experiment_3_generation.jsonl"
)

OUTPUT_DIR = ROOT / "benchmark_results" / "ace_metrics"
RESULTS_PATH = OUTPUT_DIR / "ace_metrics.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "ace_metrics_summary.json"

METRIC_VERSION = "ace-v0.1"

EXP3_ACE_AVAILABLE = {
    "corrective_exposed_k2": True,
    "corrective_below_k1": True,
    "corrective_absent_k1": False,
    "wrong_entity_control_k2": True,
}

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def reference_ace_file(claim_id: str) -> str:
    return f"{claim_id}__critical.txt"


def compute_exp1() -> list[dict[str, Any]]:
    rows = load_jsonl(EXP1_CONTEXTS)
    results: list[dict[str, Any]] = []

    for row in rows:
        claim_id = row["claim_id"]
        ace_file = reference_ace_file(claim_id)

        context_files = row["context_source_files"]
        context_k = len(context_files)

        ace_available = bool(row["critical_present_in_corpus"])
        ace_exposed = ace_file in context_files

        if ace_exposed != bool(row["critical_visible"]):
            raise ValueError(
                f"Exp1 exposure mismatch for "
                f"{claim_id}/{row['experimental_state']}"
            )

        if ace_exposed and not ace_available:
            raise ValueError(
                f"Exp1 ACE exposed while unavailable for "
                f"{claim_id}/{row['experimental_state']}"
            )

        results.append(
            {
                "metric_version": METRIC_VERSION,
                "experiment": "Exp1",
                "claim_id": claim_id,
                "condition": row["experimental_state"],
                "context_k": context_k,
                "reference_ace_file": ace_file,
                "context_source_files": context_files,
                "ace_available": ace_available,
                "ace_exposed": ace_exposed,
                "ace_a": float(ace_available),
                "ace_e_at_k": float(ace_exposed),
            }
        )

    return results


def compute_exp3() -> list[dict[str, Any]]:
    rows = load_jsonl(EXP3_RESULTS)
    results: list[dict[str, Any]] = []

    for row in rows:
        claim_id = row["claim_id"]
        condition = row["condition"]

        if condition not in EXP3_ACE_AVAILABLE:
            raise ValueError(
                f"Unexpected Exp3 condition: {condition}"
            )

        ace_file = reference_ace_file(claim_id)
        context_files = row["context_source_files"]

        ace_available = EXP3_ACE_AVAILABLE[condition]
        ace_exposed = ace_file in context_files

        if ace_exposed and not ace_available:
            raise ValueError(
                f"Exp3 ACE exposed while unavailable for "
                f"{claim_id}/{condition}"
            )

        results.append(
            {
                "metric_version": METRIC_VERSION,
                "experiment": "Exp3",
                "claim_id": claim_id,
                "condition": condition,
                "context_k": row["context_k"],
                "reference_ace_file": ace_file,
                "context_source_files": context_files,
                "wrong_entity_claim_id": row[
                    "wrong_entity_claim_id"
                ],
                "ace_available": ace_available,
                "ace_exposed": ace_exposed,
                "ace_a": float(ace_available),
                "ace_e_at_k": float(ace_exposed),
                "correct": bool(row["correct"]),
            }
        )

    return results


def summarize(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in results:
        grouped[
            (row["experiment"], row["condition"])
        ].append(row)

    by_condition: dict[str, Any] = {}

    for experiment, condition in sorted(grouped):
        rows = grouped[(experiment, condition)]

        key = f"{experiment}:{condition}"

        entry: dict[str, Any] = {
            "experiment": experiment,
            "condition": condition,
            "n": len(rows),
            "context_k_values": sorted(
                {row["context_k"] for row in rows}
            ),
            "mean_ace_a": fmean(
                row["ace_a"]
                for row in rows
            ),
            "mean_ace_e_at_k": fmean(
                row["ace_e_at_k"]
                for row in rows
            ),
        }

        if all("correct" in row for row in rows):
            entry["accuracy"] = fmean(
                float(row["correct"])
                for row in rows
            )

        by_condition[key] = entry

    return {
        "metric_version": METRIC_VERSION,
        "construct": "Answer-Changing Evidence (ACE)",
        "definitions": {
            "reference_ace": (
                "The claim-matched critical passage designated "
                "by the benchmark and shown by the controlled "
                "interventions to change the warranted answer."
            ),
            "ace_a": (
                "ACE Availability: whether the reference "
                "answer-changing evidence exists in the corpus "
                "for the evaluated condition."
            ),
            "ace_e_at_k": (
                "ACE Exposure at k: whether the reference "
                "answer-changing evidence appears in the "
                "generator-visible context at depth k."
            ),
        },
        "important_note": (
            "Correctness is not used to compute ACE-A or "
            "ACE-E@k. Exp3 correctness is reported only as "
            "construct validation."
        ),
        "n_cases": len(results),
        "by_condition": by_condition,
    }


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True)
                + "\n"
            )


def main() -> None:
    results = [
        *compute_exp1(),
        *compute_exp3(),
    ]

    summary = summarize(results)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        RESULTS_PATH,
        results,
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print("ANSWER-CHANGING EVIDENCE METRICS")
    print("=" * 88)

    for values in summary[
        "by_condition"
    ].values():
        accuracy = values.get("accuracy")
        accuracy_text = (
            f"  accuracy={accuracy:.3f}"
            if accuracy is not None
            else ""
        )

        print(
            f"{values['experiment']:4}  "
            f"{values['condition']:26} "
            f"n={values['n']}  "
            f"ACE-A={values['mean_ace_a']:.3f}  "
            f"ACE-E@k={values['mean_ace_e_at_k']:.3f}  "
            f"k={values['context_k_values']}"
            f"{accuracy_text}"
        )

    print()
    print(
        "Wrote:",
        RESULTS_PATH.relative_to(ROOT),
    )
    print(
        "Wrote:",
        SUMMARY_PATH.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()