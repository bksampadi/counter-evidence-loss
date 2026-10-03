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
EXP1_CONTEXTS_PATH = Path(
    "benchmark_results/Exp1/formal/experiment_1_contexts.jsonl"
)
STRICT_PREFLIGHT_PATH = Path(
    "benchmark_results/Exp1/preflight_strict/"
    "stimulus_preflight_strict_summary.json"
)
CORPUS_ROOT = Path("benchmarks/Exp1/corpora/formal_v1")
OUTPUT_ROOT = Path("benchmark_results/Exp2/formal")

CONDITIONS = (
    "exposed",
    "available_not_exposed",
    "absent",
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
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def assert_gates() -> tuple[dict, list[dict]]:
    exp1 = load_json(EXP1_SUMMARY_PATH)

    if exp1.get("formal_experiment") is not True:
        raise RuntimeError("E1 summary is not marked as formal.")

    if exp1.get("formal_acceptance_pass") is not True:
        raise RuntimeError(
            "Formal E1 acceptance is not True. Do not run E2."
        )

    strict = load_json(STRICT_PREFLIGHT_PATH)

    if strict.get("all_checks_pass") is not True:
        raise RuntimeError(
            "Strict no-label preflight did not pass. Do not run E2."
        )

    if strict.get("passed_checks") != 36:
        raise RuntimeError(
            "Formal E2 requires the frozen 36/36 strict preflight."
        )

    if strict.get("model_visible_file_labels") is not False:
        raise RuntimeError(
            "Strict preflight exposed file/role labels to the model."
        )

    if strict.get("stimulus_version") != exp1.get("stimulus_version"):
        raise RuntimeError(
            "Strict preflight and E1 stimulus versions do not match."
        )

    contexts = load_jsonl(EXP1_CONTEXTS_PATH)

    if len(contexts) != 18:
        raise RuntimeError(
            f"Formal E2 expects 18 frozen E1 contexts; got {len(contexts)}."
        )

    counts = defaultdict(int)
    for row in contexts:
        counts[row["experimental_state"]] += 1

        if row.get("manipulation_pass") is not True:
            raise RuntimeError(
                f"E1 context failed manipulation: "
                f"{row['claim_id']} / {row['experimental_state']}"
            )

        if row.get("support_visible") is not True:
            raise RuntimeError(
                f"Support not visible in frozen E1 context: "
                f"{row['claim_id']} / {row['experimental_state']}"
            )

    for condition in CONDITIONS:
        if counts[condition] != 6:
            raise RuntimeError(
                f"Expected 6 contexts for {condition}; got {counts[condition]}."
            )

    return exp1, contexts


def read_context(row: dict) -> str:
    corpus_state = row["corpus_state"]
    source_files = row["context_source_files"]

    if len(source_files) != 5:
        raise RuntimeError(
            f"{row['claim_id']} / {row['experimental_state']} "
            f"does not contain exactly 5 frozen context files."
        )

    blocks: list[str] = []
    for rank, filename in enumerate(source_files, start=1):
        path = CORPUS_ROOT / corpus_state / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Frozen context file missing: {path}"
            )

        # IMPORTANT: filename/role is intentionally NOT shown to the model.
        blocks.append(
            f"[Evidence {rank}]\n"
            f"{path.read_text(encoding='utf-8').strip()}"
        )

    return "\n\n".join(blocks)


def truth_to_verdict(truth_label: bool) -> Verdict:
    return "TRUE" if truth_label else "FALSE"


def accuracy(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(row["correct"] for row in rows) / len(rows)


def main() -> None:
    exp1, contexts = assert_gates()

    provider = build_groq_provider(
        model_name=MODEL_NAME,
        max_retries=MAX_RETRIES,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    llm = LLMService([provider])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Preserve a deterministic order: condition, then claim id.
    order = {condition: i for i, condition in enumerate(CONDITIONS)}
    contexts = sorted(
        contexts,
        key=lambda row: (
            order[row["experimental_state"]],
            row["claim_id"],
        ),
    )

    rows: list[dict] = []
    by_condition: dict[str, list[dict]] = defaultdict(list)

    print()
    print("FORMAL EXPERIMENT 2 — GENERATOR RESPONSE TO CORRECTIVE EVIDENCE")
    print("=" * 92)
    print(f"Stimuli:       {exp1['stimulus_version']}")
    print(f"Model:         {MODEL_NAME}")
    print("Temperature:   0")
    print("Fallbacks:     none")
    print("Context source: frozen Formal E1 contexts")
    print("Model-visible filenames/roles: NO")
    print("Cases:         18")
    print()

    current_condition = None

    for row in contexts:
        condition = row["experimental_state"]

        if condition != current_condition:
            current_condition = condition
            print(condition.upper())
            print("-" * 92)
            print(
                f"{'Claim':<12}"
                f"{'Truth':<10}"
                f"{'Verdict':<12}"
                f"{'Correct':<10}"
            )

        claim_id = row["claim_id"]
        truth_label = bool(row["truth_label"])
        query = row["query"]
        context = read_context(row)

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
        expected_verdict = truth_to_verdict(truth_label)
        correct = verdict == expected_verdict

        result = {
            "experiment": "formal_experiment_2",
            "formal_experiment": True,
            "stimulus_version": exp1["stimulus_version"],
            "model_provider": "groq",
            "model_name": MODEL_NAME,
            "temperature": 0,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "fallback_enabled": False,
            "model_visible_file_labels": False,
            "claim_id": claim_id,
            "truth_label": truth_label,
            "condition": condition,
            "corpus_state": row["corpus_state"],
            "query": query,
            "context_k": exp1["context_k"],
            # Audit metadata only; never shown to model.
            "context_source_files": row["context_source_files"],
            "critical_visible": row["critical_visible"],
            "support_visible": row["support_visible"],
            "verdict": verdict,
            "correct": correct,
        }

        rows.append(result)
        by_condition[condition].append(result)

        print(
            f"{claim_id:<12}"
            f"{str(truth_label):<10}"
            f"{verdict:<12}"
            f"{str(correct):<10}"
        )

        if len(by_condition[condition]) == 6:
            print(
                f"\nAccuracy: "
                f"{sum(r['correct'] for r in by_condition[condition])}"
                f"/{len(by_condition[condition])} "
                f"({accuracy(by_condition[condition]):.3f})\n"
            )

        time.sleep(INTER_CALL_SECONDS)

    detail_path = OUTPUT_ROOT / "experiment_2_generation.jsonl"
    with detail_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    indexed = {
        (row["claim_id"], row["condition"]): row
        for row in rows
    }
    claim_ids = sorted({row["claim_id"] for row in rows})

    contrasts = {}
    for left, right in (
        ("exposed", "available_not_exposed"),
        ("exposed", "absent"),
        ("available_not_exposed", "absent"),
    ):
        paired = []
        switch_count = 0
        left_correct_right_wrong = 0
        left_wrong_right_correct = 0

        for claim_id in claim_ids:
            a = indexed[(claim_id, left)]
            b = indexed[(claim_id, right)]

            switched = a["verdict"] != b["verdict"]
            deterioration = a["correct"] and not b["correct"]
            improvement = (not a["correct"]) and b["correct"]

            switch_count += switched
            left_correct_right_wrong += deterioration
            left_wrong_right_correct += improvement

            paired.append(
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
            "verdict_switch_count": switch_count,
            "verdict_switch_rate": switch_count / len(claim_ids),
            "left_correct_right_wrong_count": left_correct_right_wrong,
            "left_wrong_right_correct_count": left_wrong_right_correct,
            "pairs": paired,
        }

    rescue_rows = []
    for claim_id in claim_ids:
        exposed = indexed[(claim_id, "exposed")]
        hidden = indexed[(claim_id, "available_not_exposed")]
        absent = indexed[(claim_id, "absent")]

        rescue_rows.append(
            {
                "claim_id": claim_id,
                "exposed_correct": exposed["correct"],
                "available_not_exposed_correct": hidden["correct"],
                "absent_correct": absent["correct"],
                "corrective_exposure_rescue_vs_hidden": (
                    exposed["correct"] and not hidden["correct"]
                ),
                "corrective_exposure_rescue_vs_absent": (
                    exposed["correct"] and not absent["correct"]
                ),
            }
        )

    condition_summary = {}
    for condition in CONDITIONS:
        condition_rows = by_condition[condition]
        true_rows = [
            row for row in condition_rows if row["truth_label"]
        ]
        false_rows = [
            row for row in condition_rows if not row["truth_label"]
        ]

        condition_summary[condition] = {
            "n": len(condition_rows),
            "correct": sum(row["correct"] for row in condition_rows),
            "accuracy": accuracy(condition_rows),
            "true_claim_accuracy": accuracy(true_rows),
            "false_claim_accuracy": accuracy(false_rows),
        }

    summary = {
        "experiment": "formal_experiment_2",
        "formal_experiment": True,
        "stimulus_version": exp1["stimulus_version"],
        "model_provider": "groq",
        "model_name": MODEL_NAME,
        "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "fallback_enabled": False,
        "model_visible_file_labels": False,
        "context_source": str(EXP1_CONTEXTS_PATH),
        "context_k": exp1["context_k"],
        "n_claims": len(claim_ids),
        "n_cases": len(rows),
        "conditions": condition_summary,
        "contrasts": contrasts,
        "rescue_by_claim": rescue_rows,
        "rescue_rate_vs_available_not_exposed": (
            sum(
                row["corrective_exposure_rescue_vs_hidden"]
                for row in rescue_rows
            )
            / len(rescue_rows)
        ),
        "rescue_rate_vs_absent": (
            sum(
                row["corrective_exposure_rescue_vs_absent"]
                for row in rescue_rows
            )
            / len(rescue_rows)
        ),
    }

    summary_path = OUTPUT_ROOT / "experiment_2_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 92)
    print("CONDITION SUMMARY")
    print("-" * 92)
    for condition in CONDITIONS:
        item = condition_summary[condition]
        print(
            f"{condition:<24}"
            f"{item['correct']}/{item['n']} correct"
            f" | accuracy={item['accuracy']:.3f}"
            f" | true={item['true_claim_accuracy']:.3f}"
            f" | false={item['false_claim_accuracy']:.3f}"
        )

    print()
    print("PAIRED CONTRASTS")
    print("-" * 92)
    for name, item in contrasts.items():
        print(
            f"{name:<42}"
            f"switches={item['verdict_switch_count']}/{item['n_claims']} "
            f"({item['verdict_switch_rate']:.3f})"
            f" | correct->wrong={item['left_correct_right_wrong_count']}"
            f" | wrong->correct={item['left_wrong_right_correct_count']}"
        )

    print()
    print(
        "Corrective-exposure rescue vs available_not_exposed: "
        f"{summary['rescue_rate_vs_available_not_exposed']:.3f}"
    )
    print(
        "Corrective-exposure rescue vs absent:                "
        f"{summary['rescue_rate_vs_absent']:.3f}"
    )
    print()
    print(f"Detailed results: {detail_path}")
    print(f"Summary:          {summary_path}")


if __name__ == "__main__":
    main()
