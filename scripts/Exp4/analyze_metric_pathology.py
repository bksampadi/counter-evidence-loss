import csv
import json
from collections import defaultdict
from pathlib import Path

EXP1_SUMMARY_PATH = Path(
    "benchmark_results/Exp1/formal/experiment_1_summary.json"
)
EXP2_RESULTS_PATH = Path(
    "benchmark_results/Exp2/formal/experiment_2_generation.jsonl"
)
EXP3_RESULTS_PATH = Path(
    "benchmark_results/Exp3/formal/experiment_3_generation.jsonl"
)
OUTPUT_ROOT = Path(
    "benchmark_results/metric_pathology"
)

E2_CONDITIONS = (
    "exposed",
    "available_not_exposed",
    "absent",
)

E3_CONDITIONS = (
    "corrective_exposed_k2",
    "corrective_below_k1",
    "corrective_absent_k1",
    "wrong_entity_control_k2",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def truth_verdict(truth_label: bool) -> str:
    return "TRUE" if truth_label else "FALSE"


def misleading_verdict(truth_label: bool) -> str:
    return "FALSE" if truth_label else "TRUE"


def build_e1_index(exp1: dict) -> dict[tuple[str, str], dict]:
    index: dict[tuple[str, str], dict] = {}

    for state_name, state in exp1["states"].items():
        for row in state["claims"]:
            index[(row["claim_id"], state_name)] = row

    return index


def target_evidence_metrics(
    *,
    truth_label: bool,
    generated_verdict: str,
    support_visible: bool,
    critical_visible: bool,
) -> dict:
    """
    Deterministic metrics for the synthetic evidence design.

    The target support passage always supports the deliberately misleading
    verdict. The target critical passage always supports the ground-truth
    verdict.
    """
    support_for_answer = (
        support_visible
        and generated_verdict == misleading_verdict(truth_label)
    )
    critical_for_answer = (
        critical_visible
        and generated_verdict == truth_verdict(truth_label)
    )

    visible_target_count = int(support_visible) + int(critical_visible)
    agreeing_target_count = int(support_for_answer) + int(critical_for_answer)

    context_support_indicator = agreeing_target_count > 0

    visible_target_evidence_agreement = (
        agreeing_target_count / visible_target_count
        if visible_target_count
        else None
    )

    target_conflict_present = support_visible and critical_visible

    # Reference-anchored sufficiency: does admitted context contain the
    # target passage that establishes ground truth?
    reference_sufficient_context = critical_visible

    return {
        "context_support_indicator": context_support_indicator,
        "visible_target_evidence_agreement": (
            visible_target_evidence_agreement
        ),
        "target_conflict_present": target_conflict_present,
        "reference_sufficient_context": reference_sufficient_context,
    }


def make_e2_rows(
    *,
    exp1: dict,
    exp2_rows: list[dict],
) -> list[dict]:
    e1_index = build_e1_index(exp1)
    rows: list[dict] = []

    for row in exp2_rows:
        condition = row["condition"]
        key = (row["claim_id"], condition)
        e1 = e1_index[key]

        generated_verdict = row["verdict"]
        truth_label = bool(row["truth_label"])

        metrics = target_evidence_metrics(
            truth_label=truth_label,
            generated_verdict=generated_verdict,
            support_visible=bool(e1["support_visible"]),
            critical_visible=bool(e1["critical_visible"]),
        )

        rows.append(
            {
                "analysis": "metric_pathology",
                "source_experiment": "E2",
                "claim_id": row["claim_id"],
                "truth_label": truth_label,
                "condition": condition,
                "generated_verdict": generated_verdict,
                "reference_correct": bool(row["correct"]),
                # Candidate-set metric: target correction was successfully
                # retrieved in the original top-5 ranking.
                "candidate_corrective_recall_at_5": bool(
                    e1["critical_originally_in_top_5"]
                ),
                # Generation-context metric: target correction actually crossed
                # the context boundary.
                "context_corrective_recall": bool(
                    e1["critical_visible"]
                ),
                "support_visible": bool(e1["support_visible"]),
                **metrics,
            }
        )

    return rows


def make_e3_rows(
    *,
    exp1: dict,
    exp3_rows: list[dict],
) -> list[dict]:
    e1_exposed = {
        row["claim_id"]: row
        for row in exp1["states"]["exposed"]["claims"]
    }
    rows: list[dict] = []

    for row in exp3_rows:
        condition = row["condition"]
        claim_id = row["claim_id"]
        truth_label = bool(row["truth_label"])
        generated_verdict = row["verdict"]

        # Formal E1 established that the target correction is rank 2 in the
        # available corpus for all six claims.
        target_candidate_recalled = bool(
            e1_exposed[claim_id]["critical_originally_in_top_5"]
        )

        if condition == "corrective_exposed_k2":
            support_visible = True
            critical_visible = True
            candidate_corrective_recall = target_candidate_recalled

        elif condition == "corrective_below_k1":
            support_visible = True
            critical_visible = False
            candidate_corrective_recall = target_candidate_recalled

        elif condition == "corrective_absent_k1":
            support_visible = True
            critical_visible = False
            candidate_corrective_recall = False

        elif condition == "wrong_entity_control_k2":
            # Target correction exists in the available retrieval environment,
            # but the constructed context contains target support plus another
            # claim's critical passage. The target corrective passage itself is
            # not visible.
            support_visible = True
            critical_visible = False
            candidate_corrective_recall = target_candidate_recalled

        else:
            raise ValueError(f"Unknown E3 condition: {condition}")

        metrics = target_evidence_metrics(
            truth_label=truth_label,
            generated_verdict=generated_verdict,
            support_visible=support_visible,
            critical_visible=critical_visible,
        )

        rows.append(
            {
                "analysis": "metric_pathology",
                "source_experiment": "E3",
                "claim_id": claim_id,
                "truth_label": truth_label,
                "condition": condition,
                "generated_verdict": generated_verdict,
                "reference_correct": bool(row["correct"]),
                "candidate_corrective_recall_at_5": (
                    candidate_corrective_recall
                ),
                "context_corrective_recall": critical_visible,
                "support_visible": support_visible,
                **metrics,
            }
        )

    return rows


def mean_bool(rows: list[dict], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def mean_nullable_float(rows: list[dict], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row[key] is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def summarize(
    rows: list[dict],
    condition_order: tuple[str, ...],
) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        grouped[row["condition"]].append(row)

    summary: dict[str, dict] = {}

    for condition in condition_order:
        condition_rows = grouped[condition]

        summary[condition] = {
            "n": len(condition_rows),
            "reference_accuracy": mean_bool(
                condition_rows,
                "reference_correct",
            ),
            "candidate_corrective_recall_at_5": mean_bool(
                condition_rows,
                "candidate_corrective_recall_at_5",
            ),
            "context_corrective_recall": mean_bool(
                condition_rows,
                "context_corrective_recall",
            ),
            "context_support_rate": mean_bool(
                condition_rows,
                "context_support_indicator",
            ),
            "visible_target_evidence_agreement": mean_nullable_float(
                condition_rows,
                "visible_target_evidence_agreement",
            ),
            "reference_sufficient_context_rate": mean_bool(
                condition_rows,
                "reference_sufficient_context",
            ),
            "target_conflict_rate": mean_bool(
                condition_rows,
                "target_conflict_present",
            ),
        }

    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def print_summary(
    title: str,
    summary: dict[str, dict],
    condition_order: tuple[str, ...],
) -> None:
    print()
    print(title)
    print("=" * 132)
    print(
        f"{'Condition':<34}"
        f"{'n':>4}"
        f"{'RefAcc':>10}"
        f"{'CandCorr':>10}"
        f"{'CtxCorr':>10}"
        f"{'CtxSupport':>12}"
        f"{'Agree':>10}"
        f"{'RefSuff':>10}"
        f"{'Conflict':>10}"
    )
    print("-" * 132)

    for condition in condition_order:
        item = summary[condition]
        print(
            f"{condition:<34}"
            f"{item['n']:>4}"
            f"{fmt(item['reference_accuracy']):>10}"
            f"{fmt(item['candidate_corrective_recall_at_5']):>10}"
            f"{fmt(item['context_corrective_recall']):>10}"
            f"{fmt(item['context_support_rate']):>12}"
            f"{fmt(item['visible_target_evidence_agreement']):>10}"
            f"{fmt(item['reference_sufficient_context_rate']):>10}"
            f"{fmt(item['target_conflict_rate']):>10}"
        )


def main() -> None:
    exp1 = load_json(EXP1_SUMMARY_PATH)
    exp2 = load_jsonl(EXP2_RESULTS_PATH)
    exp3 = load_jsonl(EXP3_RESULTS_PATH)

    if exp1.get("formal_acceptance_pass") is not True:
        raise RuntimeError("Formal E1 acceptance is not True.")

    if len(exp2) != 18:
        raise RuntimeError(
            f"Expected 18 formal E2 rows, got {len(exp2)}."
        )

    if len(exp3) != 24:
        raise RuntimeError(
            f"Expected 24 formal E3 rows, got {len(exp3)}."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    e2_rows = make_e2_rows(
        exp1=exp1,
        exp2_rows=exp2,
    )
    e3_rows = make_e3_rows(
        exp1=exp1,
        exp3_rows=exp3,
    )

    e2_summary = summarize(e2_rows, E2_CONDITIONS)
    e3_summary = summarize(e3_rows, E3_CONDITIONS)

    print()
    print("METRIC-PATHOLOGY ANALYSIS")
    print("=" * 132)
    print(
        "Deterministic analysis of the frozen formal E1-E3 outputs. "
        "No LLM judge and no RAGAS calls."
    )
    print()
    print("Metric definitions:")
    print(
        "  RefAcc      = generated verdict matches known reference truth."
    )
    print(
        "  CandCorr    = target corrective passage was retrieved in the "
        "original top-5 candidate ranking."
    )
    print(
        "  CtxCorr     = target corrective passage actually entered the "
        "generation context."
    )
    print(
        "  CtxSupport  = at least one visible target passage supports the "
        "generated verdict."
    )
    print(
        "  Agree       = fraction of visible target adjudicative passages "
        "that agree with the generated verdict."
    )
    print(
        "  RefSuff     = admitted context contains the target corrective "
        "passage that establishes reference truth."
    )
    print(
        "  Conflict    = both target misleading support and target correction "
        "are visible."
    )

    print_summary(
        "E2 — AVAILABILITY / EXPOSURE PATHOLOGY",
        e2_summary,
        E2_CONDITIONS,
    )
    print_summary(
        "E3 — RETRIEVAL-CUTOFF REPLICATION",
        e3_summary,
        E3_CONDITIONS,
    )

    # Explicitly test the central pathology in AVAILABLE_NOT_EXPOSED.
    hidden = e2_summary["available_not_exposed"]

    pathology_pass = (
        hidden["candidate_corrective_recall_at_5"] == 1.0
        and hidden["context_corrective_recall"] == 0.0
        and hidden["context_support_rate"] == 1.0
        and hidden["visible_target_evidence_agreement"] == 1.0
        and hidden["reference_accuracy"] == 0.0
    )

    # E3 replication at the natural cutoff.
    below_k = e3_summary["corrective_below_k1"]

    boundary_pathology_pass = (
        below_k["candidate_corrective_recall_at_5"] == 1.0
        and below_k["context_corrective_recall"] == 0.0
        and below_k["context_support_rate"] == 1.0
        and below_k["visible_target_evidence_agreement"] == 1.0
        and below_k["reference_accuracy"] == 0.0
    )

    summary = {
        "analysis": "metric_pathology",
        "formal_experiment": False,
        "analysis_of_frozen_formal_outputs": True,
        "stimulus_version": exp1["stimulus_version"],
        "e2": e2_summary,
        "e3": e3_summary,
        "central_pathology": {
            "condition": "E2_available_not_exposed",
            "candidate_corrective_recall_at_5": (
                hidden["candidate_corrective_recall_at_5"]
            ),
            "context_corrective_recall": (
                hidden["context_corrective_recall"]
            ),
            "context_support_rate": hidden["context_support_rate"],
            "visible_target_evidence_agreement": (
                hidden["visible_target_evidence_agreement"]
            ),
            "reference_accuracy": hidden["reference_accuracy"],
            "pathology_pattern_present": pathology_pass,
        },
        "retrieval_boundary_replication": {
            "condition": "E3_corrective_below_k1",
            "candidate_corrective_recall_at_5": (
                below_k["candidate_corrective_recall_at_5"]
            ),
            "context_corrective_recall": (
                below_k["context_corrective_recall"]
            ),
            "context_support_rate": below_k["context_support_rate"],
            "visible_target_evidence_agreement": (
                below_k["visible_target_evidence_agreement"]
            ),
            "reference_accuracy": below_k["reference_accuracy"],
            "pathology_pattern_present": boundary_pathology_pass,
        },
        "interpretation": (
            "In the hidden/below-cutoff conditions, retrieval successfully "
            "located the target corrective evidence, and the generated wrong "
            "verdict remained fully supported by the visible target evidence. "
            "Yet the corrective passage did not cross the context boundary, "
            "reference sufficiency failed, and reference accuracy was zero."
        ),
    }

    summary_path = OUTPUT_ROOT / "metric_pathology_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    e2_detail_path = OUTPUT_ROOT / "metric_pathology_e2.csv"
    e3_detail_path = OUTPUT_ROOT / "metric_pathology_e3.csv"

    write_csv(e2_detail_path, e2_rows)
    write_csv(e3_detail_path, e3_rows)

    manuscript_table = OUTPUT_ROOT / "metric_pathology_table.md"
    manuscript_table.write_text(
        """# Metric-pathology summary

## E2

| Condition | n | Reference accuracy | Candidate corrective recall@5 | Context corrective recall | Context support rate | Visible target-evidence agreement | Reference-sufficient context |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(
            f"| {condition} | {e2_summary[condition]['n']} "
            f"| {fmt(e2_summary[condition]['reference_accuracy'])} "
            f"| {fmt(e2_summary[condition]['candidate_corrective_recall_at_5'])} "
            f"| {fmt(e2_summary[condition]['context_corrective_recall'])} "
            f"| {fmt(e2_summary[condition]['context_support_rate'])} "
            f"| {fmt(e2_summary[condition]['visible_target_evidence_agreement'])} "
            f"| {fmt(e2_summary[condition]['reference_sufficient_context_rate'])} |"
            for condition in E2_CONDITIONS
        )
        + """

## E3

| Condition | n | Reference accuracy | Candidate corrective recall@5 | Context corrective recall | Context support rate | Visible target-evidence agreement | Reference-sufficient context |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(
            f"| {condition} | {e3_summary[condition]['n']} "
            f"| {fmt(e3_summary[condition]['reference_accuracy'])} "
            f"| {fmt(e3_summary[condition]['candidate_corrective_recall_at_5'])} "
            f"| {fmt(e3_summary[condition]['context_corrective_recall'])} "
            f"| {fmt(e3_summary[condition]['context_support_rate'])} "
            f"| {fmt(e3_summary[condition]['visible_target_evidence_agreement'])} "
            f"| {fmt(e3_summary[condition]['reference_sufficient_context_rate'])} |"
            for condition in E3_CONDITIONS
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 132)
    print(
        "E2 hidden-condition pathology present:       "
        f"{pathology_pass}"
    )
    print(
        "E3 below-k boundary replication present:     "
        f"{boundary_pathology_pass}"
    )
    print()
    print(f"Summary:          {summary_path}")
    print(f"E2 detail CSV:    {e2_detail_path}")
    print(f"E3 detail CSV:    {e3_detail_path}")
    print(f"Manuscript table: {manuscript_table}")


if __name__ == "__main__":
    main()
