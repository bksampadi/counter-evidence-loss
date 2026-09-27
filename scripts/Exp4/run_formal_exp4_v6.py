import json
import re
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
EXP1_RANKING_PATH = Path(
    "benchmark_results/Exp1/formal/experiment_1_retrieval.jsonl"
)
EXP3_RESULTS_PATH = Path(
    "benchmark_results/Exp3/formal/experiment_3_generation.jsonl"
)
EXP3_SUMMARY_PATH = Path(
    "benchmark_results/Exp3/formal/experiment_3_summary.json"
)
EXP4_PREFLIGHT_PATH = Path(
    "benchmark_results/Exp4/preflight_v6/"
    "candidate_gate_preflight_v6_summary.json"
)

STIMULUS_ROOT = Path("benchmarks/Exp1/stimuli_v1")
AVAILABLE_CORPUS = Path("benchmarks/Exp1/corpora/formal_v1/available")
ABSENT_CORPUS = Path("benchmarks/Exp1/corpora/formal_v1/absent")
OUTPUT_ROOT = Path("benchmark_results/Exp4/formal_v6")

MODEL_NAME = "openai/gpt-oss-20b"
MAX_RETRIES = 0
MAX_OUTPUT_TOKENS = 1024
INTER_CALL_SECONDS = 10
INSPECTION_K = 5

SCENARIOS = (
    "wrong_baseline_valid_correction",
    "wrong_baseline_no_correction",
    "correct_baseline_no_opposition",
    "correct_baseline_stale_opposition",
)

POLICIES = (
    "conservative_abstain",
    "retain_if_no_opposition",
)

CANDIDATE_SYSTEM_PROMPT = """You are evaluating whether ONE candidate evidence passage establishes a factual claim.

Use ONLY the candidate evidence. Do not use outside knowledge.

Return exactly ONE of these three words and nothing else:
TRUE
FALSE
INSUFFICIENT

Use TRUE if the candidate establishes that the claim is true.
Use FALSE if the candidate establishes that the claim is false.
Use INSUFFICIENT if the candidate does not establish either verdict.

Important:
- A passage about a different entity, test, condition, endpoint, or sample is INSUFFICIENT.
- A passage that only describes handling, setup, storage, documentation, equipment, or administration is INSUFFICIENT.
- Missing outcome information is INSUFFICIENT, not FALSE.
"""

GENERATOR_SYSTEM_PROMPT = """You are a claim-verification system.

Judge the claim using ONLY the supplied evidence.
Do not use outside knowledge.

Return TRUE if the supplied evidence supports the claim.
Return FALSE if the supplied evidence contradicts the claim.

You must return one binary verdict.
"""

EvidenceDecision = Literal["TRUE", "FALSE", "INSUFFICIENT"]
Verdict = Literal["TRUE", "FALSE"]

EVIDENCE_PATTERN = re.compile(
    r"^\s*(TRUE|FALSE|INSUFFICIENT)[.!]?\s*$",
    re.IGNORECASE,
)


class ClaimAssessment(BaseModel):
    verdict: Verdict


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def truth_to_verdict(truth_label: bool) -> Verdict:
    return "TRUE" if truth_label else "FALSE"


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


def parse_evidence_verdict(text: str) -> EvidenceDecision:
    match = EVIDENCE_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(
            "Candidate gate returned a non-canonical verdict: "
            f"{text!r}"
        )

    return match.group(1).upper()  # type: ignore[return-value]


def generate_binary(
    *,
    llm: LLMService,
    query: str,
    evidence_texts: list[str],
) -> Verdict:
    blocks = [
        f"[Evidence {rank}]\n{text}"
        for rank, text in enumerate(evidence_texts, start=1)
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
            SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ],
        ClaimAssessment,
    )
    return assessment.verdict


def evaluate_candidate(
    *,
    llm: LLMService,
    query: str,
    candidate_text: str,
) -> tuple[EvidenceDecision, str]:
    prompt = (
        f"Claim:\n{query}\n\n"
        f"Candidate evidence:\n{candidate_text}\n\n"
        "Return the evidential verdict."
    )

    message = llm.invoke(
        [
            SystemMessage(content=CANDIDATE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    raw_output = ai_message_text(message)
    return parse_evidence_verdict(raw_output), raw_output


def assert_gates() -> tuple[dict, list[dict], list[dict], dict]:
    exp1 = load_json(EXP1_SUMMARY_PATH)
    exp3_summary = load_json(EXP3_SUMMARY_PATH)
    exp3_rows = load_jsonl(EXP3_RESULTS_PATH)
    preflight = load_json(EXP4_PREFLIGHT_PATH)

    if exp1.get("formal_acceptance_pass") is not True:
        raise RuntimeError("Formal E1 acceptance is not True.")

    if exp3_summary.get("formal_experiment") is not True:
        raise RuntimeError("Formal E3 is not marked formal.")

    if preflight.get("all_checks_pass") is not True:
        raise RuntimeError("E4 v6 candidate-gate preflight did not pass.")

    if preflight.get("passed_checks") != 36:
        raise RuntimeError(
            "Formal E4 v6 requires 36/36 candidate-gate checks."
        )

    stimulus_version = exp1["stimulus_version"]

    if exp3_summary.get("stimulus_version") != stimulus_version:
        raise RuntimeError("E1/E3 stimulus versions do not match.")

    if preflight.get("stimulus_version") != stimulus_version:
        raise RuntimeError("E1/E4-v6 stimulus versions do not match.")

    ranking_rows = load_jsonl(EXP1_RANKING_PATH)

    return exp1, ranking_rows, exp3_rows, preflight


def ranking_by_claim_state(
    rows: list[dict],
) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for row in rows:
        grouped[(row["claim_id"], row["corpus_state"])].append(row)

    for key in grouped:
        grouped[key] = sorted(
            grouped[key],
            key=lambda row: row["rank"],
        )

    return grouped


def e3_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (row["claim_id"], row["condition"]): row
        for row in rows
    }


def derive_policy_outcome(
    *,
    policy: str,
    baseline_verdict: Verdict,
    revised_verdict: Verdict | None,
    opposition_found: bool,
    truth_label: bool,
    desired_action: str,
) -> dict:
    if opposition_found:
        action = "ANSWER"
        final_verdict = revised_verdict
    elif policy == "conservative_abstain":
        action = "ABSTAIN"
        final_verdict = None
    elif policy == "retain_if_no_opposition":
        action = "ANSWER"
        final_verdict = baseline_verdict
    else:
        raise ValueError(f"Unknown policy: {policy}")

    final_correct = (
        final_verdict == truth_to_verdict(truth_label)
        if final_verdict is not None
        else None
    )

    false_answer = (
        action == "ANSWER"
        and final_correct is False
    )

    false_abstention = (
        action == "ABSTAIN"
        and desired_action == "ANSWER"
    )

    correct_abstention = (
        action == "ABSTAIN"
        and desired_action == "ABSTAIN"
    )

    desired_outcome = (
        (desired_action == "ABSTAIN" and correct_abstention)
        or (
            desired_action == "ANSWER"
            and action == "ANSWER"
            and final_correct is True
        )
    )

    return {
        "policy": policy,
        "action": action,
        "final_verdict": final_verdict,
        "final_correct": final_correct,
        "false_answer": false_answer,
        "false_abstention": false_abstention,
        "correct_abstention": correct_abstention,
        "desired_outcome": desired_outcome,
    }


def main() -> None:
    exp1, ranking_rows, exp3_rows, _ = assert_gates()

    ranked = ranking_by_claim_state(ranking_rows)
    e3 = e3_index(exp3_rows)

    claims = sorted(
        exp1["states"]["exposed"]["claims"],
        key=lambda row: row["claim_id"],
    )

    provider = build_groq_provider(
        model_name=MODEL_NAME,
        max_retries=MAX_RETRIES,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    llm = LLMService([provider])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Generate the six intentionally correct baselines once.
    correct_baselines: dict[str, Verdict] = {}

    print()
    print("FORMAL EXPERIMENT 4 V6 — MITIGATION RISK / COVERAGE TRADE-OFF")
    print("=" * 112)
    print(f"Stimuli:       {exp1['stimulus_version']}")
    print(f"Model:         {MODEL_NAME}")
    print("Temperature:   0")
    print("Fallbacks:     none")
    print("Model-visible filenames/roles: NO")
    print("Candidate gate: plain TRUE/FALSE/INSUFFICIENT")
    print("Policies:")
    print("  conservative_abstain      -> abstain if no opposition is found")
    print("  retain_if_no_opposition   -> keep baseline if no opposition is found")
    print()

    print("CORRECT-BASELINE CONSTRUCTION CHECK")
    print("-" * 112)

    for claim in claims:
        claim_id = claim["claim_id"]
        truth_label = bool(claim["truth_label"])
        query = claim["query"]

        critical_path = STIMULUS_ROOT / f"{claim_id}__critical.txt"
        verdict = generate_binary(
            llm=llm,
            query=query,
            evidence_texts=[read_text(critical_path)],
        )

        if verdict != truth_to_verdict(truth_label):
            raise RuntimeError(
                f"Critical-only baseline is not correct for {claim_id}: "
                f"{verdict}"
            )

        correct_baselines[claim_id] = verdict
        print(
            f"{claim_id:<12}"
            f"truth={str(truth_label):<6}"
            f" baseline={verdict:<6}"
            f" correct=True"
        )
        time.sleep(INTER_CALL_SECONDS)

    scenario_rows: list[dict] = []
    policy_rows: list[dict] = []

    for scenario in SCENARIOS:
        print()
        print(scenario.upper())
        print("-" * 112)
        print(
            f"{'Claim':<12}"
            f"{'Truth':<9}"
            f"{'Baseline':<12}"
            f"{'OppFound':<11}"
            f"{'Revised':<12}"
            f"{'Conservative':<16}"
            f"{'Retain':<16}"
        )

        for claim in claims:
            claim_id = claim["claim_id"]
            truth_label = bool(claim["truth_label"])
            query = claim["query"]

            if scenario == "wrong_baseline_valid_correction":
                baseline_verdict = e3[
                    (claim_id, "corrective_below_k1")
                ]["verdict"]

                if baseline_verdict == truth_to_verdict(truth_label):
                    raise RuntimeError(
                        f"Expected wrong E3 baseline for {claim_id}."
                    )

                support_path = AVAILABLE_CORPUS / f"{claim_id}__support.txt"
                baseline_text = read_text(support_path)

                available_ranked = ranked[(claim_id, "available")]

                if available_ranked[0]["source_file"] != support_path.name:
                    raise RuntimeError(
                        f"Available rank 1 mismatch for {claim_id}."
                    )

                if available_ranked[1]["source_file"] != f"{claim_id}__critical.txt":
                    raise RuntimeError(
                        f"Available rank 2 is not target critical for {claim_id}."
                    )

                candidate_paths = [
                    AVAILABLE_CORPUS / available_ranked[1]["source_file"]
                ]

                desired_action = "ANSWER"

            elif scenario == "wrong_baseline_no_correction":
                baseline_verdict = e3[
                    (claim_id, "corrective_absent_k1")
                ]["verdict"]

                if baseline_verdict == truth_to_verdict(truth_label):
                    raise RuntimeError(
                        f"Expected wrong absent baseline for {claim_id}."
                    )

                absent_ranked = ranked[(claim_id, "absent")][:INSPECTION_K]

                support_path = ABSENT_CORPUS / f"{claim_id}__support.txt"

                if absent_ranked[0]["source_file"] != support_path.name:
                    raise RuntimeError(
                        f"Absent rank 1 mismatch for {claim_id}."
                    )

                baseline_text = read_text(support_path)

                candidate_paths = [
                    ABSENT_CORPUS / row["source_file"]
                    for row in absent_ranked[1:]
                ]

                desired_action = "ABSTAIN"

            elif scenario == "correct_baseline_no_opposition":
                baseline_verdict = correct_baselines[claim_id]

                critical_path = STIMULUS_ROOT / f"{claim_id}__critical.txt"
                baseline_text = read_text(critical_path)

                candidate_paths = [
                    STIMULUS_ROOT / f"{claim_id}__neutral_{i}.txt"
                    for i in range(1, 5)
                ]

                desired_action = "ANSWER"

            else:
                baseline_verdict = correct_baselines[claim_id]

                critical_path = STIMULUS_ROOT / f"{claim_id}__critical.txt"
                support_path = STIMULUS_ROOT / f"{claim_id}__support.txt"

                baseline_text = read_text(critical_path)
                candidate_paths = [support_path]

                desired_action = "ANSWER"

            opposition_found = False
            admitted_path: Path | None = None
            admitted_text: str | None = None
            candidate_trace: list[dict] = []

            for candidate_path in candidate_paths:
                evidence_verdict, raw_output = evaluate_candidate(
                    llm=llm,
                    query=query,
                    candidate_text=read_text(candidate_path),
                )

                admitted = (
                    evidence_verdict in {"TRUE", "FALSE"}
                    and evidence_verdict != baseline_verdict
                )

                candidate_trace.append(
                    {
                        "candidate_file": candidate_path.name,
                        "evidence_verdict": evidence_verdict,
                        "raw_output": raw_output,
                        "admitted": admitted,
                    }
                )

                time.sleep(INTER_CALL_SECONDS)

                if admitted:
                    opposition_found = True
                    admitted_path = candidate_path
                    admitted_text = read_text(candidate_path)
                    break

            revised_verdict: Verdict | None = None

            if opposition_found:
                if admitted_text is None:
                    raise RuntimeError("Opposition found without admitted text.")

                revised_verdict = generate_binary(
                    llm=llm,
                    query=query,
                    evidence_texts=[
                        baseline_text,
                        admitted_text,
                    ],
                )
                time.sleep(INTER_CALL_SECONDS)

            scenario_row = {
                "experiment": "formal_experiment_4_v6",
                "formal_experiment": True,
                "stimulus_version": exp1["stimulus_version"],
                "model_provider": "groq",
                "model_name": MODEL_NAME,
                "temperature": 0,
                "claim_id": claim_id,
                "truth_label": truth_label,
                "query": query,
                "scenario": scenario,
                "baseline_verdict": baseline_verdict,
                "baseline_correct": (
                    baseline_verdict == truth_to_verdict(truth_label)
                ),
                "candidate_trace": candidate_trace,
                "opposition_found": opposition_found,
                "admitted_file": (
                    admitted_path.name
                    if admitted_path is not None
                    else None
                ),
                "revised_verdict": revised_verdict,
                "revised_correct": (
                    revised_verdict == truth_to_verdict(truth_label)
                    if revised_verdict is not None
                    else None
                ),
                "desired_action": desired_action,
                "model_visible_file_labels": False,
            }

            scenario_rows.append(scenario_row)

            policy_outcomes = {}

            for policy in POLICIES:
                outcome = derive_policy_outcome(
                    policy=policy,
                    baseline_verdict=baseline_verdict,
                    revised_verdict=revised_verdict,
                    opposition_found=opposition_found,
                    truth_label=truth_label,
                    desired_action=desired_action,
                )

                policy_outcomes[policy] = outcome

                policy_rows.append(
                    {
                        **scenario_row,
                        **outcome,
                    }
                )

            print(
                f"{claim_id:<12}"
                f"{str(truth_label):<9}"
                f"{baseline_verdict:<12}"
                f"{str(opposition_found):<11}"
                f"{str(revised_verdict):<12}"
                f"{policy_outcomes['conservative_abstain']['action']:<16}"
                f"{policy_outcomes['retain_if_no_opposition']['action']:<16}"
            )

    scenarios_path = OUTPUT_ROOT / "experiment_4_v6_scenarios.jsonl"
    with scenarios_path.open("w", encoding="utf-8") as file:
        for row in scenario_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    policy_path = OUTPUT_ROOT / "experiment_4_v6_policy_outcomes.jsonl"
    with policy_path.open("w", encoding="utf-8") as file:
        for row in policy_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summaries: dict[str, dict] = {}

    print()
    print("=" * 112)
    print("POLICY SUMMARY")
    print("-" * 112)

    for policy in POLICIES:
        rows = [row for row in policy_rows if row["policy"] == policy]
        n = len(rows)

        answered = sum(row["action"] == "ANSWER" for row in rows)
        abstained = sum(row["action"] == "ABSTAIN" for row in rows)
        false_answers = sum(row["false_answer"] for row in rows)
        false_abstentions = sum(row["false_abstention"] for row in rows)
        correct_abstentions = sum(row["correct_abstention"] for row in rows)
        desired_outcomes = sum(row["desired_outcome"] for row in rows)

        answered_rows = [
            row for row in rows if row["action"] == "ANSWER"
        ]

        correct_answers = sum(
            row["final_correct"] is True
            for row in answered_rows
        )

        coverage = answered / n
        selective_accuracy = (
            correct_answers / answered
            if answered
            else None
        )
        answered_risk = (
            false_answers / answered
            if answered
            else None
        )

        by_scenario = {}

        for scenario in SCENARIOS:
            scenario_policy_rows = [
                row for row in rows
                if row["scenario"] == scenario
            ]

            by_scenario[scenario] = {
                "n": len(scenario_policy_rows),
                "answered": sum(
                    row["action"] == "ANSWER"
                    for row in scenario_policy_rows
                ),
                "abstained": sum(
                    row["action"] == "ABSTAIN"
                    for row in scenario_policy_rows
                ),
                "false_answers": sum(
                    row["false_answer"]
                    for row in scenario_policy_rows
                ),
                "false_abstentions": sum(
                    row["false_abstention"]
                    for row in scenario_policy_rows
                ),
                "desired_outcomes": sum(
                    row["desired_outcome"]
                    for row in scenario_policy_rows
                ),
            }

        summaries[policy] = {
            "n": n,
            "answered": answered,
            "abstained": abstained,
            "coverage": coverage,
            "correct_answers": correct_answers,
            "selective_accuracy": selective_accuracy,
            "false_answers": false_answers,
            "false_answer_rate_all_cases": false_answers / n,
            "answered_risk": answered_risk,
            "false_abstentions": false_abstentions,
            "false_abstention_rate_all_cases": false_abstentions / n,
            "correct_abstentions": correct_abstentions,
            "desired_outcomes": desired_outcomes,
            "desired_outcome_rate": desired_outcomes / n,
            "by_scenario": by_scenario,
        }

        selective_str = (
            "N/A"
            if selective_accuracy is None
            else f"{selective_accuracy:.3f}"
        )
        risk_str = (
            "N/A"
            if answered_risk is None
            else f"{answered_risk:.3f}"
        )

        print(
            f"{policy:<30}"
            f"coverage={coverage:.3f}"
            f" | selective_acc={selective_str}"
            f" | answered_risk={risk_str}"
            f" | false_answers={false_answers}/{n}"
            f" | false_abstentions={false_abstentions}/{n}"
            f" | desired={desired_outcomes}/{n}"
        )

    summary = {
        "experiment": "formal_experiment_4_v6",
        "formal_experiment": True,
        "stimulus_version": exp1["stimulus_version"],
        "model_provider": "groq",
        "model_name": MODEL_NAME,
        "temperature": 0,
        "n_claims": 6,
        "n_scenarios": len(SCENARIOS),
        "n_scenario_cases": len(scenario_rows),
        "policies": list(POLICIES),
        "scenarios": list(SCENARIOS),
        "policy_summaries": summaries,
        "interpretation_note": (
            "E4 v6 is designed to expose the risk-coverage trade-off. "
            "The conservative policy can avoid false answers at the cost "
            "of false abstentions on correct baselines; the retain policy "
            "can preserve coverage at the cost of false answers when a "
            "wrong baseline has no valid opposition."
        ),
    }

    summary_path = OUTPUT_ROOT / "experiment_4_v6_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Scenario results: {scenarios_path}")
    print(f"Policy outcomes:  {policy_path}")
    print(f"Summary:          {summary_path}")


if __name__ == "__main__":
    main()
