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
    "benchmark_results/Exp4/preflight_v5/"
    "candidate_gate_preflight_v5_summary.json"
)

AVAILABLE_CORPUS = Path("benchmarks/Exp1/corpora/formal_v1/available")
ABSENT_CORPUS = Path("benchmarks/Exp1/corpora/formal_v1/absent")
OUTPUT_ROOT = Path("benchmark_results/Exp4/formal")

MODEL_NAME = "openai/gpt-oss-20b"
MAX_RETRIES = 0
MAX_OUTPUT_TOKENS = 1024
INTER_CALL_SECONDS = 10
INSPECTION_K = 5

CONDITIONS = (
    "available_opposition_gate",
    "absent_opposition_gate",
    "wrong_entity_opposition_gate",
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

EVIDENCE_VERDICT_PATTERN = re.compile(
    r"^\s*(TRUE|FALSE|INSUFFICIENT)[.!]?\s*$",
    re.IGNORECASE,
)


class ClaimAssessment(BaseModel):
    verdict: Verdict


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
    match = EVIDENCE_VERDICT_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(
            "Candidate gate returned a non-canonical verdict: "
            f"{text!r}"
        )
    return match.group(1).upper()  # type: ignore[return-value]


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
        raise RuntimeError("E4 candidate-gate preflight did not pass.")

    if preflight.get("passed_checks") != 30:
        raise RuntimeError("Formal E4 requires 30/30 candidate-gate checks.")

    if preflight.get("model_visible_file_labels") is not False:
        raise RuntimeError("Candidate-gate preflight leaked file labels.")

    stimulus_version = exp1["stimulus_version"]
    if exp3_summary.get("stimulus_version") != stimulus_version:
        raise RuntimeError("E1/E3 stimulus versions do not match.")
    if preflight.get("stimulus_version") != stimulus_version:
        raise RuntimeError("E1/E4-preflight stimulus versions do not match.")

    ranking_rows = load_jsonl(EXP1_RANKING_PATH)
    return exp1, ranking_rows, exp3_rows, preflight


def ranking_by_claim_state(
    rows: list[dict],
) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for row in rows:
        grouped[(row["claim_id"], row["corpus_state"])].append(row)

    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda row: row["rank"])

    return grouped


def e3_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (row["claim_id"], row["condition"]): row
        for row in rows
    }


def evaluate_candidate(
    *,
    llm: LLMService,
    query: str,
    candidate_text: str,
) -> EvidenceDecision:
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


def generate_final(
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


def main() -> None:
    exp1, ranking_rows, exp3_rows, preflight = assert_gates()

    ranked = ranking_by_claim_state(ranking_rows)
    baseline = e3_index(exp3_rows)
    wrong_map = preflight["wrong_entity_map"]

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
    rows: list[dict] = []

    print()
    print("FORMAL EXPERIMENT 4 — OPPOSITION-AWARE EVIDENCE ADMISSION / ABSTENTION")
    print("=" * 104)
    print(f"Stimuli:       {exp1['stimulus_version']}")
    print(f"Model:         {MODEL_NAME}")
    print("Temperature:   0")
    print("Fallbacks:     none")
    print(f"Inspection k:  {INSPECTION_K}")
    print("Model-visible filenames/roles: NO")
    print("Gate: candidate-alone TRUE/FALSE/INSUFFICIENT (plain one-word output);")
    print("      admit only if candidate verdict opposes provisional verdict.")
    print()

    for condition in CONDITIONS:
        print(condition.upper())
        print("-" * 104)
        print(
            f"{'Claim':<12}"
            f"{'Truth':<9}"
            f"{'Baseline':<12}"
            f"{'Action':<10}"
            f"{'Final':<12}"
            f"{'Safe':<8}"
        )

        for claim in claims:
            claim_id = claim["claim_id"]
            truth_label = bool(claim["truth_label"])
            query = claim["query"]

            if condition == "available_opposition_gate":
                baseline_row = baseline[
                    (claim_id, "corrective_below_k1")
                ]
                corpus_state = "available"
                ranked_rows = ranked[(claim_id, "available")][:INSPECTION_K]

                if ranked_rows[0]["source_file"] != f"{claim_id}__support.txt":
                    raise RuntimeError(
                        f"Available rank 1 is not target support for {claim_id}."
                    )

                initial_path = AVAILABLE_CORPUS / ranked_rows[0]["source_file"]
                initial_text = read_text(initial_path)
                candidate_rows = ranked_rows[1:]

            elif condition == "absent_opposition_gate":
                baseline_row = baseline[
                    (claim_id, "corrective_absent_k1")
                ]
                corpus_state = "absent"
                ranked_rows = ranked[(claim_id, "absent")][:INSPECTION_K]

                if ranked_rows[0]["source_file"] != f"{claim_id}__support.txt":
                    raise RuntimeError(
                        f"Absent rank 1 is not target support for {claim_id}."
                    )

                initial_path = ABSENT_CORPUS / ranked_rows[0]["source_file"]
                initial_text = read_text(initial_path)
                candidate_rows = ranked_rows[1:]

            else:
                baseline_row = baseline[
                    (claim_id, "wrong_entity_control_k2")
                ]
                corpus_state = "wrong_entity_control"
                initial_path = AVAILABLE_CORPUS / f"{claim_id}__support.txt"
                initial_text = read_text(initial_path)

                wrong_id = wrong_map[claim_id]
                wrong_path = AVAILABLE_CORPUS / f"{wrong_id}__critical.txt"

                candidate_rows = [
                    {
                        "rank": 2,
                        "source_file": wrong_path.name,
                        "_explicit_path": str(wrong_path),
                    }
                ]

            provisional = baseline_row["verdict"]

            if provisional == truth_to_verdict(truth_label):
                raise RuntimeError(
                    f"E4 expects an incorrect baseline for {claim_id} / {condition}."
                )

            admitted_file = None
            admitted_text = None
            verifier_trace: list[dict] = []

            for candidate in candidate_rows:
                if condition == "wrong_entity_opposition_gate":
                    candidate_path = Path(candidate["_explicit_path"])
                else:
                    corpus_root = (
                        AVAILABLE_CORPUS
                        if corpus_state == "available"
                        else ABSENT_CORPUS
                    )
                    candidate_path = corpus_root / candidate["source_file"]

                candidate_text = read_text(candidate_path)

                evidence_verdict, raw_candidate_output = evaluate_candidate(
                    llm=llm,
                    query=query,
                    candidate_text=candidate_text,
                )

                admit = (
                    evidence_verdict in {"TRUE", "FALSE"}
                    and evidence_verdict != provisional
                )

                verifier_trace.append(
                    {
                        "candidate_rank": candidate["rank"],
                        "candidate_file": candidate_path.name,
                        "candidate_evidence_verdict": evidence_verdict,
                        "raw_candidate_output": raw_candidate_output,
                        "candidate_output_mode": "plain_one_word",
                        "admitted": admit,
                    }
                )

                time.sleep(INTER_CALL_SECONDS)

                if admit:
                    admitted_file = candidate_path.name
                    admitted_text = candidate_text
                    break

            if admitted_file is not None and admitted_text is not None:
                final_verdict = generate_final(
                    llm=llm,
                    query=query,
                    evidence_texts=[initial_text, admitted_text],
                )
                action = "ANSWER"
                correct = final_verdict == truth_to_verdict(truth_label)
                safe = correct
                time.sleep(INTER_CALL_SECONDS)
            else:
                final_verdict = None
                action = "ABSTAIN"
                correct = None
                safe = True

            expected_action = (
                "ANSWER"
                if condition == "available_opposition_gate"
                else "ABSTAIN"
            )

            gate_correct = action == expected_action
            false_answer = action == "ANSWER" and correct is False
            false_abstention = (
                condition == "available_opposition_gate"
                and action == "ABSTAIN"
            )

            rows.append(
                {
                    "experiment": "formal_experiment_4",
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
                    "baseline_verdict": provisional,
                    "baseline_correct": False,
                    "inspection_k": INSPECTION_K,
                    "verifier_trace": verifier_trace,
                    "admitted_counterevidence_file": admitted_file,
                    "action": action,
                    "expected_action": expected_action,
                    "gate_correct": gate_correct,
                    "final_verdict": final_verdict,
                    "final_correct": correct,
                    "false_answer": false_answer,
                    "false_abstention": false_abstention,
                    "safe_outcome": safe,
                }
            )

            print(
                f"{claim_id:<12}"
                f"{str(truth_label):<9}"
                f"{provisional:<12}"
                f"{action:<10}"
                f"{str(final_verdict):<12}"
                f"{str(safe):<8}"
            )

        print()

    detail_path = OUTPUT_ROOT / "experiment_4_mitigation.jsonl"
    with detail_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    condition_summary = {}

    for condition in CONDITIONS:
        condition_rows = [r for r in rows if r["condition"] == condition]
        n = len(condition_rows)
        answered = sum(r["action"] == "ANSWER" for r in condition_rows)
        abstained = sum(r["action"] == "ABSTAIN" for r in condition_rows)
        false_answers = sum(r["false_answer"] for r in condition_rows)
        false_abstentions = sum(
            r["false_abstention"] for r in condition_rows
        )
        gate_correct = sum(r["gate_correct"] for r in condition_rows)
        safe = sum(r["safe_outcome"] for r in condition_rows)

        answered_rows = [
            r for r in condition_rows if r["action"] == "ANSWER"
        ]
        correct_answered = sum(
            r["final_correct"] is True for r in answered_rows
        )

        selective_accuracy = (
            correct_answered / answered if answered else None
        )
        answered_risk = (
            false_answers / answered if answered else None
        )

        condition_summary[condition] = {
            "n": n,
            "answered": answered,
            "abstained": abstained,
            "coverage": answered / n,
            "abstention_rate": abstained / n,
            "correct_answered": correct_answered,
            "selective_accuracy": selective_accuracy,
            "false_answers": false_answers,
            "false_answer_rate_all_cases": false_answers / n,
            "answered_risk": answered_risk,
            "false_abstentions": false_abstentions,
            "gate_correct": gate_correct,
            "gate_accuracy": gate_correct / n,
            "safe_outcomes": safe,
            "safe_outcome_rate": safe / n,
        }

    all_n = len(rows)
    all_answered = sum(r["action"] == "ANSWER" for r in rows)
    all_false_answers = sum(r["false_answer"] for r in rows)
    all_safe = sum(r["safe_outcome"] for r in rows)

    overall = {
        "n": all_n,
        "answered": all_answered,
        "coverage": all_answered / all_n,
        "false_answers": all_false_answers,
        "false_answer_rate_all_cases": all_false_answers / all_n,
        "safe_outcomes": all_safe,
        "safe_outcome_rate": all_safe / all_n,
    }

    summary = {
        "experiment": "formal_experiment_4",
        "formal_experiment": True,
        "stimulus_version": exp1["stimulus_version"],
        "model_provider": "groq",
        "model_name": MODEL_NAME,
        "temperature": 0,
        "inspection_k": INSPECTION_K,
        "gate": (
            "candidate-alone evidential verdict via plain one-word output; "
            "admit only if TRUE/FALSE verdict opposes provisional verdict"
        ),
        "policy": (
            "admit target-specific opposing evidence when found; "
            "otherwise abstain"
        ),
        "n_claims": 6,
        "n_cases": all_n,
        "conditions": condition_summary,
        "overall": overall,
    }

    summary_path = OUTPUT_ROOT / "experiment_4_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 104)
    print("CONDITION SUMMARY")
    print("-" * 104)

    for condition in CONDITIONS:
        item = condition_summary[condition]
        selective = (
            "N/A"
            if item["selective_accuracy"] is None
            else f"{item['selective_accuracy']:.3f}"
        )
        risk = (
            "N/A"
            if item["answered_risk"] is None
            else f"{item['answered_risk']:.3f}"
        )

        print(
            f"{condition:<32}"
            f"coverage={item['coverage']:.3f}"
            f" | selective_acc={selective}"
            f" | answered_risk={risk}"
            f" | false_answers={item['false_answers']}/{item['n']}"
            f" | false_abstentions={item['false_abstentions']}/{item['n']}"
            f" | gate_acc={item['gate_accuracy']:.3f}"
            f" | safe={item['safe_outcome_rate']:.3f}"
        )

    print()
    print("OVERALL")
    print("-" * 104)
    print(
        f"coverage={overall['coverage']:.3f}"
        f" | false_answer_rate={overall['false_answer_rate_all_cases']:.3f}"
        f" | safe_outcome_rate={overall['safe_outcome_rate']:.3f}"
    )
    print()
    print(f"Detailed results: {detail_path}")
    print(f"Summary:          {summary_path}")


if __name__ == "__main__":
    main()
