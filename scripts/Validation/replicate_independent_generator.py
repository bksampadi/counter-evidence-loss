import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

EXP1_SUMMARY_PATH = Path(
    "benchmark_results/Exp1/formal/experiment_1_summary.json"
)
EXP2_RESULTS_PATH = Path(
    "benchmark_results/Exp2/formal/experiment_2_generation.jsonl"
)
EXP3_RESULTS_PATH = Path(
    "benchmark_results/Exp3/formal/experiment_3_generation.jsonl"
)

AVAILABLE_CORPUS = Path("benchmarks/Exp1/corpora/formal_v1/available")
ABSENT_CORPUS = Path("benchmarks/Exp1/corpora/formal_v1/absent")

OUTPUT_ROOT = Path(
    "benchmark_results/independent_generator_replication"
)

# IMPORTANT:
# These stimuli were NOT selected, rewritten, or preflighted on this model.
MODEL_NAME = "qwen/qwen3.6-27b"
MAX_RETRIES = 0
MAX_OUTPUT_TOKENS = 64
INTER_CALL_SECONDS = 3

SYSTEM_PROMPT = """You are a claim-verification system.

Judge the claim using ONLY the supplied evidence.
Do not use outside knowledge.

Return exactly ONE word and nothing else:
TRUE
or
FALSE
"""

Verdict = Literal["TRUE", "FALSE"]
VERDICT_PATTERN = re.compile(
    r"^\s*(TRUE|FALSE)[.!]?\s*$",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def parse_verdict(text: str) -> Verdict:
    match = VERDICT_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(
            f"Independent generator returned non-canonical verdict: {text!r}"
        )
    return match.group(1).upper()  # type: ignore[return-value]


def truth_verdict(truth_label: bool) -> Verdict:
    return "TRUE" if truth_label else "FALSE"


def context_root(source_experiment: str, condition: str, corpus_state: str | None) -> Path:
    if source_experiment == "E2":
        if corpus_state == "available":
            return AVAILABLE_CORPUS
        if corpus_state == "absent":
            return ABSENT_CORPUS
        raise ValueError(f"Unknown E2 corpus_state: {corpus_state}")

    if condition == "corrective_absent_k1":
        return ABSENT_CORPUS

    return AVAILABLE_CORPUS


def load_context_texts(
    *,
    source_experiment: str,
    condition: str,
    corpus_state: str | None,
    source_files: list[str],
) -> list[str]:
    root = context_root(source_experiment, condition, corpus_state)
    texts: list[str] = []

    for filename in source_files:
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Frozen context file missing: {path}")
        texts.append(path.read_text(encoding="utf-8").strip())

    return texts


def run_case(
    *,
    llm: ChatGroq,
    source_experiment: str,
    row: dict,
) -> dict:
    query = row["query"]
    truth_label = bool(row["truth_label"])
    condition = row["condition"]

    source_files = list(row["context_source_files"])
    corpus_state = row.get("corpus_state")

    context_texts = load_context_texts(
        source_experiment=source_experiment,
        condition=condition,
        corpus_state=corpus_state,
        source_files=source_files,
    )

    blocks = [
        f"[Evidence {rank}]\n{text}"
        for rank, text in enumerate(context_texts, start=1)
    ]
    context = "\n\n".join(blocks)

    prompt = (
        f"Claim:\n{query}\n\n"
        f"Evidence:\n{context}\n\n"
        "Determine whether the claim is TRUE or FALSE "
        "based only on the supplied evidence."
    )

    message = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    raw_output = ai_message_text(message)
    verdict = parse_verdict(raw_output)
    correct = verdict == truth_verdict(truth_label)

    return {
        "analysis": "independent_generator_replication",
        "source_experiment": source_experiment,
        "generator_model": MODEL_NAME,
        "generator_family": "qwen",
        "stimuli_selected_with_this_model": False,
        "stimuli_preflighted_with_this_model": False,
        "temperature": 0,
"reasoning_effort": "none",
        "claim_id": row["claim_id"],
        "truth_label": truth_label,
        "query": query,
        "condition": condition,
        "context_source_files": source_files,
        "verdict": verdict,
        "correct": correct,
        "raw_model_output": raw_output,
    }


def summarize(rows: list[dict], condition_order: tuple[str, ...]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    summary = {}

    for condition in condition_order:
        items = grouped[condition]
        correct = sum(item["correct"] for item in items)
        summary[condition] = {
            "n": len(items),
            "correct": correct,
            "accuracy": correct / len(items),
            "true_claim_accuracy": (
                sum(
                    item["correct"]
                    for item in items
                    if item["truth_label"]
                )
                / sum(item["truth_label"] for item in items)
            ),
            "false_claim_accuracy": (
                sum(
                    item["correct"]
                    for item in items
                    if not item["truth_label"]
                )
                / sum(not item["truth_label"] for item in items)
            ),
        }

    return summary


def paired_contrast(
    rows: list[dict],
    left: str,
    right: str,
) -> dict:
    index = {
        (row["claim_id"], row["condition"]): row
        for row in rows
    }
    claim_ids = sorted({row["claim_id"] for row in rows})

    switches = 0
    left_correct_right_wrong = 0
    left_wrong_right_correct = 0

    pairs = []

    for claim_id in claim_ids:
        a = index[(claim_id, left)]
        b = index[(claim_id, right)]

        switched = a["verdict"] != b["verdict"]
        deterioration = a["correct"] and not b["correct"]
        improvement = (not a["correct"]) and b["correct"]

        switches += switched
        left_correct_right_wrong += deterioration
        left_wrong_right_correct += improvement

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

    return {
        "n_claims": len(claim_ids),
        "verdict_switch_count": switches,
        "verdict_switch_rate": switches / len(claim_ids),
        "left_correct_right_wrong_count": left_correct_right_wrong,
        "left_wrong_right_correct_count": left_wrong_right_correct,
        "pairs": pairs,
    }


def main() -> None:
    exp1 = load_json(EXP1_SUMMARY_PATH)
    exp2 = load_jsonl(EXP2_RESULTS_PATH)
    exp3 = load_jsonl(EXP3_RESULTS_PATH)

    if exp1.get("formal_acceptance_pass") is not True:
        raise RuntimeError("Formal E1 acceptance is not True.")

    if len(exp2) != 18:
        raise RuntimeError(f"Expected 18 frozen E2 rows, got {len(exp2)}.")

    if len(exp3) != 24:
        raise RuntimeError(f"Expected 24 frozen E3 rows, got {len(exp3)}.")

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        max_retries=MAX_RETRIES,
        max_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort="none",
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print()
    print("INDEPENDENT GENERATOR REPLICATION")
    print("=" * 100)
    print(f"Generator:        {MODEL_NAME}")
    print("Model family:     Qwen")
    print("Temperature:      0")
    print("Reasoning effort: none (Qwen non-thinking mode)")
    print("Fallbacks:        none")
    print("Stimulus preflight on this model: NONE")
    print("Stimulus selection on this model: NONE")
    print("Frozen contexts:  Formal E2 + E3")
    print()

    all_rows: list[dict] = []

    for source_name, source_rows in (("E2", exp2), ("E3", exp3)):
        print(source_name)
        print("-" * 100)

        for row in source_rows:
            result = run_case(
                llm=llm,
                source_experiment=source_name,
                row=row,
            )
            all_rows.append(result)

            print(
                f"{source_name:<4}"
                f"{result['condition']:<34}"
                f"{result['claim_id']:<10}"
                f"truth={str(result['truth_label']):<6}"
                f"verdict={result['verdict']:<6}"
                f"correct={result['correct']}"
            )

            time.sleep(INTER_CALL_SECONDS)

        print()

    e2_rows = [r for r in all_rows if r["source_experiment"] == "E2"]
    e3_rows = [r for r in all_rows if r["source_experiment"] == "E3"]

    e2_order = (
        "exposed",
        "available_not_exposed",
        "absent",
    )
    e3_order = (
        "corrective_exposed_k2",
        "corrective_below_k1",
        "corrective_absent_k1",
        "wrong_entity_control_k2",
    )

    e2_summary = summarize(e2_rows, e2_order)
    e3_summary = summarize(e3_rows, e3_order)

    e2_contrasts = {
        "exposed_vs_available_not_exposed": paired_contrast(
            e2_rows,
            "exposed",
            "available_not_exposed",
        ),
        "exposed_vs_absent": paired_contrast(
            e2_rows,
            "exposed",
            "absent",
        ),
    }

    e3_contrasts = {
        "k2_exposed_vs_k1_below": paired_contrast(
            e3_rows,
            "corrective_exposed_k2",
            "corrective_below_k1",
        ),
        "k1_below_vs_absent": paired_contrast(
            e3_rows,
            "corrective_below_k1",
            "corrective_absent_k1",
        ),
    }

    detail_path = OUTPUT_ROOT / "qwen_replication_results.jsonl"
    with detail_path.open("w", encoding="utf-8") as file:
        for row in all_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "analysis": "independent_generator_replication",
        "formal_experiment": False,
        "analysis_of_frozen_formal_contexts": True,
        "stimulus_version": exp1["stimulus_version"],
        "generator_model": MODEL_NAME,
        "generator_family": "qwen",
        "stimuli_selected_with_this_model": False,
        "stimuli_preflighted_with_this_model": False,
        "temperature": 0,
"reasoning_effort": "none",
        "e2": {
            "conditions": e2_summary,
            "contrasts": e2_contrasts,
        },
        "e3": {
            "conditions": e3_summary,
            "contrasts": e3_contrasts,
        },
    }

    summary_path = OUTPUT_ROOT / "qwen_replication_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("E2 SUMMARY")
    print("-" * 100)
    for condition in e2_order:
        item = e2_summary[condition]
        print(
            f"{condition:<30}"
            f"{item['correct']}/{item['n']} correct "
            f"({item['accuracy']:.3f})"
        )

    print()
    print("E3 SUMMARY")
    print("-" * 100)
    for condition in e3_order:
        item = e3_summary[condition]
        print(
            f"{condition:<30}"
            f"{item['correct']}/{item['n']} correct "
            f"({item['accuracy']:.3f})"
        )

    print()
    print(f"Detailed results: {detail_path}")
    print(f"Summary:          {summary_path}")


if __name__ == "__main__":
    main()
