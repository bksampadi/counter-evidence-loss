import json
from collections import defaultdict
from pathlib import Path

import yaml

from signalrank.components.ranking.flashrank import FlashRankReranker
from signalrank.pipelines.indexing_pipeline import IndexingPipeline
from signalrank.pipelines.retrieval_pipeline import RetrievalPipeline
from signalrank.services.ranking_service import RankingService
from signalrank.services.retrieval_service import RetrievalService

BASE_CONFIG_PATH = Path("configs/benchmark.yaml")

STIMULUS_ROOT = Path("benchmarks/Exp1/stimuli_v1")
CLAIMS_PATH = STIMULUS_ROOT / "claims.json"
PREFLIGHT_SUMMARY_PATH = Path(
    "benchmark_results/Exp1/preflight/stimulus_preflight_summary.json"
)

CORPUS_ROOT = Path("benchmarks/Exp1/corpora/formal_v1")
OUTPUT_ROOT = Path("benchmark_results/Exp1/formal")
GENERATED_CONFIG_ROOT = OUTPUT_ROOT / "generated_configs"

CORPUS_STATES = ("available", "absent")
EXPERIMENTAL_STATES = (
    "exposed",
    "available_not_exposed",
    "absent",
)

CONTEXT_K = 5
RETRIEVAL_MODE = "hybrid"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def assert_preflight_passed(stimulus_version: str) -> None:
    if not PREFLIGHT_SUMMARY_PATH.exists():
        raise RuntimeError(
            "Formal E1 is gated by the anti-leak preflight. "
            f"Missing: {PREFLIGHT_SUMMARY_PATH}"
        )

    preflight = load_json(PREFLIGHT_SUMMARY_PATH)

    if preflight.get("stimulus_version") != stimulus_version:
        raise RuntimeError(
            "Preflight stimulus version does not match formal E1 stimuli: "
            f"{preflight.get('stimulus_version')} != {stimulus_version}"
        )

    if preflight.get("all_checks_pass") is not True:
        raise RuntimeError(
            "Anti-leak preflight did not pass. Do NOT run formal E1."
        )

    if preflight.get("passed_checks") != 36:
        raise RuntimeError(
            "Formal E1 expects the frozen 36/36 anti-leak preflight."
        )


def make_corpus_config(state: str) -> Path:
    with BASE_CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["data_ingestion"]["source_path"] = str(
        CORPUS_ROOT / state
    ).replace("\\", "/")

    config["retrieval"]["top_k"] = CONTEXT_K

    config["qdrant"]["mode"] = "local"
    config["qdrant"]["collection_name"] = (
        f"signalrank_exp1_formal_v1_{state}"
    )
    config["qdrant"]["recreate_collection"] = True
    config["qdrant"]["path"] = str(
        Path("data/qdrant/experiments/exp1_formal_v1") / state
    ).replace("\\", "/")

    GENERATED_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    path = GENERATED_CONFIG_ROOT / f"{state}.yaml"

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return path


def load_manifest(state: str) -> dict[str, dict]:
    path = CORPUS_ROOT / state / "manifest.json"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run "
            "scripts/Exp1/prepare_formal_corpora.py first."
        )

    manifest = load_json(path)
    return {
        item["filename"]: item
        for item in manifest["documents"]
    }


def build_services(
    config_path: Path,
) -> tuple[RetrievalService, RankingService, int]:
    IndexingPipeline(config_filepath=config_path).run()

    bm25, dense, hybrid = RetrievalPipeline(
        config_filepath=config_path
    ).build()

    retrieval_service = RetrievalService(
        retrievers={
            "bm25": bm25,
            "dense": dense,
            "hybrid": hybrid,
        }
    )

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    ranking_config = config.get("ranking", {})

    if ranking_config.get("provider", "flashrank") != "flashrank":
        raise ValueError(
            "Formal Experiment 1 requires the frozen FlashRank reranker."
        )

    reranker = FlashRankReranker(
        model_name=ranking_config.get(
            "model_name",
            "ms-marco-MiniLM-L-12-v2",
        ),
        max_length=int(ranking_config.get("max_length", 512)),
    )

    ranking_service = RankingService(
        rerankers={"flashrank": reranker}
    )

    candidate_multiplier = int(
        ranking_config.get(
            "candidate_multiplier",
            config["retrieval"].get("candidate_multiplier", 4),
        )
    )

    if candidate_multiplier <= 0:
        raise ValueError(
            "candidate_multiplier must be greater than zero"
        )

    inspection_k = CONTEXT_K * candidate_multiplier
    return retrieval_service, ranking_service, inspection_k


def find_rank(
    ranked_results,
    *,
    expected_filename: str,
) -> int | None:
    for rank, result in enumerate(ranked_results, start=1):
        if Path(result.source_path).name == expected_filename:
            return rank
    return None


def build_context(
    ranked_results,
    *,
    condition: str,
    critical_filename: str,
):
    if condition in {"exposed", "absent"}:
        return ranked_results[:CONTEXT_K]

    if condition == "available_not_exposed":
        return [
            result
            for result in ranked_results
            if Path(result.source_path).name != critical_filename
        ][:CONTEXT_K]

    raise ValueError(f"Unknown condition: {condition}")


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def main() -> None:
    stimuli = load_json(CLAIMS_PATH)
    stimulus_version = stimuli["stimulus_version"]
    claims = stimuli["claims"]

    assert_preflight_passed(stimulus_version)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    detailed_rows: list[dict] = []
    context_rows: list[dict] = []
    state_summaries: dict[str, list[dict]] = defaultdict(list)

    print()
    print("FORMAL EXPERIMENT 1 — EVIDENCE AVAILABILITY / EXPOSURE")
    print("=" * 84)
    print(f"Stimuli:         {stimulus_version}")
    print(f"Retrieval mode:  {RETRIEVAL_MODE}")
    print(f"Context k:       {CONTEXT_K}")
    print("Reranker:        FlashRank")
    print("Preflight gate:  36/36 PASS")
    print()

    for corpus_state in CORPUS_STATES:
        config_path = make_corpus_config(corpus_state)
        manifest = load_manifest(corpus_state)

        retrieval_service, ranking_service, inspection_k = build_services(
            config_path
        )

        for claim in claims:
            claim_id = claim["claim_id"]
            truth_label = bool(claim["truth_label"])
            query = claim["query"]

            candidates = retrieval_service.retrieve(
                query=query,
                mode=RETRIEVAL_MODE,
                top_k=inspection_k,
            )

            ranked = ranking_service.rerank(
                query=query,
                candidates=candidates,
                mode="flashrank",
                top_k=inspection_k,
            )

            support_filename = f"{claim_id}__support.txt"
            critical_filename = f"{claim_id}__critical.txt"

            support_rank = find_rank(
                ranked,
                expected_filename=support_filename,
            )
            critical_rank = find_rank(
                ranked,
                expected_filename=critical_filename,
            )

            critical_present_in_corpus = critical_filename in manifest

            critical_originally_in_top_5 = (
                critical_rank is not None
                and critical_rank <= CONTEXT_K
            )

            if corpus_state == "available":
                experimental_states = (
                    "exposed",
                    "available_not_exposed",
                )
            else:
                experimental_states = ("absent",)

            for experimental_state in experimental_states:
                context = build_context(
                    ranked,
                    condition=experimental_state,
                    critical_filename=critical_filename,
                )

                context_source_files = [
                    Path(result.source_path).name
                    for result in context
                ]
                context_filenames = set(context_source_files)

                support_visible = support_filename in context_filenames
                critical_visible = critical_filename in context_filenames

                if experimental_state == "exposed":
                    manipulation_pass = (
                        support_visible
                        and critical_present_in_corpus
                        and critical_originally_in_top_5
                        and critical_visible
                    )

                elif experimental_state == "available_not_exposed":
                    manipulation_pass = (
                        support_visible
                        and critical_present_in_corpus
                        and critical_originally_in_top_5
                        and not critical_visible
                    )

                else:
                    manipulation_pass = (
                        support_visible
                        and not critical_present_in_corpus
                        and critical_rank is None
                        and not critical_visible
                    )

                summary_row = {
                    "claim_id": claim_id,
                    "truth_label": truth_label,
                    "query": query,
                    "corpus_state": corpus_state,
                    "experimental_state": experimental_state,
                    "support_rank": support_rank,
                    "critical_rank": critical_rank,
                    "critical_present_in_corpus": critical_present_in_corpus,
                    "critical_originally_in_top_5": (
                        critical_originally_in_top_5
                    ),
                    "support_visible": support_visible,
                    "critical_visible": critical_visible,
                    "critical_rr": reciprocal_rank(critical_rank),
                    "context_source_files": context_source_files,
                    "manipulation_pass": manipulation_pass,
                }

                state_summaries[experimental_state].append(summary_row)

                context_rows.append(
                    {
                        "experiment": "formal_experiment_1",
                        "stimulus_version": stimulus_version,
                        **summary_row,
                    }
                )

            for rank, result in enumerate(ranked, start=1):
                source_file = Path(result.source_path).name
                annotation = manifest.get(
                    source_file,
                    {
                        "claim_id": None,
                        "role": "unmapped",
                    },
                )

                detailed_rows.append(
                    {
                        "experiment": "formal_experiment_1",
                        "stimulus_version": stimulus_version,
                        "claim_id": claim_id,
                        "truth_label": truth_label,
                        "corpus_state": corpus_state,
                        "query": query,
                        "rank": rank,
                        "chunk_id": result.chunk_id,
                        "doc_id": result.doc_id,
                        "source_file": source_file,
                        "source_claim_id": annotation["claim_id"],
                        "evidence_role": annotation["role"],
                        "retrieval_score": result.metadata.get(
                            "retrieval_score",
                            result.score,
                        ),
                        "retrieval_rank": result.metadata.get(
                            "retrieval_rank"
                        ),
                        "reranker_score": result.metadata.get(
                            "reranker_score",
                            result.score,
                        ),
                        "in_top_5": rank <= CONTEXT_K,
                        "is_target_support": (
                            source_file == support_filename
                        ),
                        "is_target_critical": (
                            source_file == critical_filename
                        ),
                    }
                )

    detail_path = OUTPUT_ROOT / "experiment_1_retrieval.jsonl"
    with detail_path.open("w", encoding="utf-8") as file:
        for row in detailed_rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    contexts_path = OUTPUT_ROOT / "experiment_1_contexts.jsonl"
    with contexts_path.open("w", encoding="utf-8") as file:
        for row in context_rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    summary = {
        "experiment": "formal_experiment_1",
        "formal_experiment": True,
        "stimulus_version": stimulus_version,
        "retrieval_mode": RETRIEVAL_MODE,
        "context_k": CONTEXT_K,
        "corpus_states": list(CORPUS_STATES),
        "experimental_states": list(EXPERIMENTAL_STATES),
        "preflight_gate": {
            "path": str(PREFLIGHT_SUMMARY_PATH),
            "passed_checks": 36,
            "all_checks_pass": True,
        },
        "states": {},
    }

    for state in EXPERIMENTAL_STATES:
        rows = state_summaries[state]

        critical_visible_rate = (
            sum(row["critical_visible"] for row in rows) / len(rows)
        )
        support_visible_rate = (
            sum(row["support_visible"] for row in rows) / len(rows)
        )
        critical_original_top5_rate = (
            sum(
                row["critical_originally_in_top_5"]
                for row in rows
            )
            / len(rows)
        )
        mean_critical_rr = (
            sum(row["critical_rr"] for row in rows) / len(rows)
        )
        pass_rate = (
            sum(row["manipulation_pass"] for row in rows) / len(rows)
        )

        summary["states"][state] = {
            "critical_visible_rate_at_5": critical_visible_rate,
            "support_visible_rate_at_5": support_visible_rate,
            "critical_original_top5_rate": critical_original_top5_rate,
            "mean_critical_reciprocal_rank": mean_critical_rr,
            "manipulation_pass_rate": pass_rate,
            "claims": rows,
        }

        print(state.upper())
        print("-" * 84)
        print(
            f"{'Claim':<12}"
            f"{'Truth':<8}"
            f"{'Support r':>10}"
            f"{'Critical r':>12}"
            f"{'Orig@5':>9}"
            f"{'Visible':>9}"
            f"{'Pass':>8}"
        )

        for row in rows:
            print(
                f"{row['claim_id']:<12}"
                f"{str(row['truth_label']):<8}"
                f"{str(row['support_rank']):>10}"
                f"{str(row['critical_rank']):>12}"
                f"{str(row['critical_originally_in_top_5']):>9}"
                f"{str(row['critical_visible']):>9}"
                f"{str(row['manipulation_pass']):>8}"
            )

        print(
            f"\nOriginal critical @5: "
            f"{critical_original_top5_rate:.3f}"
            f" | Critical visible: {critical_visible_rate:.3f}"
            f" | Support visible: {support_visible_rate:.3f}"
            f" | Manipulation pass: {pass_rate:.3f}\n"
        )

    all_manipulations_pass = all(
        row["manipulation_pass"]
        for state in EXPERIMENTAL_STATES
        for row in state_summaries[state]
    )

    all_support_visible = all(
        row["support_visible"]
        for state in EXPERIMENTAL_STATES
        for row in state_summaries[state]
    )

    summary["all_manipulation_checks_pass"] = (
        all_manipulations_pass
    )
    summary["all_support_visible"] = all_support_visible
    summary["formal_acceptance_pass"] = (
        all_manipulations_pass and all_support_visible
    )

    summary_path = OUTPUT_ROOT / "experiment_1_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 84)
    print(
        f"All manipulation checks pass: "
        f"{all_manipulations_pass}"
    )
    print(
        f"Support remains visible:       "
        f"{all_support_visible}"
    )
    print(
        f"FORMAL E1 ACCEPTANCE:           "
        f"{summary['formal_acceptance_pass']}"
    )
    print()
    print(f"Detailed ranking: {detail_path}")
    print(f"Frozen contexts: {contexts_path}")
    print(f"Summary:         {summary_path}")

    if not summary["formal_acceptance_pass"]:
        raise SystemExit(
            "Formal E1 did not satisfy the frozen acceptance criteria. "
            "Do not proceed to E2."
        )


if __name__ == "__main__":
    main()
