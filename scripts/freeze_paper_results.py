import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

FREEZE_ROOT = Path("benchmark_results")
OUTPUT_PATH = Path("benchmark_results/PAPER_FREEZE_MANIFEST.json")

FROZEN_DIRS = [
    Path("benchmark_results/Exp1/formal"),
    Path("benchmark_results/Exp2/formal"),
    Path("benchmark_results/Exp3/formal"),
    Path("benchmark_results/Exp4/formal_v6"),
    Path("benchmark_results/metric_pathology"),
    Path("benchmark_results/independent_generator_replication"),
    Path("benchmark_results/external_metric_validation"),
]

# Validation/preflight summaries are included as provenance, but pilot/development
# results are deliberately excluded from the paper freeze.
PROVENANCE_FILES = [
    Path("benchmark_results/Exp1/preflight_strict/stimulus_preflight_strict_summary.json"),
    Path("benchmark_results/Exp3/preflight/wrong_entity_preflight_summary.json"),
    Path("benchmark_results/Exp4/preflight_v6/candidate_gate_preflight_v6_summary.json"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    missing = [str(p) for p in FROZEN_DIRS if not p.exists()]
    if missing:
        raise SystemExit(
            "Cannot freeze paper outputs; missing directories:\n- "
            + "\n- ".join(missing)
        )

    files: list[dict] = []

    for directory in FROZEN_DIRS:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                files.append(file_record(path))

    provenance = []
    for path in PROVENANCE_FILES:
        if path.exists():
            provenance.append(file_record(path))

    # Hash the ordered list of individual file hashes to produce one stable
    # fingerprint for the complete paper result set.
    aggregate = hashlib.sha256()
    for record in sorted(files, key=lambda x: x["path"]):
        aggregate.update(record["path"].encode("utf-8"))
        aggregate.update(record["sha256"].encode("ascii"))

    manifest = {
        "freeze_name": "SignalRank-RAG paper proof-of-mechanism freeze",
        "freeze_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": None,
        "git_note": (
            "Git was intentionally disabled during the experiment sprint. "
            "Populate git_commit after repository Git metadata is restored "
            "and this frozen result set is committed without modification."
        ),
        "paper_scope": {
            "formal_experiments": ["E1", "E2", "E3", "E4_v6"],
            "analysis_layers": [
                "metric_pathology",
                "independent_generator_replication",
                "external_metric_validation",
            ],
            "scale": "controlled proof-of-mechanism",
            "n_claims": 6,
        },
        "frozen_directories": [p.as_posix() for p in FROZEN_DIRS],
        "n_files": len(files),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": files,
        "provenance_files": provenance,
        "exclusions": [
            "pilot experiments",
            "failed/development E4 variants",
            "temporary RAGAS environments",
            "non-paper benchmark outputs",
        ],
    }

    OUTPUT_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("PAPER RESULT FREEZE COMPLETE")
    print("=" * 78)
    print(f"Manifest:          {OUTPUT_PATH}")
    print(f"Frozen dirs:       {len(FROZEN_DIRS)}")
    print(f"Frozen files:      {len(files)}")
    print(f"Aggregate SHA-256: {manifest['aggregate_sha256']}")
    print()
    print("Do not modify frozen result files after this point.")
    print("When Git is restored, commit these outputs + the manifest together.")


if __name__ == "__main__":
    main()
