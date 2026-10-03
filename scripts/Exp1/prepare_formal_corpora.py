import json
import shutil
from pathlib import Path

STIMULUS_ROOT = Path("benchmarks/Exp1/stimuli_v1")
CLAIMS_PATH = STIMULUS_ROOT / "claims.json"
CORPUS_ROOT = Path("benchmarks/Exp1/corpora/formal_v1")

CORPUS_STATES = ("available", "absent")
EXPECTED_NEUTRALS_PER_CLAIM = 4


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def copy_document(
    *,
    source: Path,
    target_dir: Path,
    claim_id: str,
    role: str,
    manifest_rows: list[dict],
) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing stimulus file: {source}")

    destination = target_dir / source.name
    shutil.copyfile(source, destination)

    manifest_rows.append(
        {
            "filename": source.name,
            "claim_id": claim_id,
            "role": role,
        }
    )


def main() -> None:
    stimuli = load_json(CLAIMS_PATH)
    stimulus_version = stimuli["stimulus_version"]
    claims = stimuli["claims"]

    expected_claim_ids = {
        "false_01",
        "false_02",
        "false_03",
        "true_01",
        "true_02",
        "true_03",
    }
    observed_claim_ids = {claim["claim_id"] for claim in claims}

    if observed_claim_ids != expected_claim_ids:
        raise RuntimeError(
            "Formal E1 expects the frozen six-claim set. "
            f"Observed: {sorted(observed_claim_ids)}"
        )

    if CORPUS_ROOT.exists():
        shutil.rmtree(CORPUS_ROOT)

    for state in CORPUS_STATES:
        target_dir = CORPUS_ROOT / state
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows: list[dict] = []

        for claim in claims:
            claim_id = claim["claim_id"]

            support = STIMULUS_ROOT / f"{claim_id}__support.txt"
            critical = STIMULUS_ROOT / f"{claim_id}__critical.txt"
            replacement = STIMULUS_ROOT / f"{claim_id}__replacement.txt"
            neutrals = [
                STIMULUS_ROOT / f"{claim_id}__neutral_{i}.txt"
                for i in range(1, EXPECTED_NEUTRALS_PER_CLAIM + 1)
            ]

            copy_document(
                source=support,
                target_dir=target_dir,
                claim_id=claim_id,
                role="misleading_support",
                manifest_rows=manifest_rows,
            )

            if state == "available":
                copy_document(
                    source=critical,
                    target_dir=target_dir,
                    claim_id=claim_id,
                    role="corrective",
                    manifest_rows=manifest_rows,
                )
            else:
                copy_document(
                    source=replacement,
                    target_dir=target_dir,
                    claim_id=claim_id,
                    role="replacement",
                    manifest_rows=manifest_rows,
                )

            for neutral in neutrals:
                copy_document(
                    source=neutral,
                    target_dir=target_dir,
                    claim_id=claim_id,
                    role="neutral",
                    manifest_rows=manifest_rows,
                )

        manifest = {
            "experiment": "formal_experiment_1",
            "stimulus_version": stimulus_version,
            "corpus_state": state,
            "n_documents": len(manifest_rows),
            "documents": manifest_rows,
        }

        (target_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        expected_count = len(claims) * (2 + EXPECTED_NEUTRALS_PER_CLAIM)
        if len(manifest_rows) != expected_count:
            raise RuntimeError(
                f"{state}: expected {expected_count} documents, "
                f"got {len(manifest_rows)}"
            )

        filenames = {row["filename"] for row in manifest_rows}

        for claim in claims:
            claim_id = claim["claim_id"]
            support_name = f"{claim_id}__support.txt"
            critical_name = f"{claim_id}__critical.txt"
            replacement_name = f"{claim_id}__replacement.txt"

            if support_name not in filenames:
                raise RuntimeError(f"{state}: missing {support_name}")

            if state == "available":
                if critical_name not in filenames:
                    raise RuntimeError(f"available: missing {critical_name}")
                if replacement_name in filenames:
                    raise RuntimeError(
                        f"available: replacement leaked into corpus: "
                        f"{replacement_name}"
                    )
            else:
                if critical_name in filenames:
                    raise RuntimeError(
                        f"absent: corrective evidence leaked into corpus: "
                        f"{critical_name}"
                    )
                if replacement_name not in filenames:
                    raise RuntimeError(f"absent: missing {replacement_name}")

        print(
            f"{state.upper():<10} "
            f"{len(manifest_rows)} documents -> {target_dir}"
        )

    print()
    print("Formal E1 corpora prepared successfully.")
    print(f"Stimulus version: {stimulus_version}")


if __name__ == "__main__":
    main()
