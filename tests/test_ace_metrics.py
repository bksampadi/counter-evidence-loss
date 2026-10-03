from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "Analysis" / "compute_ace_metrics.py"


def load_ace_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "compute_ace_metrics",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load ACE metric module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ace_recovers_expected_condition_pattern() -> None:
    ace = load_ace_module()

    results = [
        *ace.compute_exp1(),
        *ace.compute_exp3(),
    ]
    summary = ace.summarize(results)

    assert summary["n_cases"] == 42

    states = summary["by_condition"]

    expected = {
        "Exp1:exposed": (1.0, 1.0),
        "Exp1:available_not_exposed": (1.0, 0.0),
        "Exp1:absent": (0.0, 0.0),
        "Exp3:corrective_exposed_k2": (1.0, 1.0),
        "Exp3:corrective_below_k1": (1.0, 0.0),
        "Exp3:corrective_absent_k1": (0.0, 0.0),
        "Exp3:wrong_entity_control_k2": (1.0, 0.0),
    }

    for key, (ace_a, ace_e) in expected.items():
        assert states[key]["n"] == 6
        assert states[key]["mean_ace_a"] == ace_a
        assert states[key]["mean_ace_e_at_k"] == ace_e


def test_wrong_entity_correction_is_not_target_ace() -> None:
    ace = load_ace_module()

    rows = [
        row
        for row in ace.compute_exp3()
        if row["condition"] == "wrong_entity_control_k2"
    ]

    assert len(rows) == 6

    for row in rows:
        assert row["ace_available"] is True
        assert row["ace_exposed"] is False
        assert row["reference_ace_file"] not in row["context_source_files"]
        assert row["wrong_entity_claim_id"] != row["claim_id"]


def test_ace_exposure_never_occurs_when_ace_is_unavailable() -> None:
    ace = load_ace_module()

    results = [
        *ace.compute_exp1(),
        *ace.compute_exp3(),
    ]

    for row in results:
        if not row["ace_available"]:
            assert row["ace_exposed"] is False