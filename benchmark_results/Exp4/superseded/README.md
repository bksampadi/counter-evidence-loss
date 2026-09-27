# Superseded Experiment 4 runs

These outputs come from development versions of Experiment 4 that were replaced before the reported run. They are kept for transparency. They are **not** reported results and are **not** covered by `benchmark_results/PAPER_FREEZE_MANIFEST.json`, which lists "failed/development E4 variants" among its exclusions.

The reported Experiment 4 is **v6**: `benchmark_results/Exp4/formal_v6/`, gated by `benchmark_results/Exp4/preflight_v6/`.

Some records here carry `"formal_experiment": true`. That flag was set by the runner at execution time; it does not mean the run is part of the reported study.

| Folder | Produced by | Original path | Outcome |
| --- | --- | --- | --- |
| `v1_preflight/` | `scripts/Exp4/superseded/preflight_counterevidence_verifier.py` | `benchmark_results/Exp4/preflight/` | Preflight failed 22/24 (false positives on `false_01` target neutral and `false_03` target replacement). No v1 formal data were collected. |
| `v5_preflight/` | `scripts/Exp4/superseded/preflight_candidate_gate_v5.py` | `benchmark_results/Exp4/preflight_v5/` | Preflight passed 30/30. |
| `v5_formal/` | `scripts/Exp4/superseded/run_formal_exp4_v5.py` | `benchmark_results/Exp4/formal/` | Completed: 18 cases, 18/18 expected policy actions. Superseded by v6 (see below). |

`v5_formal/` is attributed to v5 because runners v1–v5 all wrote to the same output folder and the v4 runner differs from v5 only in a regex that cannot parse the canonical one-word outputs recorded here.

Outputs of the v2, v3 and v4 preflights were not retained.

**Why v6 replaced v5.** v1–v5 tested only incorrect baselines. A policy that abstains whenever no opposing evidence is found could therefore look perfectly safe without ever being tested on a correct baseline. v6 added positive-baseline controls and compared two explicit policies.

The superseded scripts are kept unmodified, so their hard-coded output paths refer to the original locations listed above. See `docs/EXPERIMENT_LOG.md` for the full v1–v6 history.
