# Experiment log

This log records how the experiments were run, in order, including each gate that had to pass before the next step. It consolidates the working notes written during the experiment sprint.

All generation and judging calls used Groq. Unless stated otherwise the model was `openai/gpt-oss-20b` at temperature 0, with no fallback models. Commands are run from the root of a SignalRank-RAG checkout (see [Reproducing](../README.md#reproducing)).

Naming: `available_not_exposed` in the scripts and outputs is the NOT EXPOSED condition in the technical report. The "critical" passage is the claim-matched counter-evidence (corrective evidence); the "support" passage is the misleading support.

## Formal Experiment 1 — retrieval and exposure

**Gate.** The stimulus anti-leak preflight must pass 36/36 for the same stimulus version.

```bash
uv run python scripts/Exp1/preflight_validate_stimuli.py
uv run python scripts/Exp1/prepare_formal_corpora.py
uv run python scripts/Exp1/run_formal_exp1.py
```

Recorded: preflight 36/36 (`benchmark_results/Exp1/preflight/`); formal acceptance passed (`benchmark_results/Exp1/formal/experiment_1_summary.json`).

Acceptance criteria, all 18 claim × condition checks:

- **EXPOSED:** misleading support visible in the 5-passage context; corrective evidence in the corpus, in the original top 5, and visible.
- **AVAILABLE_NOT_EXPOSED:** misleading support visible; corrective evidence in the corpus and in the original top 5, but withheld from the 5-passage context.
- **ABSENT:** misleading support visible; corrective evidence absent from the corpus and from the ranked results.

Experiment 2 does not proceed unless formal acceptance is true.

## Formal Experiment 2 — generation

**Gate: strict anti-leak preflight.** The first preflight showed source filenames such as `__support.txt` and `__critical.txt` to the model. Retrieval in Experiment 1 is unaffected, but Experiment 2 is gated on a strict repeat in which the model sees only neutral labels ("Evidence 1", "Evidence 2", …).

```bash
uv run python scripts/Exp1/preflight_validate_stimuli_strict.py
uv run python scripts/Exp2/run_formal_exp2.py
```

Recorded: strict preflight 36/36 (`benchmark_results/Exp1/preflight_strict/`).

Experiment 2 consumes the 18 frozen contexts from Experiment 1 (no retrieval rerun), shows no filenames or evidence-role labels, uses 5 passages per context over 6 claims × 3 conditions, and records condition accuracy, paired verdict switches and corrective-exposure rescue.

## Formal Experiment 3 — retrieval cutoff / boundary robustness

Experiment 1 produced the same rank structure for all six claims: misleading support at rank 1, corrective evidence at rank 2. Experiment 3 uses that frozen boundary directly, with no retrieval rerun.

| Condition | Context |
| --- | --- |
| `corrective_exposed_k2` | top 2: target support + target corrective evidence |
| `corrective_below_k1` | available corpus, k = 1: target support only; corrective evidence exists at rank 2 |
| `corrective_absent_k1` | absent corpus, k = 1: target support only; corrective evidence unavailable |
| `wrong_entity_control_k2` | target support + another claim's correction-shaped critical passage |

The wrong-entity control tests whether generic correction/audit language by itself causes rescue.

**Gate.** The wrong-entity preflight must pass 12/12.

```bash
uv run python scripts/Exp3/preflight_wrong_entity_control.py
uv run python scripts/Exp3/run_formal_exp3.py
```

Recorded: preflight 12/12 (`benchmark_results/Exp3/preflight/`).

## Formal Experiment 4 — opposition-aware mitigation

Experiments 2 and 3 showed that a misleading rank-1 passage produces the wrong answer unless target-specific corrective evidence crosses the context boundary. Experiment 4 tests a mitigation: inspect candidate passages for target-specific opposing evidence, admit one and regenerate if found, and otherwise abstain or retain the baseline depending on the policy. In the correctable scenarios, the candidate is the rank-2 critical passage from the frozen Experiment 1 ranking; the correct-baseline controls use constructed candidate sets. Experiment 4 reuses frozen Experiment 1 rankings and Experiment 3 baseline outputs and does not rerun retrieval..

Experiment 4 went through six versions. Only **v6** is reported (technical report §2.6 and §3.5). Superseded outputs are in `benchmark_results/Exp4/superseded/` and superseded scripts in `scripts/Exp4/superseded/`.

| Version | Change | Outcome |
| --- | --- | --- |
| v1 | Bespoke `DIRECT_COUNTEREVIDENCE` verifier. | Preflight failed 22/24 (false positives on `false_01` target neutral and `false_03` target replacement). No formal data collected. |
| v2 | Replaced the verifier with the tri-state evidential test TRUE / FALSE / INSUFFICIENT, via strict JSON output. Admit a candidate only if its verdict is TRUE or FALSE and opposes the provisional verdict, so neutral, replacement and wrong-entity passages fail closed as INSUFFICIENT. 30-check preflight. | The gate behaved correctly through 25 consecutive checks, then Groq returned a correct INSUFFICIENT rationale that failed strict JSON serialization. Outputs not retained. |
| v3 | Same gate; added a narrow retry (at most 3 identical attempts, only when no parseable result was returned; retry count recorded). | One deterministic case repeatedly reasoned to the correct INSUFFICIENT but failed strict JSON serialization at temperature 0. Outputs not retained. |
| v4 | Changed only the transport: the gate replies with exactly one word (TRUE, FALSE or INSUFFICIENT), parsed by a deterministic regex; raw outputs stored. | The regex contained escaped backslashes and looked for a literal `\s`, so valid outputs could not be parsed. Outputs not retained. |
| v5 | Regex fix only: `^\s*(TRUE\|FALSE\|INSUFFICIENT)[.!]?\s*$`. | Preflight passed 30/30; formal run completed (18/18 expected actions). **Superseded** — see below. |
| v6 | Added positive-baseline controls and compared two explicit policies using the same evidence-admission calls. Preflight made bidirectional (36 checks). | Preflight passed 36/36. **Reported run.** |

**Why v6 replaced v5.** v1–v5 tested only incorrect baselines. A policy that abstains whenever no opposition is found could therefore look perfectly safe without ever being tested on a correct baseline.

v6 scenarios:

1. `wrong_baseline_valid_correction` — baseline wrong; target correction exists. Desired: answer after revision.
2. `wrong_baseline_no_correction` — baseline wrong; no valid opposition. Desired: abstain.
3. `correct_baseline_no_opposition` — baseline correct; no opposing evidence. Desired: answer (retain baseline).
4. `correct_baseline_stale_opposition` — baseline correct; stale, superseded same-claim evidence argues the other way. Desired: answer correctly after conflict resolution.

v6 policies:

- `conservative_abstain` — if opposition is found, regenerate and answer; otherwise abstain.
- `retain_if_no_opposition` — if opposition is found, regenerate and answer; otherwise keep the baseline answer.

Metrics: coverage, false answers, false abstentions, selective accuracy, answered risk, desired-outcome rate.

```bash
uv run python scripts/Exp4/preflight_candidate_gate_v6.py
uv run python scripts/Exp4/run_formal_exp4_v6.py
```

## Metric pathology analysis

Computes context-level signals (corrective recall, context support rate, visible target-evidence agreement, reference-sufficient context) for the frozen Experiment 2 and 3 outputs. No model calls.

```bash
uv run python scripts/Exp4/analyze_metric_pathology.py
```

Output: `benchmark_results/metric_pathology/`.

metric_pathology is a developmental directory name. The analysis characterises metric behaviour under the controlled evidence intervention; the name does not imply that the evaluated metrics are defective.

## Validation package

Neither validation changes any stimuli.

**Independent generator replication.** Model `qwen/qwen3.6-27b`, with no stimulus selection or preflight on Qwen, run on the exact frozen Experiment 2 and 3 contexts with the same binary task and no retrieval rerun. It uses `reasoning_effort="none"` (non-thinking mode), fixed before any replication verdicts were observed, to avoid a reasoning-token/output-budget interaction on Groq.

```bash
uv run python scripts/Validation/replicate_independent_generator.py
```

**External metric and blind sufficiency validation.** RAGAS Faithfulness with openai/gpt-oss-120b as evaluator, plus a reference-free sufficiency judge using the same model. The generator output is a binary TRUE/FALSE verdict; for RAGAS it is rendered as the fixed sentence For the factual question '{query}', the answer is {verdict}. rather than evaluated as free-form answer prose. The sufficiency judge sees only the question and generator-visible context, never the reference answer, and is explicitly instructed to judge apparent answerability from the supplied context only rather than speculate about hidden evidence.

```bash
uv run --with "ragas==0.4.3" --with groq python scripts/Validation/validate_external_metrics.py
```

The original external-validation run recorded the evaluator model but not the installed RAGAS package version. The reproduction command pins RAGAS 0.4.3 because that version is recorded for the later restoration traces and in the current dependency record; this should not be read as evidence that the original external-validation run used 0.4.3.


Outputs: `benchmark_results/independent_generator_replication/`, `benchmark_results/external_metric_validation/`.

Checkpoint behaviour. External validation and the restoration scripts resume completed cases from *_checkpoint.jsonl files. The archival repository retains these checkpoint files as part of the experimental record. To make fresh evaluator calls during reproduction, delete the relevant checkpoint file only in the local reproduction copy before running that stage. Otherwise the script may reuse completed cases without issuing new evaluator calls.


## Restoration analyses

These hold the six incorrect AVAILABLE_NOT_EXPOSED answers from Experiment 2 fixed and restore the omitted counter-evidence to the evaluator context. They require `GROQ_API_KEY` and RAGAS but not SignalRank-RAG.

- `scripts/Validation/run_counter_evidence_restoration.py` → `benchmark_results/counter_evidence_restoration/`
- `scripts/Validation/run_counter_evidence_restoration_trace.py` → `benchmark_results/counter_evidence_restoration_trace/` (records statement-level verdicts, the RAGAS version and prompt fingerprints)
- `scripts/Validation/run_counter_evidence_baseline_trace.py` → `benchmark_results/counter_evidence_paired_trace/`

```bash
uv run --with "ragas==0.4.3" --with groq python scripts/Validation/run_counter_evidence_restoration.py

uv run --with "ragas==0.4.3" --with groq python scripts/Validation/run_counter_evidence_restoration_trace.py

uv run --with "ragas==0.4.3" --with groq python scripts/Validation/run_counter_evidence_baseline_trace.py
```

Run the restoration trace before the baseline trace. The baseline trace checks the RAGAS version and prompt fingerprints recorded by the restoration trace so that the paired comparison uses the same evaluation configuration.


The `baseline_faithfulness` of 1.000 in the restoration and restoration-trace outputs is taken from the external evaluation run, not re-measured. The paired trace re-measures the hidden-context baseline with the same RAGAS configuration (mean 0.917); the reported matched comparison uses the paired trace.

These outputs are not listed in the freeze manifest.

## Reproduction notes

- Some evaluator calls retry transient rate-limit failures; these retries do not change the prompt or model configuration.
- The external-validation script contains a compatibility fallback between RAGAS import paths.
- The Experiment 1 stimulus preflight model can be overridden with `SIGNALRANK_EXP1_PREFLIGHT_MODEL`; the reported run used the documented GPT-OSS-20B configuration.
- Hosted embedding, reranker and model endpoints may change over time, so the frozen outputs remain the reference record.

## Freeze manifest

`benchmark_results/PAPER_FREEZE_MANIFEST.json` records SHA-256 hashes for the frozen result files. Its `git_commit` is `null` because Git was disabled during the experiment sprint. The manifest was left unedited after the freeze, so its `git_note` asking for `git_commit` to be filled in is out of date: the frozen files were first committed, unmodified, in `a7c9eb2`. The manifest was written after the independent-generator replication and external metric validation, which it covers, so it does not by itself establish run order. Verify files against the listed hashes; do not rerun `scripts/freeze_paper_results.py`, which rewrites the manifest.

## Not included

Earlier exploratory pilot experiments are not part of this record, matching the manifest's exclusions.
