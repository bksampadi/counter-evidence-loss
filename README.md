# Counter-Evidence Loss in Retrieval-Augmented Generation

What happens when evidence capable of correcting an answer exists, but does not reach the generator?

<p align="center">
  <img src="figures/main_result.svg" alt="Main result" width="850">
</p>

Across six matched claims, correctness fell from **6/6** when counter-evidence was exposed to **0/6** when it was not exposed, and **0/6** when it was absent.

All **12 answers** in the two counter-evidence-loss conditions were incorrect, yet mean **RAGAS Faithfulness** remained **1.000** in NOT EXPOSED and **0.917** in ABSENT.

A RAG answer can therefore be fully supported by the context it sees and still be wrong because evidence capable of changing the answer never reached the generator.

The same correctness pattern was reproduced with a second generator family.

**Technical report:** [`paper/technical_report.md`](paper/technical_report.md)

This is a proof-of-concept study using six synthetic claims. It isolates a failure mechanism under controlled conditions; it does not estimate its prevalence in deployed RAG systems.

## What this repository contains

This is the full experimental record. The compact benchmark release, covering Experiments 1–2 and their validations, is [Counter-Evidence-RAG](https://github.com/bksampadi/Counter-Evidence-RAG) ([DOI 10.5281/zenodo.22810910](https://doi.org/10.5281/zenodo.22810910)).

| Component | Scripts | Results |
| --- | --- | --- |
| Benchmark: six synthetic claims and corpora | `scripts/Exp1/prepare_formal_corpora.py` | `benchmarks/Exp1/` |
| E1 — retrieval and exposure | `scripts/Exp1/` | `benchmark_results/Exp1/` |
| E2 — generation under three evidence conditions | `scripts/Exp2/` | `benchmark_results/Exp2/` |
| E3 — retrieval cutoff and wrong-entity control | `scripts/Exp3/` | `benchmark_results/Exp3/` |
| E4 — opposition-aware mitigation (v6) | `scripts/Exp4/*_v6.py` | `benchmark_results/Exp4/formal_v6/`, `preflight_v6/` |
| Metric-pathology analysis | `scripts/Exp4/analyze_metric_pathology.py` | `benchmark_results/metric_pathology/` |
| Independent generator replication | `scripts/Validation/replicate_independent_generator.py` | `benchmark_results/independent_generator_replication/` |
| External metric and blind sufficiency validation | `scripts/Validation/validate_external_metrics.py` | `benchmark_results/external_metric_validation/` |
| Restoration analyses | `scripts/Validation/run_counter_evidence_*.py` | `benchmark_results/counter_evidence_*/` |
| Superseded E4 development runs (not reported) | `scripts/Exp4/superseded/` | `benchmark_results/Exp4/superseded/` |

[`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) records the run order, the gate each step had to pass, and why Experiment 4 went through six versions.

`benchmark_results/PAPER_FREEZE_MANIFEST.json` lists SHA-256 hashes for the frozen result files.

## Reproducing

The scripts run inside [SignalRank-RAG](https://github.com/bksampadi/SignalRank-RAG), the RAG system used for retrieval and generation.

1. Clone SignalRank-RAG at commit `bc6dc9d91c653bae782be72fc0220fd77858e027` (version 0.2.0) and install it with `uv sync`.
2. Copy `benchmarks/`, `benchmark_results/` and `scripts/` from this repository into the root of that checkout. Later stages read earlier frozen outputs from `benchmark_results/`, and rerunning a stage overwrites its output folder, so work on a copy.
3. Set `GROQ_API_KEY`.
4. Run the commands in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) from the checkout root.

See [`requirements.txt`](requirements.txt) for the recorded package versions. All model calls go through a hosted API, so reruns may not reproduce the frozen outputs exactly; the frozen outputs are the reference.

## AI assistance

ChatGPT (OpenAI) and Claude (Anthropic) were used for coding, debugging, and manuscript editing. The author reviewed and approved all outputs.

## License

Code in `scripts/` is released under the [MIT License](LICENSE). Benchmark data, results, documentation, figures and the technical report are released under [CC BY 4.0](LICENSE-DATA).

## Citation

See [`CITATION.cff`](CITATION.cff).
