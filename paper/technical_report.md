# Counter-Evidence Loss in Retrieval-Augmented Generation

**Bharath Sampadi**  
Independent Researcher

## Abstract

Retrieval-augmented generation (RAG) systems are often evaluated against the evidence visible to the generator. This creates a blind spot when evidence capable of changing an answer exists but does not reach that context.

We tested this with six synthetic claims under three matched evidence conditions. When counter-evidence was exposed, all six claims were answered correctly. When the same counter-evidence remained available but was withheld from the generator-visible context, correctness fell to **0/6**; removing it from the corpus also produced **0/6** correctness.

Two evaluation signals moved in the wrong direction under this failure. Retrieved-evidence agreement increased from **0.5 to 1.0**, and a reference-free sufficiency judge classified **18/18 contexts as sufficient**, including all 12 contexts associated with incorrect answers. Mean **RAGAS Faithfulness** was **1.000** in NOT EXPOSED and **0.917** in ABSENT despite **0/6 correctness** in both conditions.

The 6/6 → 0/6 → 0/6 correctness pattern replicated with a second generator family. These results show that local support within retrieved context does not establish that the evidence set itself was adequate.

## 1. Background

RAG conditions generation on retrieved evidence rather than parametric memory alone [1]. Evaluation therefore often asks whether retrieved material is relevant and whether the answer is faithful to that material [2].

That framing is useful, but it has an observational boundary: an evaluator cannot directly assess evidence it never receives.

Related work approaches different parts of this problem. Glockner et al. showed that missing counter-evidence creates unrealistic assumptions in automated fact-checking [3]. Joren et al. introduced *sufficient context* to distinguish answerable from insufficient retrieved contexts [4]. CUE-R uses evidence interventions such as removal and replacement to measure the operational utility of individual retrieved items [5].

The present study isolates a narrower RAG failure: **answer-changing counter-evidence remains available to the system but is not exposed to the generator**. The question is not whether deleting useful evidence can reduce accuracy, but whether the resulting one-sided context can simultaneously look *more* coherent and *sufficient* while producing the wrong answer.

## 2. Methods

### 2.1 Benchmark and evidence conditions

The benchmark contains six synthetic binary claims: three with a predefined TRUE label and three with a predefined FALSE label. Synthetic entities were used to reduce contamination from real-world or parametric knowledge.

Each claim has:

- a supporting passage,
- a claim-matched critical passage containing answer-changing counter-evidence,
- neutral passages,
- matched replacement material.

Each claim was evaluated in three conditions:

| Condition | Counter-evidence in corpus | Visible to generator |
| --- | :---: | :---: |
| **EXPOSED** | Yes | Yes |
| **NOT EXPOSED** | Yes | No |
| **ABSENT** | No | No |

This produced 18 primary claim-condition instances.

In **NOT EXPOSED**, the critical passage remained indexed and retrievable but was deliberately excluded when the generator context was assembled. In **ABSENT**, it was removed from the corpus and replaced so that corpus size remained matched. All 18 manipulation checks passed before generation.

### 2.2 Retrieval

The formal retrieval run used the frozen SignalRank retrieval stack:

- hybrid BM25 + dense retrieval,
- FlashRank reranking (`ms-marco-MiniLM-L-12-v2`),
- candidate multiplier 4,
- generator-visible context depth **k = 5**.

The available corpus contained the claim-matched critical passage for both EXPOSED and NOT EXPOSED. The primary experiment therefore directly manipulated **exposure**, rather than conflating exposure with corpus availability.

A separate cutoff validation tested the same mechanism under native ranking, where a smaller generator-visible cutoff omitted the critical passage while leaving it retrievable.

### 2.3 Generation and correctness

Primary answers were generated with:

- **OpenAI GPT-OSS-20B**
- served through **Groq**
- temperature **0**
- fallback disabled.

The generator returned a binary TRUE/FALSE verdict. Correctness was determined by exact agreement with the predefined truth label stored in the frozen benchmark.

For independent replication, the frozen formal contexts were evaluated with:

- **Qwen3.6-27B**
- temperature **0**
- reasoning effort disabled.

This model was not used for stimulus selection or benchmark preflight.

### 2.4 Evaluation

The primary evaluation signals were:

**Reference correctness.** Binary verdict compared with the predefined truth label.

**RAGAS Faithfulness.** RAGAS 0.4.x (the restoration trace recorded 0.4.3) evaluated whether statements in the answer were supported by the generator-visible context. **GPT-OSS-120B**, served through Groq at temperature 0, was used as evaluator.

**Retrieved-evidence agreement.** The proportion of generator-visible evidence whose stance was consistent with the generated answer.

**Reference-free sufficiency.** GPT-OSS-120B received only the question and generator-visible context, with no reference answer, and classified whether the context was sufficient to answer.

### 2.5 Validation

To test whether faithfulness could respond once missing evidence became observable, a restoration analysis held each of the six incorrect NOT EXPOSED answers fixed and restored the omitted counter-evidence to the evaluator-visible context. The answer was not regenerated.

Experimental inputs and primary outputs were frozen before the independent-generator replication and restoration analyses.

## 3. Results

### 3.1 Counter-evidence exposure determined correctness

| Condition | Correct | Accuracy |
| --- | ---: | ---: |
| **EXPOSED** | **6/6** | **1.00** |
| **NOT EXPOSED** | **0/6** | **0.00** |
| **ABSENT** | **0/6** | **0.00** |

The critical result is NOT EXPOSED: the corrective evidence still existed in the indexed corpus, but withholding it from the generator context reversed the answer on every claim.

### 3.2 Agreement increased while correctness collapsed

Retrieved-evidence agreement increased from:

**0.5 → 1.0**

under counter-evidence loss.

The direction is mechanically understandable: removing dissenting evidence makes the surviving context more homogeneous. The important observation is that this apparently favourable consistency signal became maximal precisely when all six answers were wrong.

### 3.3 The sufficiency judge failed to distinguish the censored contexts

The reference-free sufficiency judge classified:

**18 / 18 contexts as sufficient.**

This included all 12 NOT EXPOSED and ABSENT contexts associated with incorrect answers.

The censored contexts contained enough coherent evidence to support *an* answer. They lacked the evidence needed to determine that the supported answer was not warranted.

### 3.4 Incorrect answers remained highly faithful to visible context

In the external evaluation run, mean RAGAS Faithfulness was 1.000 in NOT EXPOSED and 0.917 in ABSENT, despite 0/6 correctness in both conditions. The lower ABSENT mean reflects one frozen evaluation case that received a faithfulness score of 0.500.

This is not a contradiction in the metric. The answers were supported by the context the evaluator could see.

The result demonstrates the distinction between:

- **local faithfulness** — support relative to visible context;
- **evidence adequacy** — whether the system exposed the evidence needed to challenge that answer.

### 3.5 The correctness pattern replicated

Qwen3.6-27B reproduced the same pattern on the frozen contexts:

| Condition | Correct |
| --- | ---: |
| **EXPOSED** | **6/6** |
| **NOT EXPOSED** | **0/6** |
| **ABSENT** | **0/6** |

The replication model had not been used to construct or preflight the benchmark.

### 3.6 Restoring counter-evidence made the conflict observable

In the matched paired trace, the six incorrect NOT EXPOSED answers were held fixed and evaluated twice with the same RAGAS 0.4.3 configuration: once against the original hidden context and once with the omitted counter-evidence restored. Mean faithfulness decreased from **0.917 to 0.000**, with a decrease in all six cases. The hidden-context baseline was 0.917 rather than the 1.000 recorded in the external evaluation run because one case (false_02) scored 0.5 on rerun, reflecting evaluator run-to-run variability.

This supports the interpretation that the evaluator can react to counter-evidence once it becomes visible; the original blind spot arose because the necessary evidence was outside its input.

## 4. Discussion

Counter-evidence loss produces a simple but important failure mode: a RAG system can be locally coherent, judged sufficient, and fully faithful to the context it sees while still producing the wrong answer.

The result does not imply that faithfulness is a defective metric. Faithfulness measures support relative to an observed context. The broader systems question is whether the retrieval process exposed enough evidence for that local judgment to be trustworthy.

A useful evaluation stack therefore needs to distinguish at least two levels:

1. **Is the answer supported by the retrieved context?**
2. **Was the retrieved context adequate to test the answer against plausible opposing evidence?**

The second question cannot be recovered from local support alone.

## 5. Limitations

This is a proof-of-concept study with **six synthetic claims**. The controlled design isolates the mechanism but does not estimate its prevalence in natural corpora or deployed RAG systems.

The NOT EXPOSED condition in the primary experiment is an explicit exposure intervention: the critical passage remains retrievable but is removed before generator-context assembly. This isolates the causal effect cleanly but is not itself a prevalence model of naturally occurring ranking failures.

The sufficiency result uses one judge model and prompt configuration. Larger benchmarks, natural corpora, multiple retrieval stacks, and multiple sufficiency evaluators are needed before making broader claims.

## AI assistance

ChatGPT (OpenAI) and Claude (Anthropic) were used for coding, debugging, and manuscript editing. The author reviewed and approved all outputs.

## References

1. Lewis, P. et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** NeurIPS, 2020.
2. Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. **RAGAs: Automated Evaluation of Retrieval Augmented Generation.** EACL System Demonstrations, 2024.
3. Glockner, M., Hou, Y., & Gurevych, I. **Missing Counter-Evidence Renders NLP Fact-Checking Unrealistic for Misinformation.** EMNLP, 2022.
4. Joren, H. et al. **Sufficient Context: A New Lens on Retrieval Augmented Generation Systems.** ICLR, 2025.
5. Jain, S. & Vedam, V. N. **CUE-R: Beyond the Final Answer in Retrieval-Augmented Generation.** arXiv:2604.05467, 2026.
