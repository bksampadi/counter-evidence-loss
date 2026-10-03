# Counter-Evidence Loss in Retrieval-Augmented Generation

**Bharath Sampadi**  
Independent Researcher

## Abstract

Retrieval-augmented generation (RAG) is usually evaluated against the evidence the generator was given. This study asks what those evaluations report when evidence that would change the answer exists in the corpus but does not reach the generator.

Six synthetic binary claims were each paired with a misleading supporting passage and a later correction that supersedes it. With the correction in the generator's context, all six answers were correct. With the correction indexed and retrievable but left out of the context, none were, and removing it from the corpus also gave 0/6. This drop is expected by design, since the generator is told to judge from the evidence it is given.

The evaluation signals did not register the failure. Mean RAGAS Faithfulness was 1.000 for the six wrong NOT EXPOSED answers and 0.917 for the six wrong ABSENT answers, compared with 0.917 for the six correct EXPOSED answers. A reference-free sufficiency judge rated all 18 contexts sufficient. The same correctness pattern appeared when the correction fell just below a rank cutoff instead of being withheld, and a second generator family (Qwen3.6-27B) reproduced all 42 correctness results across E2 and E3; evaluator, restoration and mitigation results were not replicated with Qwen. When the omitted correction was restored to the evaluator's context, faithfulness for the same wrong answers fell from 0.917 to 0.000, so the evaluator does respond to the conflict when it can see it.

A mitigation that tests candidate passages for claim-specific opposing evidence recovered all six correctable answers in the controlled mitigation scenarios. What to do when no opposing evidence is found is a risk–coverage trade-off: abstaining avoided every false answer but also abstained on five of the six correct answers that had nothing opposing them.

## 1. Background

RAG conditions generation on retrieved evidence rather than parametric memory alone [1]. Evaluation therefore often asks whether the retrieved material is relevant and whether the answer is faithful to it [2]. Both questions concern the context the generator received, and neither can say anything about evidence that was never retrieved or never passed on.

Related work covers several adjacent failure modes. Glockner et al. showed that missing counter-evidence makes automated fact-checking unrealistic for misinformation [3]. Joren et al. introduced *sufficient context* to distinguish answerable from insufficient retrieved contexts [4]. CUE-R uses evidence interventions such as removal and replacement to measure the operational utility of individual retrieved items [5]. More recent work has examined related evidence-boundary failures: EviScope uses paired counterfactual conditions that add, remove, distract or contradict evidence [6]; LayerRAG-Bench reports false positives from groundedness-only evaluation under stale and wrong-session evidence [7]; and CROWN-QA studies cases where models treat incomplete evidence as sufficient coverage for a query [8]. RAGChecker provides retrieval-level diagnostics, including claim recall, for measuring whether expected information was recovered by the retriever [9].

The present study isolates a narrower comparison: answer-changing evidence remains available in the same corpus while only its exposure to the generator changes. The AVAILABLE-BUT-NOT-EXPOSED condition therefore separates corpus availability from generator exposure without removing the corrective evidence from the retrieval collection.

This study isolates a narrower failure: counter-evidence that would change the answer remains available to the system but is not shown to the generator. Removing useful evidence will lower accuracy, so that is not the object of study. The aim is to see how standard evaluation signals score the resulting one-sided context, and whether it looks any less faithful, sufficient or consistent when the answer it produces is wrong.

## 2. Methods

### 2.1 Benchmark and evidence conditions

The benchmark contains six synthetic binary claims, three labelled TRUE and three labelled FALSE. The entities (for example a "Kestrelase" enzyme and an "Orinex" battery cell) are invented, so the generator cannot answer from parametric knowledge.

Each claim has seven passages:

| Passage | Content |
| --- | --- |
| Support | An original analysis that states the wrong answer. |
| Critical | A later audit correction that supersedes the support passage and states the correct answer. This is the counter-evidence. |
| Neutral (4) | Records about the same entity (setup, handling, documentation) with no outcome data. |
| Replacement | A further outcome-free record that takes the critical passage's place in the ABSENT corpus. |

Each claim was tested in three conditions:

| Condition | Critical passage in corpus | Critical passage in generator context |
| --- | :---: | :---: |
| EXPOSED | Yes | Yes |
| NOT EXPOSED | Yes | No |
| ABSENT | No | No |

NOT EXPOSED is called `available_not_exposed` in the scripts and result files. Six claims in three conditions give 18 claim–condition cases.

In NOT EXPOSED the critical passage stayed indexed and retrievable but was excluded when the generator context was assembled. In ABSENT it was removed from the corpus and replaced by the replacement passage, so both corpora hold 36 passages. All 18 manipulation checks passed before generation.

### 2.2 Retrieval

Retrieval used the SignalRank-RAG stack at a fixed commit: BM25 and dense retrieval (`all-mpnet-base-v2`) combined by reciprocal rank fusion, FlashRank reranking (`ms-marco-MiniLM-L-12-v2`) with a candidate multiplier of 4, and a generator context of k = 5 passages.

In the available corpus, all six claims ranked the same way, with the support passage first and the critical passage second. The EXPOSED context was therefore the support passage, the critical passage and three neutral passages. NOT EXPOSED dropped the critical passage and filled the fifth slot with the next-ranked passage. Because the critical passage was in the corpus in both conditions, the comparison between them changes exposure without changing corpus content.

### 2.3 Generation

Answers were generated with GPT-OSS-20B (OpenAI) served through Groq at temperature 0, with fallback models disabled. Passages were shown as "Evidence 1", "Evidence 2" and so on, with no filenames or role labels. The system prompt tells the model to judge the claim using only the supplied evidence and to return a TRUE or FALSE verdict. An answer is correct if the verdict matches the claim's predefined label.

A preflight with the same neutral labels confirmed that each passage type behaves as intended for every claim: the support passage alone produces the wrong verdict, the critical passage alone produces the correct one, and the neutral and replacement passages alone are judged insufficient (36/36 checks).

Under this instruction, a wrong answer in NOT EXPOSED and ABSENT is the expected response, because the only outcome-bearing passage in the context states the wrong answer. The design makes the correctness drop near-certain so that the evaluation signals can be examined on contexts known to produce wrong answers.

### 2.4 Evaluation signals

*Reference correctness.* The binary verdict compared with the predefined truth label.

*RAGAS Faithfulness.* The generator returned a binary TRUE/FALSE verdict rather than free-form answer prose. For evaluation, that verdict was rendered as the fixed sentence For the factual question '{query}', the answer is {verdict}. and scored against the generator-visible context with GPT-OSS-120B served through Groq at temperature 0. The main evaluation run recorded the evaluator model but not the RAGAS package version; the later restoration traces used RAGAS 0.4.3.

*Reference-free sufficiency.* GPT-OSS-120B received only the question and the generator-visible context, with no reference answer, and classified the context as SUFFICIENT or INSUFFICIENT to answer the question. The judge was explicitly instructed to assess apparent answerability from the supplied context only and not to ask whether hidden evidence might exist elsewhere.

*Target-evidence agreement.* Of the claim-specific passages visible to the generator (the support and critical passages; neutral passages are excluded), the fraction whose stance matches the generated answer. It is computed from the design labels without a model call.

### 2.5 Rank-cutoff experiment

NOT EXPOSED withholds the critical passage deliberately. A second experiment (Experiment 3 in the results folders) produces the same loss from rank position alone. Because the critical passage ranked second for every claim, a context of k = 1 contains only the support passage while the correction stays retrievable one rank below the cutoff. The experiment reused the frozen Experiment 1 rankings without rerunning retrieval:

| Condition | Generator context |
| --- | --- |
| k = 2, exposed | Support passage + critical passage |
| k = 1, below cutoff | Support passage only; critical passage at rank 2 |
| k = 1, absent | Support passage only; critical passage not in corpus |
| k = 2, wrong-entity control | Support passage + another claim's critical passage |

The wrong-entity control tests whether generic audit or correction wording is enough to change the verdict. Its preflight passed 12/12 checks.

### 2.6 Mitigation experiment

Experiment 4 tested a simple mitigation. Starting from a provisional answer, a gate examines candidate passages one at a time. For each, GPT-OSS-20B answers TRUE, FALSE or INSUFFICIENT using only that passage. A candidate is admitted if its verdict is decisive and opposes the provisional answer. The generator then answers again from the provisional evidence plus the admitted passage. The gate stops at the first admitted candidate. In the correctable condition, the candidate was the rank-2 critical passage from the frozen retrieval result; the correct-baseline control scenarios used constructed candidate sets rather than retrieval rankings.

Two policies differ only in what they do when no candidate is admitted: `conservative_abstain` abstains, and `retain_if_no_opposition` keeps the provisional answer. Both use the same gate calls. They were tested on four scenarios per claim, 24 cases in all:

| Scenario | Provisional answer | Candidates inspected | Desired outcome |
| --- | --- | --- | --- |
| Wrong baseline, correction available | Wrong (k = 1, below cutoff) | The rank-2 critical passage | Correct answer |
| Wrong baseline, no correction | Wrong (k = 1, absent) | Ranks 2–5 of the ABSENT ranking | Abstain |
| Correct baseline, no opposition | Correct (critical passage only) | The four neutral passages | Correct answer |
| Correct baseline, stale opposition | Correct (critical passage only) | The superseded support passage | Correct answer |

The reported design is the sixth version. Versions 1–4 failed their preflights, one on gate false positives and three on output parsing. Version 5 ran, but tested only wrong baselines, so it did not adequately test the cost of abstaining when no opposition is found. The history is recorded in `docs/EXPERIMENT_LOG.md`.

### 2.7 Replication and restoration

For replication, Qwen3.6-27B served through Groq answered the exact frozen contexts from the main and rank-cutoff experiments at temperature 0 with reasoning disabled, using a separately implemented but semantically matched binary TRUE/FALSE prompt. This model was not used to select or preflight any stimuli, and the reasoning setting was fixed before any of its verdicts were seen.

The restoration analysis held the six wrong NOT EXPOSED answers fixed and scored them with RAGAS Faithfulness twice: against their original context, and against the EXPOSED context, which includes the critical passage. The answers were not regenerated.

Both analyses read the frozen Experiment 1 contexts and Experiment 2 answers, whose SHA-256 hashes are listed in `benchmark_results/PAPER_FREEZE_MANIFEST.json`. Neither reran retrieval or changed any stimuli.

### 2.8 Answer-changing evidence diagnostics

To make the benchmark's distinction between evidence availability and evidence exposure explicit, two oracle-labelled diagnostics were computed from the experimental design.

**ACE-A (Answer-Changing Evidence Availability)** records whether the benchmark-labelled answer-changing passage exists in the corpus.

**ACE-E@k (Answer-Changing Evidence Exposure at k)** records whether that passage appears in the generator-visible top-k context.

These quantities are descriptive diagnostics for this controlled benchmark. The identity of the answer-changing passage is known by construction, so ACE-A and ACE-E@k do not solve the harder problem of discovering answer-changing evidence in an arbitrary corpus without privileged labels.

## 3. Results

### 3.1 Correctness and evaluation signals

| Condition | Correct | Mean RAGAS Faithfulness | Judged sufficient | Target-evidence agreement |
| --- | ---: | ---: | ---: | ---: |
| EXPOSED | 6/6 | 0.917 | 6/6 | 0.50 |
| NOT EXPOSED | 0/6 | 1.000 | 6/6 | 1.00 |
| ABSENT | 0/6 | 0.917 | 6/6 | 1.00 |

Every claim flipped from correct to incorrect when the critical passage was withheld, although it was still indexed and ranked second for the query. None of the three evaluation signals scored the wrong answers lower compared to the right ones.

Faithfulness was highest for the six wrong NOT EXPOSED answers. Each 0.917 mean comes from a single answer scored 0.5 (false_01 in EXPOSED, false_03 in ABSENT), and the other 16 answers scored 1.0. The wrong answers were fully supported by the context the evaluator saw, which is what faithfulness measures.

The sufficiency judge rated all 18 contexts sufficient, including the 12 that produced wrong answers. This is expected under the judge’s instruction: those contexts did contain a direct answer to the question, and the judge was told to assess only the supplied context rather than speculate about evidence that might exist elsewhere. The missing information was the superseding passage, which was outside the judge’s view.

Target-evidence agreement rose from 0.50 to 1.00. This follows from the design: once the critical passage is gone, the only claim-specific passage left is the support passage, and the answer agrees with it. The metric is included to show that a consistency-style signal reaches its maximum on exactly the contexts that produced wrong answers.

### 3.2 Rank cutoff and wrong-entity control

| Condition | GPT-OSS-20B | Qwen3.6-27B |
| --- | ---: | ---: |
| k = 2, exposed | 6/6 | 6/6 |
| k = 1, below cutoff | 0/6 | 0/6 |
| k = 1, absent | 0/6 | 0/6 |
| k = 2, wrong-entity control | 0/6 | 0/6 |

Cutting the context at k = 1 had the same effect as withholding the critical passage: all six answers were wrong, although the correction sat one rank below the cutoff. Pairing the support passage with another claim's correction rescued none of the six, so recovery in the exposed condition depends on claim-specific counter-evidence rather than on correction-style wording.

The metric-pathology analysis gives the same agreement values and the same design-derived critical-evidence-visibility labels for these conditions as for their Section 3.1 counterparts (`benchmark_results/metric_pathology/`). RAGAS Faithfulness and the sufficiency judge were run only on the Section 3.1 contexts.

### 3.3 Answer-changing evidence diagnostics

The controlled conditions separate availability from exposure:

| Condition | ACE-A | ACE-E@k | Correctness |
| --- | ---: | ---: | ---: |
| Evidence absent | 0.000 | 0.000 | 0/6 |
| Available but not exposed | 1.000 | 0.000 | 0/6 |
| Exposed | 1.000 | 1.000 | 6/6 |

The middle condition is the distinction of interest: answer-changing evidence exists in the corpus, but does not reach the generator.

The rank-cutoff experiment gives the same pattern. When the corrective passage sits at rank 2 but the generator receives only k = 1, ACE-A = 1.000 and ACE-E@1 = 0.000 while correctness remains 0/6.

These values follow from the controlled experimental labels and should not be interpreted as evidence that the identity of answer-changing passages can be inferred automatically in natural corpora.

### 3.4 Second generator

Qwen3.6-27B gave the same correctness as GPT-OSS-20B in every condition: 6/6, 0/6 and 0/6 in the main experiment, and the values in the table above for the rank-cutoff experiment.

### 3.5 Restoring the counter-evidence

Scored with the same RAGAS 0.4.3 configuration, the six wrong NOT EXPOSED answers had a mean faithfulness of 0.917 against their original context and 0.000 against the context with the critical passage restored. Faithfulness fell in all six cases. A statement-level trace found that none of the 12 statements extracted from these answers was supported by the restored context.

The baseline here is 0.917 rather than the 1.000 in Section 3.1 because one answer (false_02) scored 0.5 when rescored. The original evaluation run did not record its RAGAS package version and used a different evaluation path, so the source of this difference cannot be isolated. Once the counter-evidence is in its input, the evaluator registers the conflict.

### 3.6 Mitigation

| Scenario | Desired | `conservative_abstain` | `retain_if_no_opposition` |
| --- | --- | --- | --- |
| Wrong baseline, correction available | Answer | 6 correct | 6 correct |
| Wrong baseline, no correction | Abstain | 6 abstained | 6 wrong |
| Correct baseline, no opposition | Answer | 1 correct, 5 abstained | 6 correct |
| Correct baseline, stale opposition | Answer | 6 correct | 6 correct |
| Desired outcomes | | 19/24 | 18/24 |
| False answers | | 0 | 6 |

The gate admitted all 12 decisive passages it was shown (six critical passages and six superseded support passages), and the revised answer was correct in all 12 cases. In the stale-opposition scenario the generator sided with the audit correction over the earlier analysis. Of the 48 outcome-free passages inspected (neutral and replacement passages), the gate rejected 47. It judged one neutral documentation record for true_02 as FALSE and admitted it. The revised answer stayed correct, which is why the conservative policy answered one no-opposition case.

Neither policy does better on every measure. The conservative policy gave no false answers but abstained on five correct baselines. The retaining policy answered every case and repeated all six wrong baselines that had no correction available.

## 4. Discussion

A RAG system can produce a wrong answer that is faithful to its context, from a context that a judge rates sufficient. Faithfulness behaves as intended in these results: it measures support in the context it is given, and the restoration analysis shows it responds once the counter-evidence is included. The gap is upstream, in whether retrieval passed on the evidence needed to test the answer.

This suggests evaluating RAG at two levels:

1. Is the answer supported by the retrieved context?
2. Did the retrieved context include the evidence that could have overturned it?

This is an observation-boundary problem rather than a failure of faithfulness itself. A context-local evaluator cannot directly assess evidence that is outside its input. The AVAILABLE-BUT-NOT-EXPOSED condition isolates that boundary by holding the corpus fixed while changing only whether the answer-changing passage reaches the generator.

Signals computed from the context alone can address the first question but not the second. The second needs information beyond the final context—for example, a reference-derived label indicating whether corrective evidence was visible, as used in this controlled analysis, or an active search for opposing evidence, as in the mitigation experiment. That mitigation worked cleanly here partly because the counter-evidence in this benchmark is explicit and claim-specific.

ACE-A and ACE-E@k provide a compact description of this distinction within the controlled benchmark, but they rely on oracle knowledge of which passage is answer-changing. Reference-based retrieval-completeness metrics address a related problem by measuring whether expected evidence was retrieved; the narrower question here is whether evidence capable of overturning the answer was available but excluded from the final context.

## 5. Limitations

This is a proof-of-concept study with six synthetic claims. The controlled design isolates the mechanism but does not estimate how often it occurs in natural corpora or deployed RAG systems.

The generator produced binary verdicts rather than free-form answer prose, and RAGAS Faithfulness was evaluated on a fixed templated sentence constructed from that verdict. The findings therefore do not establish how the same effect would behave for unconstrained long-form generation.

The counter-evidence is explicit: each critical passage presents itself as an audit correction that supersedes the earlier result. Natural corpora rarely resolve conflicts this cleanly, so both the failure and the mitigation are easier to see here than they would be in practice.

The main NOT EXPOSED condition is an explicit intervention. The rank-cutoff experiment reaches the same outcome through ranking, but only at a boundary where the correction always ranked second.

The sufficiency result uses one judge model and one prompt, and faithfulness scores varied between runs for one answer. The mitigation gate uses the same model as the generator and inspects at most four candidates. Larger benchmarks, natural corpora, multiple retrieval stacks and multiple evaluators are needed before broader claims can be made. The sufficiency judge was explicitly instructed to assess only the supplied context, so its 18/18 SUFFICIENT result should not be interpreted as evidence that a context-only judge ought to detect omitted evidence.

ACE-A and ACE-E@k are oracle-labelled diagnostics rather than general-purpose retrieval metrics. Their computation relies on the benchmark's known critical passage. Applying the same idea to natural corpora would require identifying which evidence is genuinely capable of changing an answer without relying on experimental labels.

## AI assistance

ChatGPT (OpenAI) and Claude (Anthropic) were used for coding assistance, debugging, and manuscript editing. Study design, methodological decisions, validation, analysis, interpretation, and conclusions remained the responsibility of the author; all AI-assisted outputs were reviewed and verified.

## References

1. Lewis, P. et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** NeurIPS, 2020.

2. Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. **RAGAs: Automated Evaluation of Retrieval Augmented Generation.** EACL System Demonstrations, 2024.

3. Glockner, M., Hou, Y., & Gurevych, I. **Missing Counter-Evidence Renders NLP Fact-Checking Unrealistic for Misinformation.** EMNLP, 2022.

4. Joren, H. et al. **Sufficient Context: A New Lens on Retrieval Augmented Generation Systems.** ICLR, 2025.

5. Jain, S. & Vedam, V. N. **CUE-R: Beyond the Final Answer in Retrieval-Augmented Generation.** arXiv:2604.05467, 2026.

6. Deswal, S. S. **EviScope: Paired Counterfactual Evidence Diagnostics for Faithful and Efficient Grounded Language Models.** arXiv:2609.17081, 2026.

7. Shams, M. **LayerRAG-Bench: A Cross-Layer Reliability Benchmark for Agentic Retrieval-Augmented Generation.** arXiv:2607.27353, 2026.

8. Min, B., Edemacu, K., Cho, S.-H., Choi, Y., Jang, B., & Kim, J. W. **When Absence Is Evidence: Evaluating Completeness-Sensitive Negative Reasoning in Large Language Models.** arXiv:2608.04591, 2026.

9.  Ru, D. et al. **RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation.** arXiv:2408.08067, 2024.
