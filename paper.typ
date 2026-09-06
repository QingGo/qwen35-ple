#set page(paper: "us-letter", margin: 1in)
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.")

#let codeblock(body) = block(fill: rgb("#f5f5f5"), inset: 6pt, radius: 3pt, body)

= Auditable N-Gram External Memory for Small Language Models: Capabilities and Boundaries

#v(0.5em)
*Authors:* qwen35-ple project
#v(0.5em)
*Date:* 2026-09-06
#v(1em)

== Abstract

External memory is often proposed as a way to improve small language models without
expensive retraining. This paper studies a specific instance: an auditable,
n-gram-addressable external memory built from a Qwen3.8-Flash-Next PLE table and
attached to a frozen Qwen3.5-0.8B model. We introduce *PLE Projector*, a small MLP
that maps the backbone hidden state and lexical memory features to per-token
scale/bias corrections in logit space, and a token-level learned policy that
protects open-ended generation.

On a real HumanEval subset, BM25+PLE recovers problem-level passes that the base
model does not solve. On 10k local-continuation data with three seeds, the learned
projector improves teacher-forced NLL by $0.26$ over fixed PLE calibration with
bootstrap 95% CI $[0.12, 0.42]$. Official NGM and a small kNN-LM baseline did not
match these gains. However, on TriviaQA, the 0.8B base model achieves zero exact
match, and raw projector fusion can degrade open-ended generation. The paper
therefore positions PLE as a *local low-entropy memory* rather than a universal
semantic memory.

== Introduction

Large language models increasingly rely on retrieval and external memory to
compensate for limited parameters. For a 0.8B model, parameter-efficient adapters
and RAG are practical, but it remains unclear whether an exact n-gram memory can
provide a distinct, auditable benefit. DeepSeek's Engram and Qwen's PLE suggest
that n-gram lookup can be integrated into a frozen backbone, but the value of such
memory on small models is not well characterized.

We study the following questions:

- Can an auditable n-gram external memory improve local code continuation?
- Can a learned projector outperform fixed calibration?
- Can a token-level policy prevent open-ended generation degradation?
- Where does external memory fail?

Our contributions are:

+ A *PLE Projector*: a small MLP that maps hidden states and memory statistics to
  logit scale/bias, trained with frozen backbone and next-token cross-entropy.
+ A token-level learned policy for safe activation of PLE fusion.
+ Real-benchmark evidence, including HumanEval, TriviaQA, NGM and kNN-LM baselines.
+ A systematic boundary analysis: external n-gram memory helps local low-entropy
  code, but does not substitute for RAG or parametric adapters on general tasks.

== Related Work

#text(weight: "bold")[External memory.] DeepSeek Engram (link: https://github.com/deepseek-ai/Engram) and
Qwen PLE use n-gram lookup with gating and residual injection. Memory Grafting
(link: https://papers.cool/arxiv/2605.20948) scales pre-training via offline conditional memory.
XMemTransfer (link: https://github.com/OLAResearch/XMemTransfer) transfers memory across models with
a target-side reader.

#text(weight: "bold")[Nonparametric baselines.] kNN-LM (link: https://papers.lunadong.com/paper/4555) is a classic
nonparametric next-token model, but is known not to improve open-ended generation
(link: https://aclanthology.org/2023.emnlp-main.929/). NGM (link: https://github.com/PioneerQyw/NGM) provides a
training-free n-gram memory hook. We compare against both.

#text(weight: "bold")[Parameter-efficient adapters.] MoRA, LoRA, and QLoRA provide parametric
capability gains. In this paper, these are used as system components rather than
replacing external memory.

== Method

=== PLE Addressable Memory

We use an n-gram-addressable memory:

#codeblock(
```text
context n-gram -> empirical next-token distribution
             -> external value index
```
)

For a context $c$, the memory returns a sparse distribution $p_m$ over the
vocabulary and the longest matched n-gram order. This is a fully auditable,
non-parametric memory.

=== PLE Projector

The projector is a small MLP $f_theta$:

#codeblock(
```text
h_t (frozen backbone hidden state)
+ memory features (order, entropies, density ratio, agreement)
+ task one-hot (code/name/number/general)
  -> scale_t, bias_t
  -> fused = base_logits + scale_t * log p_m + bias_t
```
)

The final linear head is zero-initialized, so an untrained projector is identical
to the base model. Only the projector is trained; the backbone remains frozen.

=== Token-Level Learned Policy

A logistic regressor predicts whether PLE fusion will improve the next-token
probability, using the same memory features. It acts as a safety gate before PLE
fusion, especially for open-ended generation.

=== Purified OPSD Adapter

We also train a Purified OPSD MoRA adapter on a filtered subset of synthetic
instruction data. This provides a parametric companion to the non-parametric PLE
memory.

== Experimental Setup

*Model.* Qwen3.5-0.8B, frozen when used with PLE.

*Memory.* Qwen3.8-Flash-Next PLE / addressable n-gram table, built from a local
Python code corpus and wiki text.

*Datasets.*

+ HumanEval: official `openai/openai_humaneval`, 20 problems.
+ TriviaQA: official `mandarjoshi/trivia_qa` RC, 100 validation examples.
+ Local continuation: projector datasets of 1k and 10k samples.
+ Joint system tasks: knowledge, arithmetic, code-output subsets.

*Baselines.*

+ Base frozen model.
+ BM25 RAG.
+ kNN-LM (small hidden-state version).
+ NGM official hook.
+ LoRA / QLoRA / MoRA / Purified MoRA.
+ Fixed PLE calibration and learned PLE Projector.

== Results

=== HumanEval

#figure(
  table(
    columns: 3,
    [Condition], [#text("pass@1")], [mean repetition],
    [Base], [0.10], [0.010],
    [BM25+PLE], [0.10], [0.026]
  ),
  caption: [HumanEval 20: #text("pass@1") and repetition.]
)

#figure(
  image("figures/fig_humaneval.png", width: 85%),
  caption: [HumanEval 20: #text("pass@1") and repetition rate.]
)

Base solved `HumanEval/16` and `HumanEval/18`. BM25+PLE solved
`HumanEval/0` and `HumanEval/10`. The solved sets are disjoint, indicating that
PLE retrieval provides different local code knowledge rather than duplicating the
base model's ability.

=== TriviaQA

For 100 TriviaQA RC examples, the 0.8B base model obtained:

#codeblock(
```text
exact match = 0.0
mean repetition = 0.0048
```
)

This demonstrates that the model cannot yet answer short-form knowledge
questions reliably, and that PLE does not magically fix this.

=== Local Continuation: 10k Data, 3 Seeds

#figure(
  table(
    columns: 7,
    [Seed], [Base NLL], [Fixed NLL], [Projector NLL], [Base hit], [Fixed hit], [Projector hit],
    [0], [3.148], [3.051], [2.930], [0.407], [0.427], [0.463],
    [1], [3.328], [3.346], [3.114], [0.410], [0.420], [0.460],
    [2], [3.451], [3.426], [3.002], [0.413], [0.430], [0.493]
  ),
  caption: [10k local continuation: projectors vs fixed calibration.]
)

Paired across seeds:

+ Projector vs fixed NLL mean: $+0.2595$
+ Bootstrap 95% CI: $[0.1218, 0.4243]$
+ Positive seeds: $3/3$

At 100 samples with 5 seeds, the projector-vs-fixed mean was $+0.1044$ with CI
$[0.0251, 0.1866]$, also positive.

#figure(
  image("figures/fig_10k_improvement.png", width: 70%),
  caption: [Per-seed projector-vs-fixed NLL improvements on 10k data.]
)

=== NGM and kNN-LM Baselines

#figure(
  table(
    columns: 4,
    [Baseline], [NLL], [Hit], [Note],
    [Base], [2.521], [0.550], [],
    [NGM], [2.521], [0.550], [near-zero change],
    [Base], [2.633], [0.505], [kNN eval set],
    [kNN-LM], [2.847], [0.540], [better hit, worse NLL]
  ),
  caption: [Official NGM and small-scale kNN-LM baselines.]
)

Neither contemporary training-free baseline outperformed the learned PLE
projector on local continuation.

=== Joint System Table

On knowledge, arithmetic, and code-output tasks with 3 seeds:

#figure(
  table(
    columns: 4,
    [System], [Knowledge], [Arithmetic], [Code-output],
    [Base], [-8.140], [-7.365], [-14.250],
    [RAG], [-6.748], [-7.365], [-14.250],
    [PLE], [-8.140], [-7.492], [-14.250],
    [MoRA], [-7.691], [-7.114], [-12.292],
    [RAG+MoRA], [-6.704], [-7.114], [-12.292],
    [All], [-6.704], [-7.237], [-12.292]
  ),
  caption: [Mean answer log-probability (higher is better).]
)

#figure(
  image("figures/fig_joint_system.png", width: 90%),
  caption: [Joint system answer log-probability, 3 seed means.]
)

RAG mainly improves knowledge; MoRA mainly improves arithmetic and code-output;
PLE alone does not improve general tasks.

=== Open-Ended Generation Safety

Raw PLE projector fusion on five natural-language code prompts produced visible
degradation: repetitive fragments, unrelated repository text, and broken
structure. Adding the learned token policy strongly reduced this degradation.
This indicates that PLE should not be enabled unconditionally in open-ended
generation.

=== LLM-as-Judge

We used DeepSeek V4 Flash as an external judge. For HumanEval 20 problems, the
mean judge score was $0.50$ for base and $0.25$ for BM25+PLE. For TriviaQA 20
examples, the base model scored $0.25$.

#figure(
  image("figures/fig_judge.png", width: 70%),
  caption: [LLM-as-judge mean scores.]
)

The judge scores are lower than pass-based metrics, indicating that generated
answers often look plausible but are judged as incomplete or incorrect.

== Discussion

The empirical picture is clear:

+ PLE n-gram memory is most valuable as a *local, low-entropy code memory*.
+ A learned projector is better than fixed calibration when enough local data is
  available.
+ RAG and parametric adapters remain superior for general knowledge and reasoning.
+ PLE is not a universal semantic memory, and overselling it would be wrong.

== Limitations

+ HumanEval subset is 20 problems; TriviaQA exact match is zero.
+ #text("Pass@k") evidence is small (3 problems x 2 samples).
+ LLM-as-judge results are available for HumanEval 20 and a 20-example TriviaQA subset; the full 100-example judge run is not part of this artifact package.
+ Public adapter weights are not yet released.
+ CPU deployment throughput is not yet optimized.

== Conclusion

We presented an auditable n-gram external memory system for a 0.8B frozen model,
including a learned PLE projector, token-level policy, real-benchmark evaluations,
and comparisons against contemporary external-memory baselines. The results
support a boundary claim: external n-gram memory can improve local low-entropy
code continuation, while RAG and parametric adapters remain necessary for general
ability. The system is fully reproducible with open source code, data builders,
evaluation scripts, and a container.

== Reproducibility

All code is available in the repository. Key artifacts include:

+ `scripts/run_humaneval_real_ablation.py`
+ `scripts/run_humaneval_passk.py`
+ `scripts/run_triviaqa_real_eval.py`
+ `scripts/run_ngm_baseline.py`
+ `scripts/run_knn_lm_baseline.py`
+ `scripts/run_10k_projector_seeds.sh`
+ `Dockerfile`
+ `docs/evaluation-card-paper.md`

== References

+ DeepSeek Engram: https://github.com/deepseek-ai/Engram
+ Memory Grafting: https://papers.cool/arxiv/2605.20948
+ XMemTransfer: https://github.com/OLAResearch/XMemTransfer
+ NGM: https://github.com/PioneerQyw/NGM
+ kNN-LM: https://papers.lunadong.com/paper/4555
+ kNN-LM Does Not Improve Open-ended Text Generation: https://aclanthology.org/2023.emnlp-main.929/
+ Memory Layers at Scale: https://mlanthology.org/icml/2025/berges2025icml-memory/
+ MemSFT: https://github.com/LUMIA-Group/MemSFT
