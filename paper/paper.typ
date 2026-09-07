#import "sty/icml2024.typ": icml2024

#show: icml2024.with(
  title: [Auditable N-Gram External Memory for Small Language Models: A Low-Resource Study of Capabilities, Calibration, and Boundaries],
  authors: (
    ((name: "QingGo", affl: "affl1", email: "qinggo@example.com"),),
    (affl1: ("Independent Researcher",),)
  ),
  abstract: [
    Small language models face a fundamental capacity bottleneck: they must either compress knowledge into parameters or retrieve it at inference time. Retrieval-augmented generation (RAG) is the dominant solution, but its behavior is opaque and its retrieved evidence is not fully auditable @rag2020. In this paper, we study a complementary mechanism: an #emph[auditable n-gram external memory] derived from the PLE / Engram line of sparse external-memory systems @engram2026. We attach this memory to a frozen Qwen3.5-0.8B model and introduce a small learned #emph[PLE Projector] that maps the backbone hidden state plus lexical memory features into per-token logit scale and bias corrections. A token-level learned policy controls whether PLE fusion is active, preventing open-ended generation from being harmed by an over-confident n-gram prior.

    We evaluate the system across local-continuation tasks, real HumanEval @humaneval2021, real TriviaQA @triviaqa2017, and a joint system with RAG and parameter-efficient adapters @lora2022 @qlora2023 @mora2024. On 10k local-continuation data with three seeds, the learned projector improves teacher-forced NLL by $0.26$ over fixed PLE calibration with bootstrap 95% CI $[0.12, 0.42]$. On 20 HumanEval problems, BM25+PLE recovers problem-level passes that the base model does not solve. However, on TriviaQA the 0.8B base model obtains zero exact match, and raw PLE fusion can degrade open-ended generation @knnopen2023. Our conclusion is deliberately a #emph[boundary] result: n-gram external memory is valuable as a local low-entropy code memory, but it is not a substitute for RAG or parametric adapters on general tasks.
  ],
  bibliography: bibliography("refs.bib"),
  accepted: none,
)

= Introduction

Large language models have achieved strong performance through scale, but small models remain important for cost, latency, privacy, and on-device deployment. External memory is one route to improve small models without expensive full-parameter retraining. Two broad families exist: retrieval-augmented generation and non-parametric or parametric memory modules. RAG is effective but has limitations: it retrieves documents, not token-level continuations; it can be noisy; and its evidence is not always easy to audit @rag2020. Non-parametric memory, such as kNN-LM @knnlm2020 and n-gram memory, offers a different granularity: it can directly influence the next-token distribution.

The recent DeepSeek Engram and Qwen PLE systems show that n-gram lookup can be integrated into transformer backbones with sparse, disk-backed tables @engram2026. Memory Grafting scales pre-training using an offline conditional memory table @memorygrafting2026, while XMemTransfer shows that a target-side reader can adapt a memory table across model families @xmemtransfer2026. However, most evaluations of such systems have focused on large models. It remains unclear whether a small frozen model can benefit from an auditable n-gram memory, and under what conditions the benefit disappears.

This paper addresses the following research questions:

- Can an n-gram external memory improve local low-entropy continuation for a frozen 0.8B model?
- Can a small learned projector outperform fixed calibration @logopinion1986?
- Can a token-level policy make PLE safe for open-ended generation @knnopen2023?
- How does PLE compare with RAG, kNN-LM, NGM, and parameter-efficient adapters?
- What are the precise boundaries of PLE usefulness?

Our contributions are as follows.

+ We introduce the #emph[PLE Projector], a small MLP that maps hidden states, memory statistics, and task identity to per-token scale/bias in logit space. The projector is trained with a frozen backbone using next-token cross-entropy and is zero-initialized to be inert at the start of training.
+ We introduce a token-level learned policy that acts as a safety gate, learned from per-token observations rather than hand-written rules @bitterlesson2019.
+ We provide a systematic comparison on real benchmarks, including HumanEval and TriviaQA, as well as training-free baselines kNN-LM and official NGM @ngm2026.
+ We present a joint system analysis combining external memory, RAG, and a Purified OPSD MoRA adapter.
+ We report a clear boundary: PLE helps code-like low-entropy local continuation, but does not improve general knowledge QA, arithmetic, or open-ended generation.

= Related Work

== External Memory Layers

DeepSeek Engram @engram2026 and Qwen PLE implement conditional memory via scalable n-gram lookup. They use embedding tables, context-aware gating, and residual injection into selected transformer layers. Memory Grafting @memorygrafting2026 scales pre-training using an offline conditional memory table, while XMemTransfer @xmemtransfer2026 shows that a target-side reader can adapt a memory table across model families. Memory Layers at Scale @memorylayers2025 demonstrates that memory layers can be integrated into large models without full retraining.

== Non-Parametric Language Models

kNN-LM @knnlm2020 is a classic non-parametric model that interpolates a base LM with a nearest-neighbor distribution over a datastore. Subsequent work showed that kNN-LM does not improve open-ended generation @knnopen2023. NGM @ngm2026 provides a training-free n-gram memory hook. Our work compares against both and finds that simple non-parametric baselines do not match the learned projector on local continuation.

== Distribution-Level Memory

MemSFT @memsft2026 and TokenMem @tokenmem2026 propose external parametric memory channels that operate at the distribution or hidden-state level, with learned routers to avoid alignment tax. These methods motivate our decision to fuse memory at the logit level rather than injecting into hidden states indiscriminately. Earlier experiments in our project found that hidden-state injection without careful orthogonalization and gating can produce large real-vs-control gaps that are not attributable to the memory content.

== Parameter-Efficient Adapters

LoRA @lora2022, QLoRA @qlora2023, and MoRA @mora2024 provide compact parametric updates. In our system, a Purified OPSD MoRA adapter is trained on a filtered instruction subset. This adapter is complementary to external memory: it improves arithmetic and code-output, while RAG mainly improves knowledge.

= Background and Theory

== N-Gram Addressable Memory

Let a context be a token sequence $c = (c_1, ..., c_t)$. We maintain sparse counts for each n-gram of order $n$:

$$ p_m(y | c) = count(c, y) / sum_y count(c, y). $$

The memory also returns the longest matched order and an external value index that can be audited. This memory is non-parametric, transparent, and can be rebuilt from any corpus.

== Real versus Control

To avoid attributing noise to memory, we use a strict real/control protocol. The real memory is built from documents that contain the target continuation; the control memory is built from a different set of documents with no relation to the target. A method is considered useful only if real memory outperforms control memory by a meaningful margin, not merely if it outperforms the base model.

== Optimal Logit Correction

From a decision-theoretic perspective, the optimal way to combine a base model and a memory distribution is not to replace the base model, but to add a calibrated log-ratio correction @logopinion1986:

$$ log p_fused(y) = log p_b(y) + lambda_t log p_m(y) + beta_t. $$

This is a log-opinion-pool formulation. Without calibration, an unweighted n-gram prior is often over-confident and degrades generation @knnopen2023. The PLE Projector learns $lambda_t$ and $beta_t$ from hidden states and memory statistics.

== Why Hidden-State Alignment Is Insufficient

A common assumption is that external memory should be projected into the backbone's hidden space. Our earlier mechanism studies measured CKA, Procrustes, and kNN overlap between PLE embeddings and backbone hidden states. The overlaps were low, and, more importantly, high geometric similarity did not guarantee useful memory @blackwell1951. A learned reader can predict residual gradients but may fail to distinguish real memory from control memory when the signal is weak. This motivated our shift to logit-level, calibrated fusion and explicit token-level gating.

= Method

== PLE Memory and Retrieval

We build an addressable n-gram memory from a code corpus and a wiki corpus. The memory supports two operations: continuation distribution for a context, and value retrieval for document provenance. The memory is used as both a retrieval channel and a logit prior. In the hybrid system, BM25 @bm252009 provides document-level retrieval, PLE provides exact n-gram continuity, and their combination forms a three-channel retriever.

== PLE Projector

The projector is a small MLP:

$$ f_theta(h_t, m_t) = (alpha_t, beta_t). $$

where $h_t$ is the last hidden state of the frozen backbone, and $m_t$ contains matched n-gram order, base entropy, memory entropy, density ratio, base top-1 probability, memory top-1 probability, memory/base agreement, and task one-hot. The output $alpha_t$ scales $log p_m$ and $beta_t$ adds a support bias. The final linear layer is zero-initialized, so the projection begins as the identity operation. Only the projector parameters are trained.

== Token-Level Policy

Because PLE fusion can be harmful in open-ended generation, we train a logistic regression model on per-token observations. The features are the same memory features used by the projector. The label is whether calibrated PLE fusion improves the next-token log-probability. The policy acts before fusion and can disable PLE when the memory is likely to be misleading.

== Calibration and Router

We use per-task calibrated scale/bias/temperature, a density-ratio gate based on $E_{p_m}[log(p_m/p_b)]$, a token-level learned policy, and the learned PLE projector when hidden states are available. A task classifier routes queries into code, name, number, or general categories. Open-ended generation uses the token policy to prevent unsafe fusion.

== Purified OPSD Adapter

To test whether PLE complements parametric learning, we train a Purified OPSD MoRA adapter on a filtered subset of synthetic instruction data @mora2024. This provides a parameterized companion to the non-parametric memory.

= Experimental Setup

== Model

All experiments use Qwen3.5-0.8B as the frozen backbone. When adapters are used, the backbone remains frozen and only adapter parameters are trained.

== Data

+ Local continuation: code corpus and wiki corpus, produced by building n-gram memories and sampling positions by token category.
+ Dataset sizes: 100, 1k, and 10k samples.
+ HumanEval: official @humaneval2021, first 20 problems.
+ TriviaQA: official @triviaqa2017, 100 examples.
+ Joint system tasks: knowledge, arithmetic, and code-output subsets.

== Baselines

+ Frozen base model.
+ BM25 RAG @bm252009.
+ kNN-LM, small hidden-state version @knnlm2020.
+ Official NGM hook @ngm2026.
+ Fixed calibrated PLE fusion.
+ Learned PLE Projector.
+ LoRA / QLoRA / MoRA / Purified MoRA @lora2022 @qlora2023 @mora2024.
+ Joint combinations: RAG+PLE, PLE+MoRA, RAG+PLE+MoRA.

== Metrics

+ Teacher-forced NLL and #text("hit@1").
+ Real vs control.
+ HumanEval #text("pass@1") and repetition.
+ TriviaQA exact match.
+ LLM-as-judge scores @llmjudge2024.
+ Paired bootstrap 95% CI across seeds.

== Statistical Protocol

We use 5 seeds for the 100-sample projector experiment and 3 seeds for the 10k experiment. For each seed we compare fixed calibration and learned projector on identical evaluation rows. Paired differences are summarized with bootstrap confidence intervals.

= Results

== Local Continuation

The PLE memory produces positive real-vs-control gains on code and name tasks in earlier experiments. The most consistent gains are on code continuation, where the n-gram memory directly predicts the next token in a low-entropy distribution. Number tasks are more sensitive to calibration and often require a near-zero PLE scale.

== Learned Projector Scaling

At 100 samples with 5 seeds, the projector-vs-fixed NLL mean is $+0.1044$ with bootstrap 95% CI $[0.0251, 0.1866]$. At 10k samples with 3 seeds, the improvement grows to $+0.2595$ with CI $[0.1218, 0.4243]$.

#figure(
  table(
    columns: 7,
    [Seed], [Base NLL], [Fixed NLL], [Proj. NLL], [Base hit], [Fixed hit], [Proj. hit],
    [0], [3.148], [3.051], [2.930], [0.407], [0.427], [0.463],
    [1], [3.328], [3.346], [3.114], [0.410], [0.420], [0.460],
    [2], [3.451], [3.426], [3.002], [0.413], [0.430], [0.493]
  ),
  caption: [10k local-continuation results. NLL is lower-is-better.]
)

#figure(
  image("figures/fig_10k_improvement.png", width: 100%),
  caption: [Per-seed projector-vs-fixed improvements on 10k data.]
)

Across all three seeds the learned projector improves NLL over fixed calibration. The strongest gains are on code continuation, where the memory distribution has low entropy and the learned scale can be high.

== HumanEval

On 20 official HumanEval problems, base and BM25+PLE both achieve $0.10$ #text("pass@1"), but the passed problems are completely disjoint.

#figure(
  table(
    columns: 3,
    [Condition], [#text("pass@1")], [mean repetition],
    [Base], [0.10], [0.010],
    [BM25+PLE], [0.10], [0.026]
  ),
  caption: [HumanEval 20 results.]
)

#figure(
  image("figures/fig_humaneval.png", width: 100%),
  caption: [HumanEval #text("pass@1") and repetition.]
)

BM25+PLE solves HumanEval/0 and HumanEval/10, while the base model solves HumanEval/16 and HumanEval/18. This shows that PLE provides a different source of local code knowledge: it does not simply amplify the base model's existing solution distribution, but can recover different problems.

== TriviaQA

On 100 TriviaQA RC examples, the 0.8B base model obtains exact match $0.0$ and mean repetition $0.0048$. This is a truthful baseline: the model fails short-form knowledge QA @triviaqa2017. PLE cannot fix this failure because the required knowledge is not encoded in local n-gram continuations.

== Training-Free Baselines

We compare against kNN-LM and official NGM.

#figure(
  table(
    columns: 4,
    [Baseline], [NLL], [Hit], [Note],
    [Base], [2.633], [0.505], [kNN eval set],
    [kNN-LM], [2.847], [0.540], [better hit, worse NLL],
    [Base], [2.521], [0.550], [NGM eval set],
    [NGM], [2.521], [0.550], [near-zero change]
  ),
  caption: [kNN-LM and NGM baselines.]
)

Both baselines fail to match the learned projector. NGM has almost no effect, while kNN-LM improves hit but worsens NLL @knnlm2020. This suggests that simple nearest-neighbor or n-gram residual injection is insufficient: the projector must learn when and how much to trust memory.

== Joint System

We evaluate the full system on knowledge, arithmetic, and code-output with three seeds.

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
  caption: [Mean answer log-probability, three seeds. Higher is better.]
)

#figure(
  image("figures/fig_joint_system.png", width: 100%),
  caption: [Joint system answer log-probability.]
)

RAG is the main contributor to knowledge. MoRA is the main contributor to arithmetic and code-output. PLE alone does not provide general ability gains.

== Open-Ended Generation and Safety

Raw PLE projector fusion on natural-language code prompts produces visible degradation: repetitive fragments, unrelated repository text, and broken structure @knnopen2023. Adding the learned token policy strongly reduces this degradation. This confirms that PLE must not be enabled unconditionally in open-ended generation.

== LLM-as-Judge

We use DeepSeek V4 Flash as an external judge @llmjudge2024. For HumanEval 20, the mean score is $0.50$ for base and $0.25$ for BM25+PLE. For a 20-example TriviaQA subset, base scores $0.25$.

#figure(
  image("figures/fig_judge.png", width: 100%),
  caption: [LLM-as-judge mean scores.]
)

The judge scores are lower than pass-based metrics. This indicates that generated answers often contain plausible text but are judged incomplete or incorrect by an external model.

= Analysis

== When Does PLE Help?

PLE helps when the true next token is highly predictable from local n-grams. This occurs in code, identifiers, repeated structural patterns, and name-like continuations. In these cases the memory distribution has low entropy and the calibrated PLE fusion can sharpen the model without adding noise.

== When Does PLE Fail?

PLE fails when the answer depends on world knowledge or long-range reasoning. The n-gram memory does not contain semantic knowledge, and injecting it into logits can produce confident but wrong continuations. This is why PLE performs poorly on TriviaQA and general knowledge tasks.

== Why a Learned Projector Matters

Fixed calibration selects one scale and bias for all tokens. A learned projector can vary the trust in PLE by context. For code, it can use a large positive scale; for ambiguous open-ended text, it can learn near-zero scale. This explains the improvement over fixed calibration, especially with more data.

== Boundary with RAG and Adapters

Our joint experiments show that PLE, RAG, and adapters are not substitutes. RAG supplies document-level evidence, PLE supplies token-level continuity, and adapters supply parametric capability @rag2020 @lora2022. The combination is useful, but PLE is only one small component.

== Historical Negative Results

Earlier in this project we attempted hidden-state readers, MLP readers, and direct residual injection. These approaches often improved loss-like metrics but failed the real-vs-control test @blackwell1951. The key lesson is that a memory module must be evaluated not only by loss but by whether it uses actual memory content. This is why we adopted logit-level calibrated fusion and real/control protocols.

= Limitations

+ HumanEval subset is only 20 problems, and TriviaQA exact match is zero.
+ #text("pass@k") evidence is a small 3-problem, 2-sample experiment.
+ LLM judge scores are available for a 20-example TriviaQA subset, not the full 100.
+ Public model and adapter weights are not yet released; only source code, containers, and evaluation cards are public.
+ CPU deployment throughput is not yet optimized.

= Conclusion

We presented an auditable n-gram external memory system for a 0.8B frozen model. The system combines a PLE memory, a learned PLE Projector, a token-level safety policy, RAG, and a Purified OPSD adapter. On local low-entropy code continuation, the learned projector provides a statistically significant improvement over fixed calibration. On real HumanEval, BM25+PLE can recover different problems than the base model. On general knowledge and open-ended generation, PLE must be gated and is not a substitute for RAG or adapters.

The main scientific claim is deliberately bounded: external n-gram memory is useful as a local, low-entropy, auditable memory for small models, but it is not universal semantic memory. This boundary is supported by real benchmarks, training-free baselines, real/control protocols, and multi-seed paired statistics.

= Appendix: Project Development Timeline

This work is the result of an extended low-resource research program. The following phases reflect the main historical threads integrated into this paper.

== Phase 0: Mechanism Diagnostics

We studied whether PLE embeddings align with backbone hidden states using CKA, Procrustes alignment, kNN overlap, and intrinsic dimension. The measured overlaps were low and close to random. More importantly, even when a learned reader could predict residual gradients, it often failed to distinguish real memory from control memory.

== Phase A: Reader Architectures

We implemented RMSNorm, ShortConv, EngramReader, QwenEngramReader, and MLPValueReader. Experiments showed that a purely linear value path under-uses nonlinear memory information, while an MLP value path can increase residual R2 substantially. However, real-vs-control differences remained tiny.

== Phase B: Distribution-Level Fusion

We shifted to n-gram addressable memory and logit-level fusion. We implemented calibration, per-task scale/bias/temperature, support-set calibration, density ratio gates, and log-opinion-pool fusion @logopinion1986. This formed the foundation of the current PLE Projector.

== Phase C: Purified OPSD and Adapters

We trained LoRA, QLoRA, and MoRA adapters, and introduced Purified OPSD to filter noisy synthetic instruction data @lora2022 @qlora2023 @mora2024. Purified MoRA improved held-out local tasks, while formal-style benchmarks remained less stable.

== Phase D: Learned Projector and Token Policy

We introduced a task-conditioned PLE Projector and a token-level learned policy. The projector maps hidden states and memory features to per-token scale and bias. The policy prevents unsafe PLE fusion during open-ended generation.

== Phase E: Real-Benchmark and Boundary Evaluation

We added official HumanEval @humaneval2021, TriviaQA @triviaqa2017, kNN-LM @knnlm2020, NGM @ngm2026, LLM-as-judge @llmjudge2024, and multi-seed paired statistics. These experiments produced the boundary result that PLE is a local low-entropy memory, not a universal semantic memory.
