#import "sty/icml2024.typ": icml2024

#show: icml2024.with(
  title: [What a Token-Keyed Residual Memory Can Carry],
  authors: (
    ((name: "QingGo", affl: "affl1", email: "zyqingjohn@qq.com"),),
    (affl1: ("Independent Researcher",),)
  ),
  abstract: [
    Sparse n-gram memory layers --- DeepSeek Engram, Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash --- inject a retrieved vector into the residual stream at every token, and are being scaled on the premise that the memory supplies knowledge the parameters do not hold. We show that this channel is bounded before any reader is chosen. The row identifier is a deterministic function of a short addressing window, so the data-processing inequality gives $I("future" ; e_t | h_t) <= I("future" ; w_t | h_t)$: the memory can supply only what the window already determines and the hidden state has not retained. Depth, width, linearity and training budget do not appear. In the Engram and Qwen3.8 designs the window is not the two tokens an order-3 key suggests but twelve, because a kernel-4 dilation-3 convolution follows the gate; in V4.1-Flash the convolution was removed and the window is exactly four.

    We test the prediction this makes about content. Three readers differing only in the corpus feeding the table --- Wikipedia, code, STEM --- are grafted into the same frozen backbones under identical prompts. They write measurably different vectors at the answer position (pairwise cosine 0.38 to 0.49 on 4B), and under the standard recipe they produce identical output distributions: total variation exactly $0.0000$ over 1500 items on 0.8B, and on 4B a largest spread of one item out of 600, below the split-half floor of the arm that produced it. Removing the answer-format SFT that saturates every arm at 2.5 generated tokens changes the picture, and the change is the bound's other half: all four format descriptors then move outside their split-half floors and both pre-registered directional lexical predictions hold (prose markers $+0.69$, $p = 4.5 times 10^(-34)$). Suppressing the chat scaffold is content-independent; the rest of the distribution carries the corpus's surface statistics, which is exactly the trigram prior the bound permits. What never appears, ceiling or no ceiling, is passage-conditioned recall.

    Two measurements fix the size of what remains, and they point in different directions. A probe of the frozen table recovers about 59% of the trigram top-1 that explicit counts reach on the same corpus, and more training makes it worse, so the loss belongs to the read-out rather than to undertraining. Measured directly, the vector the reader injects has an effective dimensionality of $1.001$ over 600 prompts --- $99.93\%$ of its variance in one direction, against $1.480$ for the hidden state it is gated on --- so the read-out is close to a constant function of its input. The design therefore fails for the bound's reason and for a read-out defect at the same time, and only the first is unfixable at a twelve-token window. The obvious repair does not pay either: at matched storage a 4-gram address ties or beats 8-token, 16-token and longest-suffix memories at every budget and on every corpus.

    The conclusion is a boundary rather than a result. Auditable n-gram memory is a real channel for the surface prior of a corpus and is not a channel for passage-conditioned knowledge; which of the two it is, is set by the addressing window and not by the reader, and that holds across the family.
  ],
  bibliography: bibliography("refs.bib"),
  accepted: none,
)

#set enum(indent: 1em, spacing: 0.55em)
#set list(indent: 1em, spacing: 0.55em)

= Introduction

Sparse n-gram memory is one of the few ways to add capacity to a language model without adding parameters. DeepSeek Engram, Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash all hash a short token window to a row of a large disk-backed table, project that row, and inject the result into the residual stream @engram2026. Memory Grafting scales pre-training with an offline conditional memory table @memorygrafting2026, and XMemTransfer adapts a table across model families with a target-side reader @xmemtransfer2026. The premise is that the table holds associations the weights compress away, and that a large enough table behind a good enough reader will supply them.

This paper asks what such a channel can carry, and answers it in a way that does not depend on the reader. The argument is short. The row identifier is a deterministic function of a short addressing window $w_t$; the reader's contribution is a deterministic function of that row and the hidden state $h_t$. The data-processing inequality then bounds the memory's contribution to the future by the window's: $I("future" ; e_t | h_t) <= I("future" ; w_t | h_t)$. Reader depth, width, linearity and training budget do not appear because they cannot. The bound is elementary; what is not elementary is that it was not written down for a family that three production systems are scaling, and that the window it depends on is not the one the headline key order suggests.

That last point is worth stating precisely, because it is where the bound becomes useful rather than obvious. An order-3 key reads two tokens, but the Engram and Qwen3.8 readers follow the gate with a depthwise convolution of kernel 4 and dilation 3, so the contribution at a position draws on memory rows up to nine positions back, each of which reads two tokens further: the true window is twelve tokens, not two. DeepSeek-V4.1-Flash removed that convolution, so its window is exactly four. The family spans a factor of three in the one quantity that bounds it, and that quantity is visible only by reading the convolution parameters.

The bound makes a sharp and testable prediction about content, and we test it on two backbones. Train three readers that differ only in the corpus feeding the table --- Wikipedia, code and STEM text --- and graft each into the same frozen model at the same layer under the same prompts. If the memory carried corpus content, the three should answer differently. Under the standard recipe they do not: on 0.8B over 1500 items per arm the answer-label distribution, the leading-surface distribution and their joint differ by a total-variation distance of exactly $0.0000$, and on 4B the largest cross-corpus spread is a single item out of 600. This is not a failed manipulation. The three readers write visibly different vectors at the answer position, and each is best on its own corpus in a cross-evaluation; the corpus changes the part of the trigram statistic the weights already encode, and under this recipe that part leaves no trace.

It would be a mistake to stop there, and we did not. Every arm in that experiment collapses to a two-token answer, so "no format difference" could mean "no room for one". Removing the answer-format supervision that causes the collapse changes the conclusion, and the change is the bound's other half rather than a counterexample to it: all four format descriptors move outside their split-half floors and both pre-registered directional lexical predictions hold, with the Wikipedia reader emitting far more prose markers than the code reader. The layering is exactly what the bound predicts. Suppression of the chat scaffold is content-independent --- all three readers suppress it and differ from one another by at most $0.01$ --- while the rest of the distribution carries the corpus's surface statistics. That dependence *is* the trigram prior the channel is permitted to carry. What never appears, with or without the ceiling, is passage-conditioned recall; the knowledge probes that require the passage return zero.

Our contributions are these.

+ We state and prove a bound on what a token-keyed residual memory can carry, in a conditional form that accounts for the gate's dependence on the hidden state, and we identify the addressing window of each production design in the family --- twelve tokens for Engram and Qwen3.8-Flash-Next, four for V4.1-Flash.
+ We test its content prediction directly on two frozen backbones with three corpus-matched readers, and show that the three inject measurably different vectors while producing identical output distributions.
+ We show that the null is scoped rather than absolute: removing the answer-length ceiling reveals the corpus's surface prior, exactly where the bound says it should be, while passage-conditioned recall remains at zero.
+ We measure the ceiling from the other side, probing the frozen table directly and finding that the read-out recovers about 59% of the trigram top-1 explicit counts reach, a loss that does not improve with more training.
+ We falsify the obvious repair. At matched storage, longer keys do not pay on any corpus or budget tested, so the short window is not the constraint that binds.
+ We report the audit that made these results checkable, including ten occasions on which a claim stood because the check that would have falsified it did not exist or silently skipped.

= Related Work

== External Memory Layers

DeepSeek Engram @engram2026 and Qwen PLE implement conditional memory via scalable n-gram lookup. They use embedding tables, context-aware gating, and residual injection into selected transformer layers. Memory Grafting @memorygrafting2026 scales pre-training using an offline conditional memory table, while XMemTransfer @xmemtransfer2026 shows that a target-side reader can adapt a memory table across model families. Memory Layers at Scale @memorylayers2025 demonstrates that memory layers can be integrated into large models without full retraining. Product-key memory layers @productkeys2019 provide a related sparse lookup mechanism, and recent latent n-gram architectures such as Lngram v2 @lngramv2 @lngram2026 and Tensorizing Engram @tensorizingengram improve the parameter efficiency and interpretability of discrete memory addressing. In the small-model regime, PEMA @pema2023, memory-augmented training @trainmem2022, Knowledge-in-Context @knowincontext2022, and plug-and-play knowledge injection @pipknowledge2023 show that external memory can be adapted after pretraining without full retraining.

== Long-Term and Agent Memory

A parallel line of work treats memory as a first-class agent resource. MemGPT @memgpt2023 introduces operating-system-style memory paging for LLM agents; HippoRAG @hipporag2024 and From RAG to Memory @fromragtomemory2025 convert retrieval corpora into long-term memory structures; Zep @zep2025, Memori @memori2026, and MemOrb @memorb2025 provide persistent or plug-and-play memory layers for conversational and agentic settings. These systems target long-range semantic memory, whereas our work focuses on a narrower, auditable token-level n-gram memory that is useful in low-entropy local continuations.

== Non-Parametric Language Models

kNN-LM @knnlm2020 is a classic non-parametric model that interpolates a base LM with a nearest-neighbor distribution over a datastore. Efficient kNN-LM @efficientknn2021 reduces the inference cost of the datastore, and subsequent analysis asks why nearest-neighbor language models work @whyknn2023. Later work showed that kNN-LM does not improve open-ended generation @knnopen2023. NGM @ngm2026 provides a training-free n-gram memory hook. Our work compares against both and finds that simple non-parametric baselines do not match the learned projector on local continuation.

== Distribution-Level Memory

MemSFT @memsft2026 and TokenMem @tokenmem2026 propose external parametric memory channels that operate at the distribution or hidden-state level, with learned routers to avoid alignment tax. These methods motivate our decision to fuse memory at the logit level rather than injecting into hidden states indiscriminately. Related work on memory-augmented language models also highlights the difficulty of distinguishing genuine memory use from shallow parametric recall @dismemreason2024. Long-context evaluations show that models often fail to use information placed in the middle of long contexts @lostmiddle2023, and memory-based models can still struggle on reasoning-in-a-haystack tasks @haystackmem2025 @needlehaystack2024 @attribution2026. LongBench @longbench2024 provides a broad bilingual long-context suite, while MemTrapBench @memtrap2026 probes cognitive traps in memory-heavy settings. Earlier experiments in our project found that hidden-state injection without careful orthogonalization and gating can produce large real-vs-control gaps that are not attributable to the memory content.

== Retrieval Reliability and Calibration

Adaptive retrieval methods such as Self-RAG @selfrag2023 and FAIR-RAG @fairrag2025 decide when retrieval is necessary; reliability-aware RAG systems further estimate whether retrieved evidence should be trusted @ragreliability2024 @era2026 @reliablerag2026 @trustmargin2026, and RAGRouter-Bench @ragrouter2026 benchmarks learned query-routing policies. Calibration research has shown that language models are often overconfident, which motivates token-level safety gates and uncertainty-aware fusion @calibsample2024 @confidencefaith2026. On code tasks, localized calibrated uncertainty helps identify unreliable predictions @codecalib2025. On the code side, benchmarks such as HumanEval Pro @humanevalpro2024 extend classic #text("pass@k") evaluation to more challenging self-invoking settings.

== Parameter-Efficient Adapters

LoRA @lora2022, QLoRA @qlora2023, and MoRA @mora2024 provide compact parametric updates. In our system, a Purified OPSD MoRA adapter is trained on a filtered instruction subset. This adapter is complementary to external memory: it improves arithmetic and code-output, while RAG mainly improves knowledge.

= What a Token-Keyed Residual Memory Can Carry

== N-Gram Addressable Memory

Let a context be a token sequence $c = (c_1, ..., c_t)$. We maintain sparse counts for each n-gram of order $n$:

$ p_m(y | c) = "count"(c, y) / sum_y "count"(c, y). $

The memory also returns the longest matched order and an external value index that can be audited. This memory is non-parametric, transparent, and can be rebuilt from any corpus.

== The Bound

The Engram / PLE line injects a retrieved vector into the residual stream. It is worth stating precisely what such a channel can carry, because the answer does not depend on the reader.

Let $w_t$ denote the addressing window ending at position $t$, and let the row identifier be a deterministic function of it, $a_t = A(w_t)$, so that the retrieved memory is $e_t = E(a_t)$. The contribution to the residual stream is $c_t = F(h_t, e_{t-r..t})$, where the backbone hidden state $h_t$ enters through the gate as the query and $r$ is the reach of the short causal convolution that follows the gate. In the Engram and Qwen3.8 designs the row id reads the last $n - 1$ tokens and the convolution has kernel 4 with dilation 3, so the convolution at a position reads memory rows up to nine positions back, each row itself reading two tokens further, and the window reaches eleven tokens back --- twelve tokens in all, not the two that $n = 3$ alone would suggest. Because $e_t$ is a deterministic function of the window, the data-processing inequality gives

$ I("future" ; e_t | h_t) <= I("future" ; w_t | h_t). $

Three consequences follow.

First, the bound is independent of the reader. Depth, width, linearity and training budget do not appear in it, so no reader, however expressive, can exceed it. This turns our earlier failures with hidden-state readers and direct residual injection from an empirical observation into a necessary one.

Second, it constrains addressing rather than capacity. The memory can only supply what the window already determines and the hidden state has not retained, so no context-conditioned recall is possible when the determining information lies outside the window. For a question answered from a passage, the window at the answer position carries the prompt's format, not its content; the memory may still sharpen *how* the answer is emitted, which is the format effect we measure, but not *which* answer is correct.

Third, it does not say the memory is useless. Backbone weights are a lossy compression of the training corpus, and rare n-gram statistics are among what is compressed away. The bound leaves room for precisely that, and it predicts where: gains should concentrate on rare n-grams and on continuations the window determines, and should vanish elsewhere. Our measurements agree --- knowledge probes that require the passage return zero, the measurable residual is a format prior, and code is markedly more predictable and more memorisable than prose.

== Scope: The Family, Not One Implementation

The scope is the family, not one implementation. DeepSeek Engram and Qwen3.8-Flash-Next address with 2- and 3-grams and keep the convolution, giving a window of roughly eleven tokens; DeepSeek-V4.1-Flash uses 2-, 3- and 4-grams but removed the convolution, so its window is exactly four tokens and is in that respect the tightest of the three. Enlarging the table does not relax the bound; only a query-dependent address, as in retrieval, would.

= Does Memory Content Reach the Output?

The bound makes a sharp and testable prediction about content, and we tested it directly on two backbones. Train three readers that differ only in the corpus feeding the table --- Wikipedia, code, and STEM text --- and graft each into the same backbone at the same layer under the same prompts. If the memory carried corpus content, the three should answer differently.

== Under the Standard Recipe, They Do Not

The bound also makes a sharp prediction about *content*, which we tested directly on two backbones. Train three readers that differ only in the corpus feeding the table --- Wikipedia, code, and STEM text --- and graft each into the same backbone at the same layer under the same prompts. If the memory carried corpus content, the three should answer differently. Under the standard recipe they do not. On Qwen3.5-0.8B in fp32, over 1500 items per arm, the answer-label distribution, the leading-surface distribution and their joint all differ by a total-variation distance of exactly 0.0000, and every pre-registered directional lexical prediction fails, with zero discordant pairs for code markers. On Qwen3.5-4B in bf16 --- the backbone on which the format effect was first observed --- the same null holds over 600 items per arm: the largest cross-corpus spread is a single item, below the split-half floor of the arm that produced it. What the corpus changes is the part of the trigram statistic the backbone weights already encode, and under this recipe that part leaves no trace in the output.

== The Manipulation Reached the Injection Point

That the null is not a failed manipulation is established independently of the output. The three readers write measurably different vectors at the answer position (pairwise cosine similarity 0.73 to 0.76, relative $L_2$ distance 0.69 to 0.78, norms within a factor of 1.10), and in a $3 times 3$ cross-evaluation each is best on its own corpus. The frozen tensors are bit-identical across arms and the adapter-norm ratios span only $0.886$ to $1.045$, well inside the threefold bound that would make the arms incomparable.

== The Effect Is Large, and Larger on the Bigger Backbone

The contrast with the zero-injection arm shows the channel is not merely inert, and the effect is far larger on the 4B backbone. There, zero injection makes the model emit chat scaffolding on 95.8% of items (mean 31.2 generated tokens), while injecting a reader trained on any of the three corpora suppresses it to 0.0% (mean 2.6 tokens), with not one of the 600 items sharing its first token with the zero-injection arm ($p = 1.6 times 10^(-173)$, McNemar exact). On 0.8B the same shift moves scaffolding from 19.5% to 0.0% and changes the first generated token on all 1500 items ($p = 1.3 times 10^(-88)$); the reader-disabled and no-reader arms are bit-identical, so the noise floor is exactly zero rather than merely small. The graft therefore does something large and reproducible --- it pushes the model off the chat template --- but that shift does not depend on what the table contains.

== Removing the Ceiling

This null has a scope, and testing it changes what the result means. On both backbones the injected arms saturate at 2.5 to 2.7 generated tokens, so "no format difference" could mean "no room for one". We therefore ran the regime that removes the saturation --- no answer-format SFT, where generated length returns to 11 to 21 tokens --- over all six arms on 600 items, so the same pre-registered rule applies without adjustment. The reader-disabled and no-reader arms are again bit-identical, giving a noise floor of exactly zero. The two halves of the rule then disagree, and the disagreement is the result. The coarse label taxonomy still reports a perturbation artifact, because the scaffold rate spans only $0.0100$ against a threshold of $0.0122$; every finer format descriptor reports content dependence. That division is why the co-primary descriptors were pre-registered alongside the taxonomy, on the stated grounds that the taxonomy can be insensitive to a real format shift. All four move outside each arm's own split-half sampling floor (leading-surface total variation 0.1433 against a floor of 0.0237; first-token 0.4983 against 0.2176), and both pre-registered directional lexical predictions hold: the code reader emits more code markers ($+0.0333$, one-sided $p = 3.9 times 10^(-4)$) and the Wikipedia reader emits far more prose markers ($+0.6933$, $p = 4.5 times 10^(-34)$), where under saturation both predictions had failed outright.

== The Layering Is What the Bound Predicts

So the correct statement is layered rather than absolute, and it is the layering the bound predicts. Suppressing the chat scaffold is content-independent --- all three readers suppress it, and they differ from each other by at most $0.01$. The rest of the output distribution is content-dependent, and that dependence *is* the trigram prior the bound allows the channel to carry: a reader trained on Wikipedia supplies Wikipedia-like surface statistics, one trained on code supplies code-like ones. What never appears, with or without the ceiling, is passage-conditioned recall --- the knowledge probes that require the passage return zero. The channel carries the corpus prior and nothing above it.

== How Much Prior Is Actually There

The bound permits the memory to carry a corpus's rare n-gram statistics, so the useful question is how much of that prior the frozen table actually holds and how much of it the read-out recovers. We probe the raw 2560-dimensional rows directly, on held-out windows, against explicit counts on the same corpus as a reference frame. Majority-class prediction gives a floor of $0.0537$ top-1. A count bigram reaches $0.2264$ and a count trigram $0.2535$. A ridge probe on the frozen rows reaches $0.1280$; an MLP reaches $0.1502$ at its best epoch and $0.1225$ at 24 epochs, so additional training makes it worse rather than better. Shuffled-row controls collapse below the majority floor at $0.0459$ and $0.0373$. The table therefore does hold a recoverable trigram prior --- well above the floor and destroyed by shuffling --- but the read-out recovers only about 59% of what explicit counts reach, and that gap is a property of the read-out rather than of undertraining.

== The Obvious Repair Does Not Pay

Nor does lengthening the key. Replacing the 4-gram address with 8-token, 16-token and longest-suffix memories at matched storage ties or loses at every budget and training size on both corpora, so the short window costs nothing measurable. That the window can be widened without gain, while the corpus the table is built from cannot be changed without gain either, is the same statement twice: the channel's output is fixed by what the window already determines.

= Problem Setting

We consider a frozen small language model with next-token distribution $p_b(y | c)$, where $c$ is the current token context. A sparse n-gram memory provides a complementary distribution $p_m(y | c)$ that can be audited by inspecting the matched n-gram and its source document. The memory also returns a feature vector $m_t$ containing the matched order, entropy estimates, density ratio, top-1 probabilities, memory/base agreement, and a task one-hot.

Our goal is to compute a per-token fused distribution:

$ p_{"fused"}(y) = "softmax"[ log p_b(y) + alpha_t log p_m(y) + beta_t ]. $

We call $(alpha_t, beta_t)$ the PLE correction. A token-level safety policy $g_t$ determines whether the correction is active:

$ g_t = "sigmoid"(w dot m_t + b). $

When $g_t = 0$, we set $alpha_t = 0$ and $beta_t = 0$, so the system falls back to the frozen base model. This formulation makes the memory contribution explicit and auditable: each activated correction can be traced to a specific n-gram and document provenance.

We evaluate with a strict real/control protocol. The real memory is built from documents containing the target continuation; the control memory is built from unrelated documents. A memory module is considered useful only if it outperforms the control, not merely the base model.

= Background

== Real versus Control

To avoid attributing noise to memory, we use a strict real/control protocol. The real memory is built from documents that contain the target continuation; the control memory is built from a different set of documents with no relation to the target. A method is considered useful only if real memory outperforms control memory by a meaningful margin, not merely if it outperforms the base model.

== Optimal Logit Correction

From a decision-theoretic perspective, the optimal way to combine a base model and a memory distribution is not to replace the base model, but to add a calibrated log-ratio correction @logopinion1986:

$ log p_{"fused"}(y) = log p_b(y) + lambda_t log p_m(y) + beta_t. $

This is a log-opinion-pool formulation. Without calibration, an unweighted n-gram prior is often over-confident and degrades generation @knnopen2023. The PLE Projector learns $lambda_t$ and $beta_t$ from hidden states and memory statistics.

== Why Hidden-State Alignment Is Insufficient

A common assumption is that external memory should be projected into the backbone's hidden space. Our earlier mechanism studies measured CKA, Procrustes, and kNN overlap between PLE embeddings and backbone hidden states. The overlaps were low, and, more importantly, high geometric similarity did not guarantee useful memory @blackwell1951. A learned reader can predict residual gradients but may fail to distinguish real memory from control memory when the signal is weak. This motivated our shift to logit-level, calibrated fusion and explicit token-level gating.


= Method

== PLE Memory and Retrieval

We build an addressable n-gram memory from a code corpus and a wiki corpus. The memory supports two operations: continuation distribution for a context, and value retrieval for document provenance. The memory is used as both a retrieval channel and a logit prior. In the hybrid system, BM25 @bm252009 provides document-level retrieval, PLE provides exact n-gram continuity, and their combination forms a three-channel retriever.

#figure(
  scope: "parent",
  image("figures/system_architecture_dd.svg", width: 100%),
  caption: [System architecture. The frozen backbone, RAG, and PLE memory provide three complementary evidence channels; the PLE Projector and token policy control logit-level fusion.]
)

== PLE Projector

The projector is a small MLP:

$ f_theta(h_t, m_t) = (alpha_t, beta_t). $

where $h_t$ is the last hidden state of the frozen backbone, and $m_t$ contains matched n-gram order, base entropy, memory entropy, density ratio, base top-1 probability, memory top-1 probability, memory/base agreement, and task one-hot. The output $alpha_t$ scales $log p_m$ and $beta_t$ adds a support bias. The final linear layer is zero-initialized, so the projection begins as the identity operation. Only the projector parameters are trained.

== Token-Level Policy

Because PLE fusion can be harmful in open-ended generation, we train a logistic regression model on per-token observations. The features are the same memory features used by the projector. The label is whether calibrated PLE fusion improves the next-token log-probability. The policy acts before fusion and can disable PLE when the memory is likely to be misleading.

#figure(
  scope: "parent",
  image("figures/inference_pipeline_dd.svg", width: 90%),
  caption: [Inference pipeline. The token policy decides whether to apply the learned PLE Projector or bypass it with base logits only.]
)

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
+ HumanEval: official @humaneval2021, first 50 problems.
+ TriviaQA: official @triviaqa2017, 200 validation examples.
+ Joint system tasks: knowledge, arithmetic, and code-output subsets, with arithmetic probes inspired by GSM8K and MATH style benchmarks @gsm8k2021 @math2021.

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

We use 5 seeds for both the 100-sample and 10k projector experiments. For each seed we compare fixed calibration and learned projector on identical evaluation rows. Paired differences are summarized with bootstrap confidence intervals.

= System Results Under the Bound

The results in this section were collected before the bound was stated, and they are reported here because the bound changes what they are evidence for. Read against it, they separate by mechanism rather than by task: the experiments that go through the residual stream behave as the bound requires, and the ones that do not go through it are unaffected by it. We take that separation to be the useful reading, and we mark it explicitly in the analysis below.


== Local Continuation

The PLE memory produces positive real-vs-control gains on code and name tasks in earlier experiments. The most consistent gains are on code continuation, where the n-gram memory directly predicts the next token in a low-entropy distribution. Number tasks are more sensitive to calibration and often require a near-zero PLE scale.

=== Projector Scaling

At 100 samples with 5 seeds, the projector-vs-fixed NLL mean is $+0.1044$ with bootstrap 95% CI $[0.0251, 0.1866]$. At 10k samples with 5 seeds, the improvement grows to $+0.2249$ with CI $[0.1436, 0.3255]$, and all five seeds are positive. The paired per-seed differences are all statistically significant in the 10k setting.

#figure(
  table(
    columns: 7,
    [Seed], [Base NLL], [Fixed NLL], [Proj. NLL], [Base hit], [Fixed hit], [Proj. hit],
    [0], [3.148], [3.051], [2.930], [0.407], [0.427], [0.463],
    [1], [3.328], [3.346], [3.114], [0.410], [0.420], [0.460],
    [2], [3.451], [3.426], [3.002], [0.413], [0.430], [0.493],
    [3], [3.238], [3.066], [2.848], [0.397], [0.420], [0.473],
    [4], [3.197], [3.041], [2.913], [0.400], [0.427], [0.417]
  ),
  caption: [10k local-continuation results, five seeds. NLL is lower-is-better.]
)

#figure(
  image("figures/fig_10k_improvement.png", width: 100%),
  caption: [Per-seed projector-vs-fixed improvements on 10k data.]
)

Across all five seeds the learned projector improves NLL over fixed calibration. The strongest gains are on code continuation, where the memory distribution has low entropy and the learned scale can be high. The scaling trend from 100 to 1k to 10k samples is consistent: more projector training data improves the learned scale/bias policy.

== HumanEval

On 50 official HumanEval problems with greedy decoding, base achieves $0.22$ #text("pass@1") while BM25+PLE achieves $0.10$. The BM25+PLE condition has a higher repetition rate. Although the overall greedy pass rate is lower, the two methods solve largely different problems.

#figure(
  table(
    columns: 3,
    [Condition], [#text("pass@1")], [mean repetition],
    [Base], [0.22], [0.025],
    [BM25+PLE], [0.10], [0.041]
  ),
  caption: [HumanEval 50 greedy results.]
)

#figure(
  image("figures/fig_humaneval.png", width: 100%),
  caption: [HumanEval #text("pass@1") and repetition.]
)

The base model passes HumanEval/16, 18, 23, 25, 32, 33, 38, 41, 43, 46, and 49. BM25+PLE passes HumanEval/0, 10, 27, 35, and 41. The only overlap is HumanEval/41. Thus BM25+PLE recovers four problems that the base model never solves, confirming that PLE provides a different source of local code knowledge rather than simply amplifying the base model's existing solution distribution.

=== Sampling and #text("pass@k")

Because greedy decoding can be sensitive to the exact modal token, we also evaluate sampling with 10 problems and 3 samples per problem. Under temperature $0.8$ and top-$p$ $0.9$, BM25+PLE improves #text("pass@k") from $0.40$ to $0.70$, while increasing repetition only slightly.

#figure(
  table(
    columns: 4,
    [Condition], [#text("pass@k")], [passed / 10], [mean repetition],
    [Base], [0.40], [4], [0.001],
    [BM25+PLE], [0.70], [7], [0.008]
  ),
  caption: [HumanEval #text("pass@k") with 10 problems and 3 samples per problem.]
)

The contrast between greedy #text("pass@1") and sampled #text("pass@k") suggests that the n-gram memory is not uniformly beneficial: it can suppress the base model's greedy mode in some cases, while helping exploration in a diverse sampling regime.

== TriviaQA

On 200 TriviaQA RC validation examples, the 0.8B base model obtains exact match $0.005$ and mean repetition $0.0076$. This is a truthful baseline: the model nearly always fails short-form knowledge QA @triviaqa2017. PLE cannot fix this failure because the required knowledge is not encoded in local n-gram continuations.

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

We use DeepSeek V4 Flash as an external judge @llmjudge2024. On 50 HumanEval problems, the mean score is $0.60$ for base and $0.20$ for BM25+PLE. On 200 TriviaQA examples, the mean judge score is $0.465$. For completeness, we also measured HumanEval 20 and TriviaQA 100 subsets at $0.375$ and $0.450$, respectively. A parallel API rerun produced slightly different values ($0.40$ overall on HumanEval 50, $0.43$ on TriviaQA 200), so judge scores should be interpreted with the usual LLM-judge variance caveat.

#figure(
  image("figures/fig_judge.png", width: 100%),
  caption: [LLM-as-judge mean scores.]
)

The judge scores are lower than pass-based metrics. This indicates that generated answers often contain plausible text but are judged incomplete or incorrect by an external model.

== Error Analysis and Sensitivity

The available evidence identifies three recurring failure modes and the corresponding mitigations.

#figure(
  table(
    columns: 3,
    [Failure mode], [Observed evidence], [Mitigation],
    [Over-confident n-gram prior], [Open-ended repetition and repository fragments], [Learned token policy],
    [Missing world knowledge], [TriviaQA exact match = 0.005], [RAG and parametric adapters],
    [Hidden-state misalignment], [Real-vs-control gains near zero], [Logit-level calibrated fusion]
  ),
  caption: [Error analysis and mitigating mechanisms.]
)

On sensitivity, we ran systematic sweeps over n-gram order and memory-bank size on the code task. The calibrated PLE fusion gain increases with n-gram order, from $0.456$ nats at order 2 to $0.605$ nats at order 5. The memory-size trend is non-monotonic: the smallest bank has the largest fused gain ($1.408$), while the largest bank has the smallest gain ($0.210$), suggesting that a compact same-domain n-gram bank is more useful than a large diffuse corpus for this local task.

#figure(
  table(
    columns: 3,
    [Setting], [Real-vs-control gap], [Calibrated fused gain],
    [order 2], [20.23], [0.456],
    [order 3], [20.69], [0.501],
    [order 4], [20.49], [0.574],
    [order 5], [20.53], [0.605],
    [memory 40/80], [22.68], [1.408],
    [memory 120/240], [19.47], [0.755],
    [memory 300/600], [20.28], [0.210]
  ),
  caption: [Sensitivity over n-gram order and memory-bank size on code continuation. The calibrated fused gain is the NLL improvement of calibrated real fusion over the base model.]
)

The token policy remains the main safety control for open-ended generation, while per-task calibration is the main control for number-like tasks. These systematic sweeps are now included in the public artifact bundle.

= Analysis

== When Does PLE Help?

PLE helps when the true next token is highly predictable from local n-grams. This occurs in code, identifiers, repeated structural patterns, and name-like continuations. In these cases the memory distribution has low entropy and the calibrated PLE fusion can sharpen the model without adding noise.

== When Does PLE Fail?

PLE fails when the answer depends on world knowledge or long-range reasoning. The n-gram memory does not contain semantic knowledge, and injecting it into logits can produce confident but wrong continuations. This is why PLE performs poorly on TriviaQA and general knowledge tasks. For token-keyed memory this failure is not contingent but necessary: the bound above shows the channel cannot carry what the addressing window does not already determine.

== Why a Learned Projector Matters

Fixed calibration selects one scale and bias for all tokens. A learned projector can vary the trust in PLE by context. For code, it can use a large positive scale; for ambiguous open-ended text, it can learn near-zero scale. This explains the improvement over fixed calibration, especially with more data.

== Boundary with RAG and Adapters

Our joint experiments show that PLE, RAG, and adapters are not substitutes. RAG supplies document-level evidence, PLE supplies token-level continuity, and adapters supply parametric capability @rag2020 @lora2022. The combination is useful, but PLE is only one small component.

= A Note on Method: Ten Times the Check Was Not Run

Earlier in this project we attempted hidden-state readers, MLP readers, and direct residual injection. These approaches often improved loss-like metrics but failed the real-vs-control test @blackwell1951. The bound above explains why they had to: a reader translating a token-keyed row into the residual stream cannot carry context the window does not determine, so the real and control arms converge once the reader is well trained. The lesson is twofold --- a memory module must be evaluated by whether it uses actual memory content rather than by loss alone, and a channel must be checked against its information-theoretic ceiling before its capacity is scaled. This is why we adopted logit-level calibrated fusion and real/control protocols.

A second and less comfortable lesson came from auditing those results. Ten times in this project a change was made, a claim was written, and the check that would have falsified the claim either did not exist or silently skipped --- so the claim stood unexamined. A runtime used a different row-identifier implementation than the one that had been proved correct. A convolution believed absent was in the executed path. A loop body was indented such that a metric scored four items instead of 1500. Four golden tests were skipping rather than passing on the machine that produced every number. And the forward pass of the reader underlying every graft measurement had never been compared to the official mathematics it claimed to reuse --- when we finally compared them they agreed bit-for-bit, which is the outcome that makes the rest of this section readable rather than a cautionary tale. None of these were found by re-reading code. Each was found by making the failure detectable: invariant assertions, regression tests that fail on the specific bug, a flag that converts a silent skip into a failure, and manifests recording a fingerprint of what was actually measured. In every case the durable fix was the detection mechanism, not the correction.

Two further mistakes of the same family surfaced during that audit and were caught within minutes --- not because the code was read more carefully, but because the checks had by then been made to run: a constant copied from the wrong fixture, and a skip flag applied to a 65 MB asset that is not in the repository. We report both families because they carry different lessons. The first says that a claim is only as trustworthy as the check behind it; the second says that once the checks are real, ordinary carelessness stops being able to hide.

A third lesson is about failure that is not a mistake. Our final experiment pre-registered a prediction --- that the injected vector would be more nearly constant for a task whose addressing window is mostly a fixed instruction suffix than for tasks whose windows carry query text --- and the prediction was wrong. Within-task similarity came out at $0.9994$ and $0.9992$, a difference of $+0.0002$ that a permutation test over task labels declined to distinguish from noise ($p = 0.27$). The value of fixing the statistic and its refutation condition in advance was not that it protected us from a wrong result. It is that the wrong result is legible: because the rule could not be adjusted after the fact, the failure pointed straight at the near-constant read-out, which is a stronger and more useful finding than the prediction would have been.


The bound does not partition these mechanisms the way one might expect, and the correction matters. Logit-level fusion of an n-gram memory is *also* a function of the addressing window, so the bound applies to it exactly as it applies to the residual graft; what the projector gets is not an exemption but a better interface. The permitted prior is the same in both cases. The graft loses it twice --- once through a read-out that recovers only about 59% of the trigram top-1 explicit counts reach, and again through a residual-stream bottleneck in which a single vector must be disentangled by the backbone before it can affect any logit. The logit correction loses it once, and spends it in the basis where it is used.

That read-out loss is not a small correction, and measuring the injected vector directly shows how large it is. Over 600 diverse prompts the vector the reader adds to the residual stream has an effective dimensionality of $1.001$: $99.93\%$ of its variance lies in a single direction, and two prompts are separated by about two degrees. The hidden state the reader is gated on sits at $1.480$, with $81.9\%$ in its leading direction. A generic linear map does not do this --- three random maps of the same shape leave the hidden state's ratio at $1.454$, $1.472$ and $1.485$ --- and replacing the retrieved rows with permuted ones, at the same checkpoint, moves the injected direction by about five degrees. The read-out is therefore close to a constant function of its input. The consequence is that the content-independence we measure under the standard recipe has two candidate explanations, and our experiments separate their consequences without separating the explanations themselves: either the window has nothing to give beyond the hidden state, or the read-out does not transmit what the window has. The bound is a theorem and holds in both cases, but the second explanation is a defect rather than a limit.

That reframing is testable against our own numbers, and it holds. The projector's gain is confined to low-entropy local continuation --- precisely the region the window determines, which is the only region the bound leaves open --- and it is absent on knowledge tasks, where the determining information lies outside the window. Two mechanisms escape the bound, and only two: retrieval, which addresses by query rather than by a fixed token window, and parametric adapters, which are not a memory channel at all. PLE, RAG and adapters are therefore not three points on one axis of strength but three different interfaces to three different resources, and only the first of them is bounded in the way this paper describes.
= Limitations

+ HumanEval is limited to 50 problems, although the pass sets are already informative.
+ TriviaQA exact match is very close to zero ($0.005$), so the paper reports a boundary rather than a positive result.
+ #text("pass@k") evidence is a 10-problem, 3-sample experiment, not a full benchmark-scale estimate.
+ LLM judge scores show run-to-run variance and should be treated as secondary evidence.
+ CPU deployment throughput is still only about $2$ tok/s with the current unoptimized eager implementation.
+ Public projector and adapter weights are available on Hugging Face, but the upstream Qwen model weights themselves are not redistributed.
+ The unsaturated regime that reveals the corpus's surface prior is measured over 600 items on one backbone under a different training recipe; it is a secondary regime and is reported as one. The saturated null it qualifies is measured on two backbones.
+ The reader-fidelity check covers the read-out, not the addressing: we verify that our reader reproduces the official mathematics bit-for-bit at production geometry, but the table itself is accessed through our own row-identifier implementation, which is separately pinned against two independent implementations.
+ The official Qwen3.8 model attaches its memory layer at 0-based layer 1, while our grafts inject at layer 2. The bound is a property of the reader's input-output map and is layer-independent, so the results above are unaffected, but the layer-1 configuration is untested rather than refuted.
+ The bound is elementary --- a data-processing inequality --- and we do not claim otherwise. Its value is in the window arithmetic it forces on a family that three production systems scaled without writing it down, and in the experiment that tests its content prediction.
+ The content-independence we measure under the standard recipe is consistent with two mechanisms --- an addressing window that carries nothing beyond the hidden state, and a read-out that transmits almost nothing --- and we do not separate them. Direct measurement shows the second is real (effective dimensionality $1.001$; a five-degree response to permuted rows), and the table probe shows the window's trigram statistics sit above the majority floor, so the first is not the whole story. The bound holds either way, but the share of the design's failure attributable to it rather than to the read-out is left open.

= Conclusion

We asked what a token-keyed residual memory can carry, and the answer is set before any reader is chosen. The row identifier is a deterministic function of a short addressing window, so the memory's contribution to the future cannot exceed what that window already determines and the hidden state has not retained. In the Engram and Qwen3.8-Flash-Next designs that window is twelve tokens rather than the two an order-3 key suggests, because a kernel-4 dilation-3 convolution follows the gate; in DeepSeek-V4.1-Flash the convolution was removed and the window is exactly four.

The prediction this makes about content survives a direct test. Three readers that differ only in the corpus feeding the table write measurably different vectors and, under the standard recipe, produce output distributions that are identical to within a single item out of 600. Removing the answer-length ceiling does not break the bound but completes it: the corpus's surface statistics then appear in the output, which is the trigram prior the channel is permitted to carry, while passage-conditioned recall remains at zero. The memory is a channel for a corpus's surface prior and not for knowledge, and which of the two it is, is fixed by the addressing window.

Two measurements fix the size of the remaining opportunity, and they point in different directions. The read-out recovers about 59% of the trigram top-1 that explicit counts reach, and more training makes it worse, so the loss is architectural rather than a matter of budget; measured directly, the vector it injects is nearly a constant, with an effective dimensionality of $1.001$ against $1.480$ for the hidden state it reads. Because the table's trigram statistics sit well above the majority floor, this gap is a target rather than a verdict: the design fails for the bound's reason and for a read-out defect at the same time, and only the first of those is unfixable at a twelve-token window. Lengthening the key does not pay at matched storage on any corpus or budget we tested. What is left is therefore the complement --- retrieval, which addresses by query, and parametric adapters, which are not a memory channel at all --- together with the read-out itself. The system results in this paper are consistent with that division and are re-read under it.

= Acknowledgement

We thank the open-source community for the PLE/Engram, Qwen, and related memory infrastructure used in this study. This work was carried out as an independent low-resource research project.

= Data Availability

All source code, evaluation scripts, container definitions, evaluation cards, and checksums are publicly available in the project repository and in the Hugging Face artifact repository. The released artifact bundle includes PLE projector checkpoints, Purified OPSD MoRA adapters, projector datasets, configurations, and the raw evaluation result files. The PLE memory tables can be rebuilt from the published corpus construction scripts, and upstream Qwen model weights are not redistributed.

#set heading(numbering: none)

#pagebreak()

= Appendix

== A. Algorithm: PLE Projector Inference

The projector produces a per-token logit correction without modifying the frozen backbone. The learned variables are $alpha_t$ (memory trust) and $beta_t$ (support bias).

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [PLE Projector inference],
  [
    + Input: context $c$, hidden state $h_t$, memory features $m_t$, memory distribution $p_m$.
    + Compute $alpha_t, beta_t = f_theta(h_t, m_t)$.
    + Form the corrected distribution $p_{"fused"}(y) = p_b(y) dot p_m(y)^(alpha_t) dot exp(beta_t)$.
    + Evaluate the token policy $g_t in {0, 1}$.
    + If $g_t = 0$, reset $alpha_t <- 0$ and $beta_t <- 0$.
    + Normalize and return $p_{"fused"}$.
  ]
)

== B. Algorithm: Token-Level Policy and Safety Gate

The policy is a small binary classifier trained on the same memory features used by the projector. It decides whether the n-gram memory should contribute to the current token.

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Token-level policy and safety gate],
  [
    + Train a logistic classifier on paired real/control observations.
    + At inference, compute features $m_t$ from the current context.
    + Predict $g_t = P("helpful" | m_t)$.
    + If $g_t < tau$, disable PLE fusion and use base logits only.
    + Otherwise, apply the learned projector correction and continue decoding.
  ]
)

== C. Real-vs-Control Protocol

To distinguish genuine memory use from model noise, we use a strict paired protocol.

+ The real memory is built from documents containing the target continuation.
+ The control memory is built from an unrelated document set with no target relation.
+ A method is considered useful only if it outperforms the control memory by a meaningful margin.
+ All projector results in the paper are paired across identical evaluation rows.

== D. HumanEval Problem-Level Passes

On the 50-problem HumanEval subset, the base model solves 11 problems and BM25+PLE solves 5 problems. The two pass sets overlap only at HumanEval/41. This supports the claim that PLE provides a different source of local code knowledge rather than simply amplifying the base model, and also shows that BM25+PLE does not uniformly dominate greedy decoding.

== E. Case Study: TriviaQA Failure

The frozen 0.8B model obtains only $0.005$ exact match on the 200-example TriviaQA validation subset. The failure is systematic rather than a calibration artifact: short-form knowledge questions require world knowledge that is not present in local n-gram continuations. A raw PLE fusion cannot recover this knowledge and may instead produce plausible but incorrect continuations.

== F. Case Study: Open-Ended Degradation

Without the token-level policy, unconditional PLE fusion on natural-language code prompts produces repetitive fragments, unrelated repository text, and broken structure. The learned policy suppresses PLE on tokens where the n-gram prior is not clearly beneficial, which substantially reduces this degradation.

== G. Limitations and Future Work

+ HumanEval is limited to 50 problems in the current version.
+ TriviaQA exact match is $0.005$, so the paper reports a boundary rather than a positive result.
+ Public projector and adapter weights are released, but upstream model weights are not redistributed.
+ CPU latency and quantization remain unoptimized.
+ Future work includes extending memory-size scaling and full joint-system evaluation on larger real benchmarks.
