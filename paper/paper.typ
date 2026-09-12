#import "sty/icml2024.typ": icml2024

#show: icml2024.with(
  title: [What a Token-Keyed Residual Memory Can Carry],
  authors: (
    ((name: "QingGo", affl: "affl1", email: "zyqingjohn@qq.com"),),
    (affl1: ("Independent Researcher",),)
  ),
  abstract: [
    Sparse n-gram memory layers --- DeepSeek Engram, Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash --- inject a retrieved vector into the residual stream at every token, and are being scaled on the premise that the memory supplies knowledge the parameters do not hold. We show that the channel is bounded before any reader is chosen: the row identifier is a deterministic function of a short addressing window, so the data-processing inequality gives $I("future" ; e_t | h_t) <= I("future" ; w_t | h_t)$. Depth, width, linearity and training budget do not appear. The window is not the two tokens an order-3 key suggests but twelve, because a kernel-4 dilation-3 convolution follows the gate; in V4.1-Flash, which removed the convolution, it is exactly four.

    Testing that prediction, three readers differing only in the corpus behind the table --- Wikipedia, code and STEM --- write measurably different vectors (pairwise cosine 0.38 to 0.49 on 4B) and yet produce identical output distributions: total variation $0.0000$ over 1500 items on 0.8B, and on 4B a largest cross-corpus spread of one item out of 600. Removing the answer-format SFT that collapses every arm to 2.5 generated tokens reveals the other half of the bound: all four format descriptors then move outside their split-half floors, and both pre-registered directional lexical predictions hold. Suppressing the chat scaffold is content-independent; the rest of the distribution carries the corpus's surface statistics, which is the trigram prior the bound permits. Passage-conditioned recall never appears, with or without the ceiling.

    Two measurements then point in different directions. A probe of the frozen table recovers about 59% of the trigram top-1 that explicit counts reach on the same corpus, and additional training makes it worse. Measured directly, the vector the reader injects has an effective dimensionality of $1.001$ over 600 prompts, against $1.480$ for the hidden state it is gated on --- the read-out is close to a constant function of its input. The design therefore fails for the bound's reason and for a read-out defect at once, and only the first of those is unfixable by improving the reader.
  ],
  bibliography: bibliography("refs.bib"),
  accepted: none,
  appendix: [
    = System Architecture and Inference Pipeline

    #figure(
      image("figures/system_architecture_dd.svg", width: 44%),
      caption: [System architecture. The frozen backbone, RAG, and PLE memory provide three complementary evidence channels; the PLE Projector and token policy control logit-level fusion.]
    )

    #figure(
      image("figures/inference_pipeline_dd.svg", width: 32%),
      caption: [Inference pipeline. The token policy decides whether to apply the learned PLE Projector or bypass it with base logits only.]
    )

    = Algorithm: PLE Projector Inference

    The projector produces a per-token logit correction without modifying the frozen backbone. The learned variables are $alpha_t$ (memory trust) and $beta_t$ (support bias).

    #figure(
      kind: "algorithm",
      supplement: [Algorithm],
      caption: [PLE Projector inference],
      [
        + Input: context $c$, hidden state $h_t$, memory features $m_t$, memory distribution $p_m$.
        + Compute $alpha_t, beta_t = f_theta(h_t, m_t)$.
        + Form the corrected distribution $p_(upright("fused"))(y) = p_b(y) dot p_m(y)^(alpha_t) dot exp(beta_t)$.
        + Evaluate the token policy $g_t in {0, 1}$.
        + If $g_t = 0$, reset $alpha_t <- 0$ and $beta_t <- 0$.
        + Normalize and return $p_(upright("fused"))$.
      ]
    )

    = Algorithm: Token-Level Policy and Safety Gate

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

    = Real-vs-Control Protocol

    To distinguish genuine memory use from model noise, we use a strict paired protocol.

    + The real memory is built from documents containing the target continuation.
    + The control memory is built from an unrelated document set with no target relation.
    + A method is considered useful only if it outperforms the control memory by a meaningful margin.
    + All projector results in the paper are paired across identical evaluation rows.

    = Projector Scaling, Baselines and Sensitivity

    Per-seed 10k local-continuation results, the kNN-LM and NGM comparisons, and the sweep over n-gram order and memory-bank size.

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
    ) <tbl-10k>

    #figure(
      table(
        columns: 4,
        [Baseline], [NLL], [Hit], [Note],
        [Base], [2.633], [0.505], [kNN eval set],
        [kNN-LM], [2.847], [0.540], [better hit, worse NLL],
        [Base], [2.521], [0.550], [NGM eval set],
        [NGM], [2.521], [0.550], [near-zero change]
      ),
      caption: [kNN-LM and NGM baselines. Neither matches the learned projector.]
    )

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
      caption: [Sensitivity over n-gram order and memory-bank size on code continuation. Gain is the NLL improvement of calibrated real fusion over the base model; all rows are single-seed exploratory sweeps, not pre-registered comparisons.]
    )

    #figure(
      image("figures/fig_10k_improvement.svg", width: 24%),
      caption: [Per-seed projector-versus-fixed improvements on 10k data. Darker bars are seeds whose per-seed artifacts are retained in the released bundle and reproduce @tbl-10k exactly; the two lighter bars are reported from the original run, whose per-seed files were not retained.]
    )

    = HumanEval and Judge Detail

    Under greedy decoding the base model passes HumanEval/16, 18, 23, 25, 32, 33, 38, 41, 43, 46 and 49; BM25+PLE passes HumanEval/0, 10, 27, 35 and 41. The only overlap is HumanEval/41, so BM25+PLE reaches four problems the base model never solves --- but it loses nine, and with no confidence interval over 50 problems we treat the recovery as suggestive rather than established.

    #figure(
      image("figures/fig_humaneval.svg", width: 30%),
      caption: [HumanEval #text("pass@1") and mean repetition rate over 50 problems.]
    )

    #figure(
      image("figures/fig_joint_system.svg", width: 26%),
      caption: [Joint system answer log-probability by resource, three seeds.]
    )

    #figure(
      image("figures/fig_judge.svg", width: 24%),
      caption: [LLM-as-judge mean scores. Judge scores carry run-to-run variance and are reported as secondary evidence.]
    )

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

    = Case Studies

    *TriviaQA failure.* The frozen 0.8B model obtains $0.005$ exact match on the 200-example validation subset. The failure is systematic rather than a calibration artifact: short-form knowledge questions require world knowledge that is not present in local n-gram continuations, and raw PLE fusion cannot recover it.

    *Open-ended degradation.* Without the token-level policy, unconditional PLE fusion on natural-language code prompts produces repetitive fragments, unrelated repository text, and broken structure. The learned policy suppresses fusion on tokens where the n-gram prior is not clearly beneficial, which substantially reduces this degradation.
  ],
)

#set enum(indent: 1em, spacing: 0.55em)
#set list(indent: 1em, spacing: 0.55em)

= Introduction

Sparse n-gram memory is one of the few ways to add capacity to a language model without adding parameters. DeepSeek Engram, Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash all hash a short token window to a row of a large disk-backed table, project that row, and inject the result into the residual stream @engram2026. Memory Grafting scales pre-training with an offline conditional memory table @memorygrafting2026, and XMemTransfer adapts a table across model families with a target-side reader @xmemtransfer2026. The premise is that the table holds associations the weights compress away, and that a large enough table behind a good enough reader will supply them.

#figure(
  scope: "parent",
  image("figures/fig_window_bound.svg", width: 100%),
  caption: [
    The channel is bounded by its addressing window, and the window is larger than
    the key order suggests. #emph[(a)] An order-3 key reads two tokens, but the
    Engram and Qwen3.8 readers follow the gate with a kernel-4 dilation-3 depthwise
    convolution, so the contribution at a position draws on rows nine positions
    back, each of which reads two tokens further: twelve tokens, not two.
    V4.1-Flash removed the convolution and its window is four. #emph[(b)] Because
    $e_t$ is a deterministic function of $w_t$, no reader can exceed what the window
    already determines and $h_t$ has not retained.
  ],
)

This paper asks what such a channel can carry, and answers it in a way that does not depend on the reader. The argument is short. The row identifier is a deterministic function of a short addressing window $w_t$; the reader's contribution is a deterministic function of that row and the hidden state $h_t$. The data-processing inequality then bounds the memory's contribution to the future by the window's. Reader depth, width, linearity and training budget do not appear because they cannot. The bound is elementary; what is not elementary is that it was not written down for a family that three production systems are scaling, and that the window it depends on is not the one the headline key order suggests.

That last point is where the bound becomes useful rather than obvious. An order-3 key reads two tokens, but the Engram and Qwen3.8 readers follow the gate with a depthwise convolution of kernel 4 and dilation 3, so the contribution at a position draws on memory rows up to nine positions back, each of which reads two tokens further: the true window is twelve tokens, not two. DeepSeek-V4.1-Flash removed that convolution, so its window is exactly four. The family spans a factor of three in the one quantity that bounds it, and that quantity is visible only by reading the convolution parameters.

The bound makes a sharp and testable prediction about content, and we test it on two backbones. Train three readers that differ only in the corpus feeding the table --- Wikipedia, code and STEM text --- and graft each into the same frozen model at the same layer under the same prompts. If the memory carried corpus content, the three should answer differently. Under the standard recipe they do not: on 0.8B over 1500 items per arm the answer-label distribution, the leading-surface distribution and their joint differ by a total-variation distance of exactly $0.0000$, and on 4B the largest cross-corpus spread is a single item out of 600. This is not a failed manipulation. The three readers write visibly different vectors at the answer position, and each is best on its own corpus in a cross-evaluation; the corpus changes the part of the trigram statistic the weights already encode, and under this recipe that part leaves no trace.

It would be a mistake to stop there, and we did not. Every arm in that experiment collapses to a two-token answer, so "no format difference" could mean "no room for one". Removing the answer-format supervision that causes the collapse changes the conclusion, and the change is the bound's other half rather than a counterexample to it: all four format descriptors move outside their split-half floors and both pre-registered directional lexical predictions hold, with the Wikipedia reader emitting far more prose markers than the code reader. The layering is exactly what the bound predicts. Suppression of the chat scaffold is content-independent --- all three readers suppress it and differ from one another by at most $0.01$ --- while the rest of the distribution carries the corpus's surface statistics. That dependence *is* the trigram prior the channel is permitted to carry. What never appears, with or without the ceiling, is passage-conditioned recall; the knowledge probes that require the passage return zero.

Finally we ask how much of the permitted prior actually survives the reader, and the answer complicates the story in a way we did not expect when we began. Probing the frozen table directly shows that it does hold a recoverable trigram prior, well above the majority floor and destroyed by shuffling, but that the read-out recovers only about 59% of what explicit counts reach. Measuring the injected vector directly shows why: over 600 diverse prompts it spans effectively one dimension, while the query state it reads sits at $1.480$. So the design fails for two independent reasons --- one the bound, which is not fixable at a twelve-token window, and one a read-out that has collapsed, which is. Separating them matters, because it changes what the negative result is evidence for, and we are explicit below about the limits of our ability to do so.

Our contributions are these.

+ We state and prove a bound on what a token-keyed residual memory can carry, in a conditional form that accounts for the gate's dependence on the hidden state, and we identify the addressing window of each production design in the family --- twelve tokens for Engram and Qwen3.8-Flash-Next, four for V4.1-Flash.
+ We test its content prediction directly on two frozen backbones with three corpus-matched readers, and show that the three inject measurably different vectors while producing identical output distributions.
+ We show that the null is scoped rather than absolute: removing the answer-length ceiling reveals the corpus's surface prior, exactly where the bound says it should be, while passage-conditioned recall remains at zero.
+ We measure the ceiling from the other side, probing the frozen table directly and finding that the read-out recovers about 59% of the trigram top-1 explicit counts reach, a loss that does not improve with more training.
+ We measure the injected vector itself and find that it is nearly a constant function of its input, which means the null has two candidate causes rather than one; we report what our experiments do and do not separate.
+ We falsify the obvious repair. At matched storage, longer keys do not pay on any corpus or budget tested, so the short window is not the constraint that binds.
+ We report the audit that made these results checkable, including ten occasions on which a claim stood because the check that would have falsified it did not exist or silently skipped, and one pre-registered prediction that failed.

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

$ p_m(y mid c) = "count"(c, y) / sum_y "count"(c, y). $

The memory also returns the longest matched order and an external value index that can be audited. This memory is non-parametric, transparent, and can be rebuilt from any corpus.

== The Bound

The Engram / PLE line injects a retrieved vector into the residual stream. What such a channel can carry does not depend on the reader, and because the scope of the negative results below is exactly the scope of this statement, it is worth setting out with its assumptions.

#emph[Setting.] Fix a token context $c = (c_1, ..., c_t)$. Write $Y$ for the *#emph[future]*: where we score a known continuation $Y$ is that continuation, and where we measure emitted text it is the completion the model goes on to produce. Let $w_t$ be the addressing window ending at $t$, $a_t = A(w_t)$ the row identifier, $e_t = E(a_t)$ the retrieved rows, $h_t$ the backbone hidden state, and

$ c_t = F(h_t, e_(t-r:t)) $

the contribution added to the residual stream, where $r$ is the reach of the short causal convolution that follows the gate. Each retrieved row is keyed by an $n$-gram ending at its own position, so $r$ positions of convolutional reach plus an $n$-token key give a window of exactly

$ |w_t| = r + n = (k - 1) n + n = k n $

tokens for a kernel-$k$ convolution with dilation $n$. Engram and Qwen3.8-Flash-Next use $k = 4$, $n = 3$, so $|w_t| = 12$: twelve tokens, not the two an order-3 key alone implies. @sec:window measures this by execution rather than by reading a config.

#emph[Assumption (deterministic read-out).] $A$, $E$ and $F$ are deterministic. This holds at inference in every design in the family: the only stochasticity in the graft is the sampling of $Y$ itself, which is the quantity being predicted rather than a part of the channel. Were the read-out stochastic --- inference-time dropout, or an address perturbed by noise $xi$ --- the bound still holds conditionally, $I(Y ; c_t | h_t, xi) <= I(Y ; w_t | h_t, xi)$, and marginalising over $xi$ cannot increase the left-hand side, since mutual information is convex in the channel.

#emph[Theorem.] Under determinism, $I(Y ; c_t | h_t) <= I(Y ; e_(t-r:t) | h_t) <= I(Y ; w_t | h_t)$.

#emph[Proof.] Conditioned on $h_t$, $c_t = F(h_t, e_(t-r:t))$ is a function of $e_(t-r:t)$ alone, so $Y -> (h_t, e_(t-r:t)) -> c_t$ is a Markov chain and the data-processing inequality gives the first bound. Each row $e_(t-j)$, $0 <= j <= r$, is $E(A(w_(t-j)))$, and $w_(t-j)$ lies inside $w_t$ because the convolution only reads rows whose keys fall within the same $k n$ tokens; so $e_(t-r:t)$ is a deterministic function of $w_t$ given $h_t$, and the second bound is the same inequality again. ∎

Three consequences follow, and three qualifications, which matter at least as much.

#emph[Consequence 1: the bound is independent of the reader.] Depth, width, linearity and training budget do not appear in it, so no reader, however expressive, can exceed it. This turns our earlier failures with hidden-state readers and direct residual injection from an empirical observation into a necessary one.

#emph[Consequence 2: it constrains addressing, not capacity.] The memory can supply only what the window already determines and the hidden state has not retained, so no context-conditioned recall is possible when the determining information lies outside the window. For a question answered from a passage, the window at the answer position carries the prompt's format rather than its content: the memory may sharpen *#emph[how]* an answer is emitted, which is the format effect we measure, but not *#emph[which]* answer is correct.

#emph[Consequence 3: it does not say the memory is useless.] Backbone weights are a lossy compression of the training corpus, and rare n-gram statistics are among what is compressed away. The bound leaves room for precisely that, and it predicts where: gains should concentrate on rare n-grams and on continuations the window determines, and vanish elsewhere. We can test two parts of that prediction and do: knowledge probes requiring the passage return zero, and the measurable residual is a format prior rather than content. The rarity half we cannot test, and it is worth saying why. Correlating gain with context rarity requires a count for the window n-gram, and no affordable reference corpus supplies one: over a 3.1M-token corpus 93% of the evaluated windows have a zero count even at the table's own addressing order of three, so a tertile split degenerates, and at the window scale essentially every twelve-gram is unique in any corpus one could count. The rarity prediction therefore remains a prediction.

#emph[Qualification 1 (the bound constrains information, not effect size).] The theorem bounds what the memory can tell the model about the future. It does not bound how much the memory can change the model's behaviour, and the two come apart sharply. A contribution $c_t$ that is constant --- independent of $w_t$ given $h_t$ --- satisfies $I(Y ; c_t | h_t) = 0$ while still shifting the residual stream and therefore the output distribution. So a large, reproducible behavioural effect is no evidence against the bound, and the bound is no argument that an injection is harmless. This is exactly the regime §4.3 and §4.7 measure: the graft moves the model off its chat template on essentially every item while, on the measurements available to us, carrying almost nothing about the future.

#emph[Qualification 2 (the address may depend on the state, and that does not help).] Because the bound is conditional on $h_t$, an address that also reads the hidden state does not escape it: if $a_t = A(w_t, h_t)$ then given $h_t$, $e_t$ remains a function of $w_t$, and the theorem is unchanged. What escapes the bound is addressing on information $h_t$ has *#emph[not]* retained --- retrieval over the full context, which is the complement of a lossy summary rather than a wider view of the same neighbourhood. An earlier draft of this paper claimed that any query-dependent address would relax the bound; that is wrong for the reason just given, and the correction matters for how the family is read.

#emph[Qualification 3 (the bound does not compose across positions).] The theorem is per position and conditions on $h_t$, which is itself a function of everything injected before $t$. It therefore bounds what the memory adds at a position given the state it has already produced; it does not bound what the memory contributes to a sequence. A chain rule over positions would have to allow for the fact that $h_t$ is not a sufficient statistic for the past, so rows injected at $t' < t$ persist in $h_t$ and can influence tokens arbitrarily far ahead. Concretely: the per-position bound forbids the channel from introducing, at any single position, more about the future than the window determines beyond the current state, and it does not forbid accumulation across positions. The effect we measure is precisely such an accumulation, and a per-position bound of zero would not have predicted it --- which is one reason we report the format shift as a finding rather than as a puzzle for the theory.

#emph[Qualification 4 (tightness).] The bound is vacuous when $I(Y ; w_t | h_t)$ is large, and for our tasks it is not zero: an explicit count trigram over the same window reaches $0.2535$ top-1 against a majority floor of $0.0537$ (§4.6), so the window's last three tokens do carry next-token information the backbone has not retained. A zero content result is therefore not by itself evidence for the bound. Separating the bound from a read-out that fails to transmit is the subject of §4.7, and we do not fully succeed.

== Scope: The Family, Not One Implementation

The scope is the family, not one implementation. DeepSeek Engram and Qwen3.8-Flash-Next address with 2- and 3-grams and keep the convolution, giving a window of twelve tokens; DeepSeek-V4.1-Flash uses 2-, 3- and 4-grams but removed the convolution, so its window is exactly four tokens and is in that respect the tightest of the three. Enlarging the table does not relax the bound, and neither does making the address depend on more of the same window.

One clarification is worth making, because it is easy to over-read. The bound is conditional on $h_t$, so an address that depends on the hidden state as well as the window does not escape it: if $a_t = A(w_t, h_t)$ then $e_t$ is still a function of $(w_t, h_t)$ given $h_t$, and $I("future" ; e_t | h_t) <= I("future" ; w_t | h_t)$ is unchanged. What escapes the bound is addressing on information that $h_t$ has *not* retained --- retrieval over the full context, which is a lossy summary's complement rather than a wider view of the same neighbourhood.

== The Window, Measured <sec:window>

The arithmetic above is a claim about the *#emph[executed]* code path, and this project has been wrong before about what is executed --- one audit item was a convolution believed absent that was in fact in the path (§7). So we measure the window rather than derive it. Build the official PLE layer from the official geometry, hold the query hidden state fixed, change a single input token at lag $L$, and record whether the layer's output at position $t$ moves. The support of that response is the effective receptive field; the test asserts support, not magnitude, so it is insensitive to dtype, seed and scale.

#figure(
  table(
    columns: 5,
    align: center,
    [kernel $k$], [key order $n$], [conv pad $(k-1)n$], [predicted $k n$], [measured],
    [4], [3], [9], [12], [*12*],
    [1], [4], [0], [4], [*4*],
    [2], [3], [3], [6], [6],
    [4], [2], [6], [8], [8],
    [3], [3], [6], [9], [9],
    [4], [4], [12], [16], [16],
  ),
  caption: [
    The window is the product of kernel size and key order, confirmed by
    perturbing one token at a time and observing which output positions respond.
    For every row, all lags inside the predicted window moved the output and all
    lags outside it moved it by exactly zero. Rows 1 and 2 are the two production
    geometries: kernel 4 with order-3 keys is Engram and Qwen3.8-Flash-Next, and
    a kernel of 1 --- the convolution removed --- is the ablation that leaves
    V4.1-Flash with the key order itself. Checked by
    @sec:window in the project's test suite.
  ],
)

Two readings follow. First, the twelve-token figure is a property of Engram and Qwen3.8 and not of the family: removing the convolution collapses the window to the key order, which is why V4.1-Flash's window is four rather than three-to-four times larger. We take V4.1-Flash's design from its published description and model it as that ablation; we have not run its released weights, so that row is a statement about the design as described rather than about an implementation we measured. Second, the window is architectural: it is fixed by $k$ and $n$ and does not move with the embedding-table size, which is what licenses the small-configuration tests above as evidence about production geometry.

= Does Memory Content Reach the Output?

We tested the bound's content prediction directly on two backbones. We train three readers that differ only in the corpus feeding the table --- Wikipedia, code and STEM text --- and graft each into the same backbone at the same layer under identical prompts. If the memory carried corpus content, the three should answer differently.

== Under the Standard Recipe, They Do Not

On Qwen3.5-0.8B in fp32, over 1500 items per arm, the answer-label distribution, the leading-surface distribution and their joint agree *#emph[exactly]*: the empirical total-variation distance is $0.0000$, not merely within sampling error, and every pre-registered directional lexical prediction fails, with zero discordant pairs for code markers. Exact agreement is a stronger statement than a null result, so it is worth being precise about what it does and does not establish. It rules out any difference the sample could resolve, which for a categorical histogram over $n$ items is $1\/n$ --- one item in 1500. It does not establish that the underlying distributions are identical; it establishes that the corpora are interchangeable at the resolution this evaluation has, and the equivalence margin is reported rather than assumed (see §8). On Qwen3.5-4B in bf16 --- the backbone on which the format effect was first observed --- the same null holds over 600 items per arm: the largest cross-corpus spread is a single item, below the split-half floor of the arm that produced it. What the corpus changes is the part of the trigram statistic the backbone weights already encode, and under this recipe that part leaves no trace in the output.

== The Manipulation Reached the Injection Point

That the null is not a failed manipulation is established independently of the output. On 0.8B the three readers write measurably different vectors at the answer position (pairwise cosine similarity $0.73$ to $0.76$, relative $L_2$ distance $0.69$ to $0.78$, norms within a factor of $1.10$); on 4B the separation is larger, cosine $0.38$ to $0.49$. In a $3 times 3$ cross-evaluation each reader is best on its own corpus. The frozen tensors are bit-identical across arms and the adapter-norm ratios span only $0.886$ to $1.045$, well inside the threefold bound that would make the arms incomparable.

== The Effect Is Large, and Larger on the Bigger Backbone

The contrast with the zero-injection arm shows the channel is not merely inert, and the effect is far larger on the 4B backbone. There, zero injection makes the model emit chat scaffolding on 95.8% of items (mean 31.2 generated tokens), while injecting a reader trained on any of the three corpora suppresses it to 0.0% (mean 2.6 tokens), with not one of the 600 items sharing its first token with the zero-injection arm ($p = 1.6 times 10^(-173)$, McNemar exact). On 0.8B the same shift moves scaffolding from 19.5% to 0.0% and changes the first generated token on all 1500 items ($p = 1.3 times 10^(-88)$); the reader-disabled and no-reader arms are bit-identical, so the noise floor is exactly zero rather than merely small. The graft therefore does something large and reproducible --- it pushes the model off the chat template --- but that shift does not depend on what the table contains.

== Removing the Ceiling

This null has a scope, and testing it changes what the result means. On both backbones the injected arms saturate at 2.5 to 2.7 generated tokens, so "no format difference" could mean "no room for one". We therefore ran the regime that removes the saturation --- no answer-format SFT, where generated length returns to 11 to 21 tokens --- over all six arms on 600 items, so the same pre-registered rule applies without adjustment. The reader-disabled and no-reader arms are again bit-identical, giving a noise floor of exactly zero. The two halves of the rule then disagree, and the disagreement is the result. The coarse label taxonomy still reports a perturbation artifact, because the scaffold rate spans only $0.0100$ against a threshold of $0.0122$; every finer format descriptor reports content dependence. That division is why the co-primary descriptors were pre-registered alongside the taxonomy, on the stated grounds that the taxonomy can be insensitive to a real format shift. All four move outside each arm's own split-half sampling floor (leading-surface total variation $0.1433$ against a floor of $0.0237$; first-token $0.4983$ against $0.2176$), and both pre-registered directional lexical predictions hold: the code reader emits more code markers ($+0.0333$, one-sided $p = 3.9 times 10^(-4)$) and the Wikipedia reader emits far more prose markers ($+0.6933$, $p = 4.5 times 10^(-34)$), where under saturation both predictions had failed outright.

#figure(
  scope: "parent",
  image("figures/fig_content_null.svg", width: 100%),
  caption: [
    Removing the answer-length ceiling reveals content dependence that the
    saturated regime could not show. #emph[(a)] Cross-corpus total variation for
    four format descriptors, against each arm's own split-half floor (short
    ticks). Under the standard recipe three descriptors are pinned at exactly
    zero; with the ceiling removed all four move outside their floors.
    #emph[(b)] The pre-registered directional lexical predictions. Both were
    refuted under saturation and both hold without it. Math markers remain
    non-significant in both regimes, which is the expected result for a contrast
    the corpus does not separate.
  ],
)

The rule, its thresholds and the co-primary descriptors were fixed in the repository before the six-arm run, and the analysis script that scores the unsaturated regime is the same one that scored the saturated regime, applied unchanged. All three directional predictions were pre-registered and all three are reported, including the one that fails in both regimes (math markers, $p = 0.16$ unsaturated, $0.84$ saturated). We apply no multiple-comparison correction across the four co-primary descriptors, and instead report each against its own arm's split-half floor, which is the stricter comparison for a single arm; across the three directional predictions we report uncorrected $p$ values, and note that a Bonferroni correction at $0.05\/3$ would leave both positive results significant.

== The Layering Is What the Bound Predicts

One point of setup has to be explicit, because it scopes the claim. Our tables are general corpus tables --- built from Wikipedia, code and STEM text --- and not from the evaluation passages. A table therefore cannot recall a passage it was never built from, and the zero we report is not evidence that a table built from the passage would fail. What the bound forbids is narrower and does not depend on that: at the answer position the window holds the prompt's format rather than its content, so passage content cannot be supplied through this channel whatever the table contains. The experiment that would separate the two readings --- a table built from the evaluation passages, with a question whose answer lies outside the window --- is not in this paper, and it is the first thing we would add. We also note that TriviaQA is a weak probe of this: the frozen base model is already at $0.005$ exact match, so the task cannot detect an improvement even if one existed, and a task where the base model is measurably above chance is required.

So the correct statement is layered rather than absolute, and it is the layering the bound predicts. Suppressing the chat scaffold is content-independent --- all three readers suppress it, and they differ from each other by at most $0.01$. The rest of the output distribution is content-dependent, and that dependence *is* the trigram prior the bound allows the channel to carry: a reader trained on Wikipedia supplies Wikipedia-like surface statistics, one trained on code supplies code-like ones. What never appears, with or without the ceiling, is passage-conditioned recall --- the knowledge probes that require the passage return zero. The channel carries the corpus prior and nothing above it.

== How Much Prior Is Actually There

The bound permits the memory to carry a corpus's rare n-gram statistics, so the useful question is how much of that prior the frozen table actually holds and how much of it the read-out recovers. We probe the raw 2560-dimensional rows directly, on held-out windows, against explicit counts on the same corpus as a reference frame. Majority-class prediction gives a floor of $0.0537$ top-1. A count bigram reaches $0.2264$ and a count trigram $0.2535$. A ridge probe on the frozen rows reaches $0.1280$; an MLP reaches $0.1502$ at its best epoch and $0.1225$ at 24 epochs, so additional training makes it worse rather than better. Shuffled-row controls collapse below the majority floor at $0.0459$ and $0.0373$. The table therefore does hold a recoverable trigram prior --- well above the floor and destroyed by shuffling --- but the read-out recovers only about 59% of what explicit counts reach.

== The Read-Out Does Not Transmit It

That gap is not a small correction, and measuring the injected vector directly shows how large it is. Over 600 diverse prompts the vector the reader adds to the residual stream has an effective dimensionality of $1.001$: $99.93\%$ of its variance lies in a single direction, and two prompts are separated by about two degrees. The hidden state the reader is gated on sits at $1.480$, with $81.9\%$ in its leading direction. A generic linear map does not do this --- three random maps of the same shape leave the hidden state's ratio at $1.454$, $1.472$ and $1.485$ --- and replacing the retrieved rows with permuted ones, at the same checkpoint, moves the injected direction by only about five degrees.

#figure(
  scope: "parent",
  image("figures/fig_readout_collapse.svg", width: 100%),
  caption: [
    The read-out is close to a constant function of its input.
    #emph[(a)] Effective dimensionality of the injected vector, of the hidden
    state it is gated on, and of that hidden state after a random linear map of
    the same shape. Participation ratio $(sum lambda)^2 \/ sum lambda^2$ over
    $n = 600$ items; error bars on the random-map bar are one standard deviation
    across three maps, and bootstrap 95% intervals are $[1.0012, 1.0015]$ for
    $c_t$ and $[1.433, 1.535]$ for $h_t$. The random maps are the control: a
    generic map does not collapse the hidden state, so the collapse to $1.001$ is
    the read-out's. #emph[(b)] Share of variance in the leading direction.
    #emph[(c)] What actually moves the injected vector. Changing the prompt
    template moves it further than permuting the retrieved rows, and changing the
    item within a template moves it least of all.
  ],
)

The measurement is 600 items at the same layer and dtype as the arms. The participation ratio is $(sum lambda)^2 \/ sum lambda^2$ over the eigenvalues of the covariance of the 600 injected vectors, with 95% intervals from 2000 bootstrap resamples. It gives $1.0014$ $[1.0012, 1.0015]$ for $c_t$ against $1.480$ $[1.433, 1.535]$ for $h_t$, so the two are separated by far more than resampling error. The injected vector is also close to orthogonal to the state that produced it (mean cosine $0.04$) and its norm is $0.53 plus-minus 0.33$ against $1.33 plus-minus 0.06$ for that state --- a mean ratio of $0.40$ with a coefficient of variation of $0.62$. The collapse is therefore not a shut gate: the injection is comparable in magnitude to the state it modifies, and the gate opens onto the same direction every time.

The consequence is that the content-independence we measure under the standard recipe has two candidate explanations, and our experiments separate their consequences without separating the explanations themselves: either the window has nothing to give beyond the hidden state, or the read-out does not transmit what the window has. The bound is a theorem and holds in both cases, but the second explanation is a defect rather than a limit. The table probe is the reason we do not simply attribute the whole null to the bound: an explicit count model over the same trigram reaches $0.2535$ top-1, which is above the majority floor and therefore genuine information that a better read-out could in principle transmit.

== The Obvious Repair Does Not Pay

Nor does lengthening the key. Replacing the 4-gram address with 8-token, 16-token and longest-suffix memories at matched storage ties or loses at every budget and training size on both corpora, so the short window costs nothing measurable. That the window can be widened without gain, while the corpus the table is built from cannot be changed without gain either, is the same statement twice: the channel's output is fixed by what the window already determines.

= The Logit-Correction Interface Under the Same Bound

Four terms recur and we use them in one sense throughout. The *#emph[reader]* is the module that turns retrieved rows into a residual contribution; its *#emph[read-out]* is the arithmetic inside it --- the projections, the gate and the short convolution --- and is what the round-16 measurements probe. The *#emph[projector]* is the logit-level analogue that produces $(alpha_t, beta_t)$, and an *#emph[adapter]* is a parametric update to the backbone. Reader and projector are two interfaces to the memory; the adapter is not a memory interface at all.

The remainder of the paper concerns a second interface to the same resource. Instead of injecting the retrieved row into the residual stream, a small projector adds a per-token correction to the logits, $log p_(upright("fused"))(y) = log p_b(y) + alpha_t log p_m(y) + beta_t$, with a learned token policy deciding when the correction is active. We describe it here because reading it under the bound changes what its results are evidence for, and we report it in that light rather than as a separate contribution.

The distinction between the two interfaces is real but it is not an exemption. Logit-level fusion of an n-gram memory is *also* a function of the addressing window, so the bound applies exactly as it does to the residual graft. What the projector gets is a better interface, not a larger allowance: the permitted prior is the same in both cases. The graft loses it twice, once through the read-out and again through a residual-stream bottleneck in which a single vector must be disentangled by the backbone before it can affect any logit; the logit correction loses it once, and spends it in the basis where it is used.

== Method

The projector is a small MLP $f_theta(h_t, m_t) = (alpha_t, beta_t)$ over the last hidden state and a feature vector $m_t$ carrying the matched n-gram order, base and memory entropy, the density ratio, both top-1 probabilities, memory/base agreement, and a task one-hot. Its final layer is zero-initialized so the projection begins as the identity. A logistic policy on the same features predicts whether fusion will help at the current token and disables it otherwise, which is what keeps open-ended generation from degrading. Per-task calibration supplies a scale, bias and temperature, and only projector and policy parameters are trained; the backbone and the memory table stay frozen.

The policy deserves a word on its information-theoretic status, since it reads hidden states and memory features and could be mistaken for a way around the bound. It cannot be. The policy only sets $alpha_t$ and $beta_t$ to zero or leaves them alone, so the corrected distribution is either the projector's or the base model's; a gate that removes a contribution cannot add information that the contribution did not carry. Formally, $p_(upright("fused"))$ is a function of $h_t$ together with $e_t$ and the policy's own deterministic features, so the chain in §3.2 applies unchanged, and the policy's role is to *reduce* the exponent rather than to supply anything new.

== What It Buys

The gain is confined to low-entropy local continuation, which is the region the bound leaves open, and it scales with projector training data. On code continuation over 100 samples with five seeds the projector beats fixed calibration by $+0.1044$ nats (bootstrap 95% CI $[0.0251, 0.1866]$); at 10k samples the margin grows to $+0.2249$ with CI $[0.1436, 0.3255]$ and all five seeds are positive. Number-like continuations are more sensitive to calibration and often want a near-zero scale.

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
  caption: [Joint system mean answer log-probability, three seeds; higher is better. RAG carries knowledge, the adapter carries arithmetic and code-output, and PLE alone carries neither. Standard deviations across seeds are below $0.012$ in every cell.]
)

The joint system separates by resource rather than by strength. RAG supplies document-level evidence and is the only component that moves the knowledge column; the parametric adapter moves arithmetic and code-output; PLE alone moves nothing, and adding it to the best retrieval-plus-adapter combination changes nothing on knowledge (exactly $0.000$ nats) or code-output, while costing $0.123$ nats on arithmetic. The full system is therefore not the best system on any column, which is what a memory channel that cannot carry content should look like. On HumanEval the same picture holds from the other direction: greedy #text("pass@1") is $0.22$ for the base model and $0.10$ for BM25+PLE, the two pass sets overlap on a single problem out of 50, and under sampling over 10 problems at temperature $0.8$, top-$p$ $0.9$ the ordering reverses ($0.40$ to $0.70$). Neither direction is a reliable gain, and we read both as the same statement: a token-level n-gram prior sharpens the base model's local mode rather than adding capability.

Knowledge QA confirms the boundary. On 200 TriviaQA RC validation examples the frozen 0.8B model reaches exact match $0.005$, and PLE does not move it, because the required knowledge is not in local n-gram continuations. Training-free baselines behave the same way: NGM leaves the distribution essentially unchanged, and kNN-LM raises #text("hit@1") from $0.505$ to $0.540$ while worsening NLL from $2.633$ to $2.847$ on the same evaluation set. An external LLM judge agrees with the pass-based metrics (mean score $0.60$ for base against $0.20$ for BM25+PLE on HumanEval 50, with a parallel API rerun differing by up to $0.20$, so we treat it as secondary evidence). Full per-condition tables, the sensitivity sweep over n-gram order and memory-bank size, the algorithm listings, and the system figures are in the appendix.

= Analysis

== When Does the Channel Help

It helps when the true next token is highly predictable from local n-grams: code, identifiers, repeated structural patterns, name-like continuations. In those cases the memory distribution has low entropy and the correction can sharpen the model without adding noise. This is exactly the region the bound leaves open.

== When Does It Fail

It fails when the answer depends on world knowledge or long-range reasoning. The memory does not contain semantic knowledge, and adding it to the logits can produce confident but wrong continuations, which is why TriviaQA sits at $0.005$ and why unconditional fusion degrades open-ended generation without the token policy. For token-keyed memory this failure is not contingent but necessary: the bound shows the channel cannot carry what the addressing window does not already determine.

== The Bound Does Not Partition the Mechanisms the Way One Might Expect

Two mechanisms escape the bound, and only two: retrieval, which addresses by query rather than by a fixed token window, and parametric adapters, which are not a memory channel at all. PLE, RAG and adapters are therefore not three points on one axis of strength but three different interfaces to three different resources, and only the first of them is bounded in the way this paper describes. That reframing is testable against our own numbers and it holds: the projector's gain is confined to low-entropy local continuation and is absent on knowledge tasks, where the determining information lies outside the window.

== Why a Learned Projector Rather Than Fixed Calibration

Fixed calibration selects one scale and bias for all tokens. A learned projector varies the trust it places in memory by context, using a large positive scale on code and a near-zero scale on ambiguous open-ended text. That is what the improvement over fixed calibration measures, and it is a statement about the interface rather than about the memory's content, which is bounded in both cases.

= Method Audit

Two lessons from this project bear on how the results should be read.

The first is that a channel must be checked against its information-theoretic ceiling before its capacity is scaled, and a memory module must be evaluated by whether it uses memory content rather than by whether loss improves. Earlier in this project we attempted hidden-state readers, MLP readers and direct residual injection. These approaches often improved loss-like metrics but failed the real-versus-control test, and the bound explains why they had to: a reader translating a token-keyed row into the residual stream cannot carry context the window does not determine, so the real and control arms converge once the reader is well trained.

The second is less comfortable. A change was made, a claim was written, and the check that would have falsified the claim either did not exist or silently skipped, so the claim stood unexamined. @tbl-audit lists every such case we found, how it was finally caught, and which claim it put at risk. None of them was found by re-reading code. Each was found by making the failure detectable --- invariant assertions, regression tests that fail on the specific bug, a flag that converts a silent skip into a failure, and manifests recording a fingerprint of what was actually measured. In every case the durable fix was the detection mechanism rather than the correction, which is why we report the detection column and not only the fix.

#figure(
  scope: "parent",
  table(
    columns: (2.15in, 2.25in, 2.35in),
    align: (left, left, left),
    inset: 4pt,
    table.header([*Unrun or skipped check*], [*How it surfaced*], [*Claim it put at risk*]),
    [Row-identifier implementation differed from the proved-correct one],
      [Cross-repo golden against two independent implementations],
      [Every addressing claim],
    [A convolution believed absent was in the executed path],
      [Receptive-field measurement of the real layer (@sec:window)],
      [The twelve-token window],
    [A loop body was indented so a metric scored 4 items, not 1500],
      [Item-count fingerprint in the result manifest],
      [The 0.8B saturated null],
    [Four golden tests were skipping, not passing, on the compute box],
      [Environment flag converting a silent skip into a failure],
      [Every graft measurement],
    [The reader's forward had never been compared to the official mathematics],
      [Bit-exact golden at production geometry; they agree exactly],
      [Every graft conclusion],
    [A constant was copied from the wrong fixture ($10^(-5)$ against $10^(-6)$)],
      [The new golden test failed on first run],
      [Nothing --- caught before use],
    [A skip flag was applied to a 65 MB asset absent from the repository],
      [CI went red],
      [Nothing --- caught before use],
    [A scheduled shutdown interrupted a queue with two arms on the data disk],
      [Manifest whose arm count disagreed with the log],
      [The 4B arm count, recovered on restart],
    [A gold-NLL metric scored only the last batch],
      [$n$-items-scored against the requested count],
      [The table-probe numbers],
    [An offline reproduction averaged over the diagonal of a similarity matrix],
      [Unit test comparing it against the online statistic],
      [The read-out collapse figure]
  ),
  caption: [Every case found where a claim stood because its falsifying check was not run. The column that matters is the second: in each case the durable fix was making the failure detectable, and the detection is what we kept.],
) <tbl-audit>

The pattern is worth naming, because it recurred at the end of this work as well. Three of these were failures of the *pipeline's account of itself* rather than of the experiment: the arm count, the items scored, and the offline reproduction. In each case the model's numbers were right and our description of our own artifacts was wrong. That asymmetry --- the measurements held up, the bookkeeping did not --- is the reason this paper reports an equivalence margin rather than a claim of identical distributions, and why the number of seeds is stated per table rather than assumed.

A third observation is about failure that is not a mistake. Our final experiment pre-registered a prediction --- that the injected vector would be more nearly constant for a task whose addressing window is mostly a fixed instruction suffix than for tasks whose windows carry query text --- and the prediction was wrong. Within-task similarity came out at $0.9994$ and $0.9992$, a difference of $+0.0002$ that a permutation test over task labels declined to distinguish from noise ($p = 0.27$). What fixing the statistic and its refutation condition in advance bought us was not protection from a wrong result. It is that the wrong result is legible: because the rule could not be adjusted afterwards, the failure pointed straight at the near-constant read-out, which is a stronger finding than the prediction would have been. A full account of the audit, including the complete list of unrun checks, is in the project repository.

= Limitations

+ HumanEval is limited to 50 problems and #text("pass@k") to 10 problems with 3 samples, so the code results are directional rather than benchmark-scale.
+ TriviaQA exact match is very close to zero ($0.005$), so the paper reports a boundary rather than a positive result. LLM-judge scores show run-to-run variance and are secondary evidence.
+ The unsaturated regime that reveals the corpus's surface prior is measured over 600 items on one backbone under a different training recipe; it is a secondary regime and is reported as one. The saturated null it qualifies is measured on two backbones.
+ The content-independence we measure under the standard recipe is consistent with two mechanisms --- an addressing window that carries nothing beyond the hidden state, and a read-out that transmits almost nothing --- and we do not fully separate them. Direct measurement shows the second is real (effective dimensionality $1.001$; a five-degree response to permuted rows), and the table probe shows the window's trigram statistics sit above the majority floor, so the first is not the whole story. The bound holds either way, but the share of the design's failure attributable to it rather than to the read-out is left open, and the random-linear-map control we use to argue the collapse is the read-out's was chosen after seeing the result rather than pre-registered.
+ Two of the five per-seed artifacts behind the 10k projector result were not retained in the released bundle. The three that were reproduce the published table exactly, and the per-seed figure marks which bars come from which source; the remaining two rows are reported from the original run and are not independently reproducible from the release.
+ The bound's rarity prediction --- that gains concentrate on rare n-grams --- is stated but not tested, and we show above that it is not testable by corpus counting at the window scale with the resources available to us.
+ The reader-fidelity check covers the read-out, not the addressing: we verify that our reader reproduces the official mathematics bit-for-bit at production geometry, but the table itself is accessed through our own row-identifier implementation, which is separately pinned against two independent implementations.
+ The official Qwen3.8 model attaches its memory layer at 0-based layer 1, while our grafts inject at layer 2. The bound is a property of the reader's input-output map and is layer-independent, so the results above are unaffected, but the layer-1 configuration is untested rather than refuted.
+ Co-training the table with the backbone --- the configuration all three production systems use, and the one most likely to change the value of the channel within the bound --- is not tested here, at this scale.
+ The equivalence result in §4.1 is an exact agreement of empirical histograms over 1500 and 600 items, which rules out differences the sample can resolve --- one item in $n$ --- and does not establish that the underlying distributions are identical. The equivalence margin is the sampling resolution, not a confidence interval on a difference.
+ The conclusion is scoped to frozen-table grafts onto a frozen backbone, injected at a single mid-stack layer, with a table of production size but a target model of 0.8B to 4B. We do not test co-training, learned routing, adaptive layer placement, or the memory-to-compute ratios the production systems use, and the bound is a statement about the channel rather than a prediction that those configurations are worthless.
+ Table size, memory bandwidth and serving cost are not evaluated here. The table we access is 47.7 GiB on disk; inference cost is dominated by row fetch rather than by arithmetic, and every arm in this paper is read from a local disk with no latency budget. Deployment trade-offs are therefore not addressed.
+ HumanEval is a public benchmark and we cannot rule out contamination of the base model's training data. The base model solves 11 of 50 problems greedily; if some of those are memorised, the comparison against BM25+PLE is affected in an unknown direction, which is a further reason we treat the code results as directional.
+ All experiments ran on a single NVIDIA RTX 4090 with 24 GiB and an AutoDL instance whose data disk held the table; software versions, container definitions, seeds, per-arm manifests and the analysis scripts are recorded with each result file, and the released bundle carries the raw evaluation outputs rather than summaries.
+ The bound is elementary --- a data-processing inequality --- and we do not claim otherwise. Its value is in the window arithmetic it forces on a family that three production systems scaled without writing it down, and in the experiment that tests its content prediction.
+ Public projector and adapter weights are available on Hugging Face, but the upstream Qwen model weights themselves are not redistributed. CPU deployment throughput is about $2$ tok/s with the current unoptimized eager implementation.

= Conclusion

We asked what a token-keyed residual memory can carry, and the answer is set before any reader is chosen. The row identifier is a deterministic function of a short addressing window, so the memory's contribution to the future cannot exceed what that window already determines and the hidden state has not retained. In the Engram and Qwen3.8-Flash-Next designs that window is twelve tokens rather than the two an order-3 key suggests, because a kernel-4 dilation-3 convolution follows the gate; in DeepSeek-V4.1-Flash the convolution was removed and the window is exactly four.

The prediction this makes about content survives a direct test. Three readers that differ only in the corpus feeding the table write measurably different vectors and, under the standard recipe, produce output distributions that are identical to within a single item out of 600. Removing the answer-length ceiling does not break the bound but completes it: the corpus's surface statistics then appear in the output, which is the trigram prior the channel is permitted to carry, while passage-conditioned recall remains at zero. The channel is a channel for a corpus's surface prior and not for knowledge, and which of the two it is, is fixed by the addressing window.

The measurements that size what remains point in different directions. The read-out recovers about 59% of the trigram top-1 that explicit counts reach, and more training makes it worse, so the loss is architectural rather than a matter of budget; measured directly, the vector it injects is nearly a constant, with an effective dimensionality of $1.001$ against $1.480$ for the hidden state it reads. Because the table's trigram statistics sit well above the majority floor, this gap is a target rather than a verdict: the design fails for the bound's reason and for a read-out defect at the same time, and only the first of those is unfixable at a twelve-token window. Lengthening the key does not pay at matched storage on any corpus or budget we tested. What is left is therefore the complement --- retrieval, which addresses by query, and parametric adapters, which are not a memory channel at all --- together with the read-out itself.

= Acknowledgement

We thank the open-source community for the PLE/Engram, Qwen, and related memory infrastructure used in this study. This work was carried out as an independent low-resource research project.

= Data Availability

All source code, evaluation scripts, container definitions, evaluation cards, and checksums are publicly available in the project repository and in the Hugging Face artifact repository. The released artifact bundle includes PLE projector checkpoints, Purified OPSD MoRA adapters, projector datasets, configurations, and the raw evaluation result files, including the per-item injected vectors behind the read-out measurement. The PLE memory tables can be rebuilt from the published corpus construction scripts, and upstream Qwen model weights are not redistributed.
