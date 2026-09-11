"""Golden: ``OfficialSourceQwenReader`` vs the official PLE read-out.

Why this file exists (round 163)
--------------------------------
Every graft conclusion in this repo -- "the reader cannot inject content",
"the read-out recovers ~59%", the whole falsification set -- is a statement
about ``OfficialSourceQwenReader``.  Before this file, that class was only ever
*instantiated* (``test_reader_registry.py``); its ``forward`` was never compared
to the official mathematics it claims to reuse.

The fixtures that mention the official layer do not cover it:

* ``test_official_ple_reference.py`` checks that ``official_ple_snapshot.py``
  reproduces its own generator, at ``hc_count=1, kernel=2, dilation=2``.
* ``test_ple_forward_golden.py`` compares *engram-peft's* ``EngramLayer`` to
  ``ple_reference``, again at ``hc_mult=1``.
* ``test_phase_b_official_loader.py`` covers the disk-table adapter.

None of them runs the multi-branch (``hc_count=4``), dilated (kernel 4,
dilation 3) read-out that the graft actually uses.  So "the design cannot carry
content" had evidence, while "our implementation of the design is faithful" had
none.  This file closes that gap.

What is compared
----------------
The *shared read-out*: everything from ``(e_t, h)`` to the returned tensor.
``e_t`` is supplied by the caller in production (``model._current_ple_e_t``) and
by the frozen table in training, so hashing/table lookup is deliberately out of
scope -- the cross-repo goldens pin that separately.  Incremental decoding is
also out of scope: our reader never sees ``past_key_values``.

The two adaptations are *ours*, not the official code's, and the tests below
pin them explicitly rather than hiding them:

* ``query_bridge`` -- the official layer applies ``norm_query`` to an already
  hyper-connection-expanded hidden state of width ``hc * hidden_size``.  A
  single-stream target model has no such tensor, so we insert a learned linear
  map from the target hidden into that space.
* branch sum + ``out_proj`` -- the official layer returns the full
  ``[B, T, hc * hidden_size]`` stream; we sum the ``hc`` branches and project to
  the target width.

With both adapters replaced by ``Identity`` the two implementations must agree
exactly, and that is the assertion that gives the rest of the project its
meaning.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from qwen35_ple.official_ple_snapshot import Qwen4ExpTextPLELayer
from qwen35_ple.reader import OfficialSourceQwenReader

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_READER_PATH = ROOT / "data" / "official_ple_reader.pt"

CHECKPOINT_PREFIX = "model.language_model.layers.1.ple."

# Production read-out geometry, taken from the official
# ``Qwen3.8-Flash-Next`` ``config.json`` (``text_config``) and cross-checked
# against ``data/official_ple_reader.pt``:
#   hidden_size 2560, hc_count 4 -> key_proj (10240, 2560)
#   ple_embed_dim 2560           -> value_proj (2560, 2560)
#   ple_conv_kernel_size 4       -> conv1d (10240, 1, 4), depthwise
#   ngram_size 3                 -> conv_dilation 3, ngram_heads 16, head_dim 160
#   rms_norm_eps 1e-06           -> the reader's default ``eps``
# ``dilation`` is the one read-out parameter absent from the checkpoint: the
# official layer sets ``conv_dilation = config.ngram_size``, giving
# ``short_conv_state_len = (4-1)*3 = 9``.
PROD_HIDDEN = 2560
PROD_HC = 4
PROD_KERNEL = 4
PROD_DILATION = 3
PROD_EMBED_DIM = 2560
PROD_HEADS_PER_NGRAM = 8
PROD_NGRAM_SIZE = 3
PROD_RMS_NORM_EPS = 1e-6

# The official ``config.json`` is 49 GB away from CI, so a verifiable excerpt is
# committed at ``tests/golden/qwen38_flash_next_text_config.json`` (it carries the
# sha256 of the original file).  ``QWEN35_OFFICIAL_CONFIG``, when set to the real
# config.json, additionally checks that the excerpt still matches the source.
OFFICIAL_CONFIG_EXCERPT = (
    ROOT / "tests" / "golden" / "qwen38_flash_next_text_config.json"
)
OFFICIAL_CONFIG_ENV = os.environ.get("QWEN35_OFFICIAL_CONFIG", "")

# The six tensors the reader reuses from the official layer, by their names in
# our module.  Checkpoint keys are ``CHECKPOINT_PREFIX + name``.
SHARED_TENSORS = (
    "key_proj.weight",
    "value_proj.weight",
    "norm_key.weight",
    "norm_query.weight",
    "norm_conv.weight",
    "conv1d.weight",
)


def _config(
    *,
    hidden_size: int,
    hc_count: int,
    ple_embed_dim: int,
    heads_per_ngram: int,
    ngram_size: int,
    ple_conv_kernel_size: int,
    ngram_vocab_size_base: int = 100,
    vocab_size: int = 1000,
    eos_token_id: int = 2,
) -> SimpleNamespace:
    """Config with the official field names ``Qwen4ExpTextPLELayer`` reads.

    ``ngram_vocab_size_base`` is kept small on purpose: it sizes only the
    *embedding table*, which both the checkpoint and these tests exclude.  Every
    tensor under test (``key_proj``/``value_proj``/norms/``conv1d``) is sized by
    ``hidden_size``/``hc_count``/``ple_embed_dim`` alone.
    """
    return SimpleNamespace(
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        vocab_size=vocab_size,
        ngram_vocab_size_base=ngram_vocab_size_base,
        seed=0,
        eos_token_id=eos_token_id,
        make_ngram_vocab_size_divisible_by=128,
        hidden_size=hidden_size,
        hc_count=hc_count,
        ple_embed_dim=ple_embed_dim,
        ple_conv_kernel_size=ple_conv_kernel_size,
        rms_norm_eps=PROD_RMS_NORM_EPS,
    )


def _small_config() -> SimpleNamespace:
    return _config(
        hidden_size=8,
        hc_count=4,
        ple_embed_dim=16,
        heads_per_ngram=4,
        ngram_size=3,
        ple_conv_kernel_size=4,
    )


def _production_config() -> SimpleNamespace:
    return _config(
        hidden_size=PROD_HIDDEN,
        hc_count=PROD_HC,
        ple_embed_dim=PROD_EMBED_DIM,
        heads_per_ngram=PROD_HEADS_PER_NGRAM,
        ngram_size=PROD_NGRAM_SIZE,
        ple_conv_kernel_size=PROD_KERNEL,
    )


class _FixedEmbedding(torch.nn.Module):
    """Stand-in for ``Qwen4ExpTextNGramEmbedding`` returning a fixed ``e_t``.

    Controlling ``e_t`` removes the table lookup from the comparison so that a
    divergence cannot be blamed on (or masked by) the embedding path.  The
    signature matches the real module because the official ``forward`` calls it
    as ``self.ple_embedding(input_ids, past_key_values)``.
    """

    def __init__(self, e_t: torch.Tensor) -> None:
        super().__init__()
        self.e_t = e_t

    def forward(self, input_ids: torch.Tensor, past_key_values=None) -> torch.Tensor:
        return self.e_t


def _load_shared_into_official(
    official: Qwen4ExpTextPLELayer, state: dict[str, torch.Tensor]
) -> None:
    """Copy the six shared tensors from a checkpoint dict into the official layer."""
    modules = {
        "key_proj.weight": official.key_proj,
        "value_proj.weight": official.value_proj,
        "norm_key.weight": official.norm_key,
        "norm_query.weight": official.norm_query,
        "norm_conv.weight": official.norm_conv,
        "conv1d.weight": official.conv1d,
    }
    with torch.no_grad():
        for name, module in modules.items():
            src = state[name] if name in state else state[CHECKPOINT_PREFIX + name]
            module.weight.copy_(src.to(module.weight.dtype))


def _make_pair(
    config: SimpleNamespace,
    e_t: torch.Tensor,
    *,
    d_target: int,
    identity_adapters: bool,
    source_state: dict[str, torch.Tensor] | None = None,
) -> tuple[Qwen4ExpTextPLELayer, OfficialSourceQwenReader]:
    """Build the official layer and our reader over the *same* read-out weights.

    ``d_target`` is the target model's hidden width.  For the identity-adapter
    comparison it must be ``hc_count * hidden_size`` so that the bridge, replaced
    by ``Identity``, hands the official layer the expanded stream it expects.
    With ``source_state=None`` both sides get the official layer's own random
    init, so the test compares *code*; with a checkpoint both sides get the real
    frozen Qwen3.8 tensors, so the test compares the numbers the graft ran.
    """
    torch.manual_seed(1234)
    official = Qwen4ExpTextPLELayer(config, layer_idx=1, ple_layer_index=0)
    official.eval()
    if source_state is not None:
        _load_shared_into_official(official, source_state)
    official.ple_embedding = _FixedEmbedding(e_t)

    reader = OfficialSourceQwenReader(
        d_target=d_target,
        d_source=config.hidden_size,
        d_mem=config.ple_embed_dim,
        hc=config.hc_count,
        kernel_size=config.ple_conv_kernel_size,
        dilation=config.ngram_size,
        source_state=source_state,
        zero_init_out=False,
        eps=config.rms_norm_eps,
    )
    reader.eval()

    if source_state is None:
        _copy_shared_to_reader(official, reader)

    if identity_adapters:
        # Remove the two adaptations that are ours rather than the official
        # code's, leaving the shared read-out on both sides.
        reader.query_bridge = torch.nn.Identity()
        reader.out_proj = torch.nn.Identity()
    return official, reader


def _copy_shared_to_reader(
    official: Qwen4ExpTextPLELayer, reader: OfficialSourceQwenReader
) -> None:
    """Copy the six shared tensors from the official layer into the reader."""
    pairs = (
        (official.key_proj, reader.key_proj),
        (official.value_proj, reader.value_proj),
        (official.norm_key, reader.norm_key),
        (official.norm_query, reader.norm_query),
        (official.norm_conv, reader.norm_conv),
        (official.conv1d, reader.conv1d),
    )
    with torch.no_grad():
        for src, dst in pairs:
            dst.weight.copy_(src.weight)


def _inputs(config: SimpleNamespace, *, batch: int, seq_len: int):
    torch.manual_seed(0)
    e_t = torch.randn(batch, seq_len, config.ple_embed_dim)
    # ``h`` is what the official layer calls ``hidden_states``: the
    # hyper-connection-expanded stream, width hc * hidden_size.
    h = torch.randn(batch, seq_len, config.hc_count * config.hidden_size)
    return e_t, h


def _official_forward(
    official: Qwen4ExpTextPLELayer, h: torch.Tensor
) -> torch.Tensor:
    """Call the official layer in its own argument order (hidden_states first)."""
    batch, seq_len = h.shape[0], h.shape[1]
    input_ids = torch.zeros(batch, seq_len, dtype=torch.long)
    with torch.no_grad():
        return official(h, input_ids, past_key_values=None)


def _assert_shared_readout_matches(
    config: SimpleNamespace, *, batch: int, seq_len: int
) -> None:
    """Our reader (identity adapters) must equal the official read-out, summed over hc."""
    e_t, h = _inputs(config, batch=batch, seq_len=seq_len)
    official, reader = _make_pair(
        config, e_t, d_target=h.shape[-1], identity_adapters=True
    )
    expected = _official_forward(official, h)
    expected = expected.view(batch, seq_len, config.hc_count, config.hidden_size)
    expected = expected.sum(dim=2)

    with torch.no_grad():
        actual = reader(h, e_t)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------------
# 1. The shared read-out, at a geometry small enough to debug.
# --------------------------------------------------------------------------


def test_shared_readout_matches_official_small() -> None:
    _assert_shared_readout_matches(_small_config(), batch=2, seq_len=17)


def test_shared_readout_matches_official_single_branch() -> None:
    """``hc_count=1`` degenerates to the official tensor with no branch sum."""
    config = _config(
        hidden_size=8,
        hc_count=1,
        ple_embed_dim=16,
        heads_per_ngram=4,
        ngram_size=3,
        ple_conv_kernel_size=4,
    )
    _assert_shared_readout_matches(config, batch=1, seq_len=5)


def test_shared_readout_matches_official_short_sequence() -> None:
    """Sequences shorter than the receptive field (pad = (4-1)*3 = 9) still line up."""
    _assert_shared_readout_matches(_small_config(), batch=1, seq_len=3)


# --------------------------------------------------------------------------
# 2. The production read-out geometry -- the one the graft actually ran.
# --------------------------------------------------------------------------


def test_shared_readout_matches_official_production_geometry() -> None:
    """hc=4, kernel=4, dilation=3, grouped norms over 2560.  No table needed."""
    _assert_shared_readout_matches(_production_config(), batch=1, seq_len=33)


# --------------------------------------------------------------------------
# 3. The adaptations are exactly the ones documented, and nothing else.
# --------------------------------------------------------------------------


def test_branch_sum_plus_out_proj_is_the_whole_adaptation() -> None:
    """With the real bridge and out_proj, ours == out_proj(sum_hc(official)).

    Here ``d_target`` is the *target* width (8), not the expanded width, so the
    bridge is exercised for real: it is the only thing that can turn an 8-wide
    target hidden into the 32-wide expanded stream the official layer wants.
    """
    config = _small_config()
    d_target = config.hidden_size
    batch, seq_len = 2, 13
    e_t, _ = _inputs(config, batch=batch, seq_len=seq_len)
    torch.manual_seed(7)
    h_target = torch.randn(batch, seq_len, d_target)

    official, reader = _make_pair(
        config, e_t, d_target=d_target, identity_adapters=False
    )

    # Feed the official layer the bridge's output, so every tensor the reader
    # owns is exercised; only the final reduction is assumed.
    with torch.no_grad():
        bridge_out = reader.query_bridge(h_target)
        assert bridge_out.shape == (batch, seq_len, config.hc_count * config.hidden_size)
        expected = _official_forward(official, bridge_out)
        expected = expected.view(batch, seq_len, config.hc_count, config.hidden_size)
        expected = reader.out_proj(expected.sum(dim=2))
        actual = reader(h_target, e_t)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_reader_defaults_match_the_official_config() -> None:
    """The constructor defaults must stay at the official values.

    A silent change here (e.g. dilation) would alter the receptive field of
    every graft result while leaving the other tests passing, because they pass
    the kernel/dilation explicitly.
    """
    reader = OfficialSourceQwenReader(d_target=1024, zero_init_out=False)
    assert reader.d_source == PROD_HIDDEN
    assert reader.d_mem == PROD_HIDDEN
    assert reader.hc == PROD_HC
    assert reader.kernel_size == PROD_KERNEL
    assert reader.dilation == PROD_DILATION
    assert reader.src_dim == PROD_HC * PROD_HIDDEN


def test_production_constants_match_official_config() -> None:
    """Pin the hard-coded constants to the official ``text_config``.

    The constants above are the load-bearing assumptions of every geometry
    claim in this repo.  Reading them from the real checkpoint config is the
    difference between "we believe dilation is 3" and "the config says so".
    ``ngram_size`` is the one that is *not* recoverable from the weight shapes
    (``conv1d.weight`` is ``(10240, 1, 4)`` for any dilation), so it is the one
    worth checking most: it fixes both the conv dilation and the head count.
    """
    excerpt = json.loads(OFFICIAL_CONFIG_EXCERPT.read_text())
    assert excerpt["hidden_size"] == PROD_HIDDEN
    assert excerpt["hc_count"] == PROD_HC
    assert excerpt["ple_embed_dim"] == PROD_EMBED_DIM
    assert excerpt["ple_conv_kernel_size"] == PROD_KERNEL
    assert excerpt["heads_per_ngram"] == PROD_HEADS_PER_NGRAM
    assert excerpt["ngram_size"] == PROD_NGRAM_SIZE
    assert excerpt["ngram_size"] == PROD_DILATION
    assert excerpt["rms_norm_eps"] == PROD_RMS_NORM_EPS
    assert (excerpt["ngram_size"] - 1) * excerpt["heads_per_ngram"] == 16


def test_official_config_excerpt_matches_the_real_config() -> None:
    """When the real config.json is reachable, verify the excerpt against it.

    Deliberately gated on ``QWEN35_OFFICIAL_CONFIG`` alone, *not* on
    ``QWEN35_REQUIRE_GOLDEN``: the latter means "the goldens committed to this
    repo must be present", and the 49 GB checkpoint is not one of them.  Making
    this test fail under that flag would turn a missing external volume into a
    red CI run.
    """
    if not OFFICIAL_CONFIG_ENV:
        pytest.skip("set QWEN35_OFFICIAL_CONFIG to the real config.json to check it")
    raw = Path(OFFICIAL_CONFIG_ENV).read_bytes()
    excerpt = json.loads(OFFICIAL_CONFIG_EXCERPT.read_text())
    assert hashlib.sha256(raw).hexdigest() == excerpt["_provenance"]["source_sha256"]
    real = json.loads(raw)["text_config"]
    for key, value in excerpt.items():
        if key.startswith("_"):
            continue
        assert real[key] == value, f"excerpt disagrees with config.json on {key}"


# --------------------------------------------------------------------------
# 4. Real official weights, production geometry.
# --------------------------------------------------------------------------


def _require_or_skip(message: str) -> None:
    if os.environ.get("QWEN35_REQUIRE_GOLDEN") == "1":
        pytest.fail(message)
    pytest.skip(message)


def _load_official_state() -> dict[str, torch.Tensor]:
    if not OFFICIAL_READER_PATH.exists():
        _require_or_skip(f"official reader checkpoint not found: {OFFICIAL_READER_PATH}")
    return torch.load(OFFICIAL_READER_PATH, map_location="cpu", weights_only=False)


def test_official_checkpoint_matches_reader_shapes() -> None:
    state = _load_official_state()
    reader = OfficialSourceQwenReader(
        d_target=1024, source_state=state, zero_init_out=False
    )
    assert reader.key_proj.weight.shape == state[CHECKPOINT_PREFIX + "key_proj.weight"].shape
    assert reader.key_proj.weight.shape == (PROD_HC * PROD_HIDDEN, PROD_HIDDEN)
    assert reader.value_proj.weight.shape == state[CHECKPOINT_PREFIX + "value_proj.weight"].shape
    assert reader.value_proj.weight.shape == (PROD_HIDDEN, PROD_HIDDEN)
    assert reader.conv1d.weight.shape == state[CHECKPOINT_PREFIX + "conv1d.weight"].shape
    assert reader.conv1d.weight.shape == (PROD_HC * PROD_HIDDEN, 1, PROD_KERNEL)
    # A depthwise conv: exactly one tap per channel.  A differently grouped conv
    # would have the same *shape* only in degenerate cases, so pin the group count.
    assert reader.conv1d.groups == reader.src_dim


def test_shared_readout_matches_official_with_real_weights() -> None:
    """The decisive run: production geometry, the real frozen Qwen3.8 tensors."""
    state = _load_official_state()

    config = _production_config()
    batch, seq_len = 1, 33
    e_t, h = _inputs(config, batch=batch, seq_len=seq_len)
    official, reader = _make_pair(
        config,
        e_t,
        d_target=h.shape[-1],
        identity_adapters=True,
        source_state=state,
    )

    expected_flat = _official_forward(official, h)
    expected = expected_flat.view(batch, seq_len, config.hc_count, config.hidden_size)
    expected = expected.sum(dim=2)
    with torch.no_grad():
        actual = reader(h, e_t)

    assert torch.isfinite(actual).all(), "real-weight output is not finite"
    assert actual.abs().max() > 0, "real-weight output is identically zero"
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------------
# 5. Anti-vacuity: prove the comparison can fail.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tensor_name", SHARED_TENSORS)
def test_every_shared_tensor_is_actually_exercised(tensor_name: str) -> None:
    """Perturbing any shared tensor must change the output.

    Without this, a comparison that silently dropped a term (or compared two
    zeros) would pass.  Each of the six tensors the reader claims to reuse is
    mutated in turn.
    """
    config = _small_config()
    batch, seq_len = 1, 11
    e_t, h = _inputs(config, batch=batch, seq_len=seq_len)
    _, reader = _make_pair(
        config, e_t, d_target=h.shape[-1], identity_adapters=True
    )

    with torch.no_grad():
        baseline = reader(h, e_t)
        target = reader
        for part in tensor_name.split("."):
            target = getattr(target, part)
        target.add_(0.5)
        mutated = reader(h, e_t)

    assert not torch.allclose(baseline, mutated), (
        f"perturbing {tensor_name} did not change the reader output; "
        "it is not actually part of the compared read-out"
    )
