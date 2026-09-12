"""The load-bearing numbers of round 164: window geometry and template suffix.

Why this file exists
--------------------
Round 164 rests on two numbers that everything else is derived from:

1. **The addressing window is 12 tokens** at the answer position.  This is not
   read off a comment -- it is ``short_conv_state_len + ngram_size``, where
   ``short_conv_state_len = (ple_conv_kernel_size - 1) * ngram_size``.  The
   short conv at ``t`` sees ``gated_value`` at ``t-9 .. t``, and each of those
   sees ``ngram_size = 3`` tokens, so the union is ``t-11 .. t``.

2. **The fixed template suffix occupies 10 of those 12 slots for BoolQ**
   (``"\\nAnswer with one word, Yes or No:"``) and 3 for the other tasks
   (``"\\nAnswer:"``).  So BoolQ's injected vector is 83% a property of the
   *template*, not the item.

Both are the kind of "fact about our own pipeline" that this repo has been
wrong about before (see ``docs/round-163-reader-forward-golden.md`` §5.1 --
five of that session's corrections were about our own artifacts and none about
the model).  So they get a check rather than a paragraph.

The geometry half runs in CI from the committed golden config.  The tokenizer
half needs ``data/models/Qwen3.5-0.8B``, which is gitignored, so it skips there
-- following the ``_skip_external`` convention: a missing gitignored asset is
never allowed to fail a run, because that only teaches people to ignore red.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_window_composition.py"
GOLDEN_CONFIG = ROOT / "tests" / "golden" / "qwen38_flash_next_text_config.json"
TOKENIZER_DIR = ROOT / "data" / "models" / "Qwen3.5-0.8B"

# The values the round-164 document and README assert.  Pinned here so that a
# template edit or a tokenizer swap cannot silently invalidate them.
EXPECTED_SUFFIX_TOKENS = {"boolq": 10, "nq": 3, "triviaqa": 3}
EXPECTED_WINDOW = 12


def _load_script():
    spec = importlib.util.spec_from_file_location("_window_composition", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _golden() -> dict:
    if not GOLDEN_CONFIG.exists():
        pytest.fail(f"committed golden config missing: {GOLDEN_CONFIG}")
    return json.loads(GOLDEN_CONFIG.read_text())


def _skip_external(path: Path, what: str) -> None:
    if not path.exists():
        pytest.skip(f"{what} not found: {path}")


def _tokenizer():
    pytest.importorskip("transformers")
    _skip_external(TOKENIZER_DIR, "Qwen3.5-0.8B tokenizer")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


# --------------------------------------------------------------------------
# Geometry: runs in CI, no external assets.
# --------------------------------------------------------------------------


def test_short_conv_state_len_is_nine() -> None:
    """``short_conv_state_len = (kernel - 1) * ngram_size`` from the official config.

    Mirrors ``official_ple_snapshot.py``::

        conv_dilation = config.ngram_size
        self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation
    """
    cfg = _golden()
    kernel = cfg["ple_conv_kernel_size"]
    ngram = cfg["ngram_size"]
    assert (kernel, ngram) == (4, 3)
    assert (kernel - 1) * ngram == 9


def test_window_is_state_len_plus_ngram_size() -> None:
    """The window the script uses must equal the code-derived receptive field."""
    module = _load_script()
    cfg = _golden()
    derived = (cfg["ple_conv_kernel_size"] - 1) * cfg["ngram_size"] + cfg["ngram_size"]
    assert derived == EXPECTED_WINDOW
    assert module.WINDOW == derived, (
        f"scripts/analyze_window_composition.py says WINDOW={module.WINDOW} "
        f"but the official config derives {derived}"
    )
    assert module.NGRAM_SIZE == cfg["ngram_size"]


# --------------------------------------------------------------------------
# Template suffix: needs the gitignored tokenizer, skips elsewhere.
# --------------------------------------------------------------------------


def test_template_suffix_token_counts() -> None:
    """The suffix is 3 tokens normally and 10 for BoolQ -- the round's premise."""
    module = _load_script()
    tok = _tokenizer()
    got = {}
    for task, template in (
        ("nq", module.PROMPT),
        ("triviaqa", module.PROMPT),
        ("boolq", module.BOOLQ_PROMPT),
    ):
        got[task] = len(module._suffix_ids(tok, template))
    assert got == EXPECTED_SUFFIX_TOKENS, (
        f"template suffix lengths changed: {got} != {EXPECTED_SUFFIX_TOKENS}; "
        f"the round-164 numbers (and README) are now stale"
    )


def test_suffix_ids_decode_back_to_the_template_tail() -> None:
    """Guard against the suffix helper silently swallowing part of the prompt.

    ``_suffix_ids`` splits on a marker token; if the template ever gains a
    second placeholder, or the marker starts appearing in the text, the split
    would move and every downstream ratio would be wrong without erroring.
    """
    module = _load_script()
    tok = _tokenizer()
    for template in (module.PROMPT, module.BOOLQ_PROMPT):
        marker = template.format(question="\x00")
        head, _, tail = marker.partition("\x00")
        assert head == "Question: " and "\x00" not in tail
        assert tok.decode(module._suffix_ids(tok, template)) == tail


def test_boolq_window_is_template_dominated() -> None:
    """The headline: item-specific slots are 2/12 for BoolQ and 9/12 otherwise.

    This is the number the pre-registered Part B prediction is derived from, so
    it is asserted rather than described.
    """
    module = _load_script()
    tok = _tokenizer()
    slots = {
        task: module.WINDOW - len(module._suffix_ids(tok, template))
        for task, template in (("boolq", module.BOOLQ_PROMPT), ("nq", module.PROMPT))
    }
    assert slots == {"boolq": 2, "nq": 9}
    # Anti-vacuity: the suffix must not exceed the window, or `slots` would
    # clamp to zero and the comparison below would be trivially true.
    assert min(slots.values()) > 0
    assert slots["boolq"] < slots["nq"]
