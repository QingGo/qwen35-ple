"""Round 167 Stage 0.3 (TD-3a): the read-out configuration must be perturbable.

The debt was that ``reader_config_from_args`` returned the *literal* ``True`` for
``zero_init_out``.  That made the read-out initialisation untestable without
editing library code -- the configuration space was not merely unexplored, it
was unreachable.  These tests pin both halves of the fix:

* the new knob is reachable, and
* the default is still ``True``, so no existing queue or checkpoint changes
  behaviour (the integration contract only permits additive changes).
"""

from __future__ import annotations

from types import SimpleNamespace

from qwen35_ple.reader_registry import (
    OFFICIAL_SOURCE_QWEN_V1,
    reader_config_from_args,
)


def test_default_is_unchanged_when_no_args_are_supplied() -> None:
    """Historical behaviour must survive: absent arg -> zero-init ON."""
    cfg = reader_config_from_args(SimpleNamespace(), 1024, OFFICIAL_SOURCE_QWEN_V1)
    assert cfg["zero_init_out"] is True


def test_default_is_unchanged_when_arg_is_none() -> None:
    cfg = reader_config_from_args(
        SimpleNamespace(zero_init_out=None), 1024, OFFICIAL_SOURCE_QWEN_V1
    )
    assert cfg["zero_init_out"] is True


def test_zero_init_can_be_turned_off() -> None:
    cfg = reader_config_from_args(
        SimpleNamespace(zero_init_out=False), 1024, OFFICIAL_SOURCE_QWEN_V1
    )
    assert cfg["zero_init_out"] is False


def test_zero_init_can_be_turned_on_explicitly() -> None:
    cfg = reader_config_from_args(
        SimpleNamespace(zero_init_out=True), 1024, OFFICIAL_SOURCE_QWEN_V1
    )
    assert cfg["zero_init_out"] is True


def test_readout_depth_knobs_flow_through() -> None:
    """``out_mlp``/``out_hidden`` are the read-out depth and width knobs."""
    cfg = reader_config_from_args(
        SimpleNamespace(out_mlp=True, out_hidden=512), 2560, OFFICIAL_SOURCE_QWEN_V1
    )
    assert cfg["out_mlp"] is True
    assert cfg["out_hidden"] == 512


def test_readout_depth_defaults_to_a_single_linear() -> None:
    cfg = reader_config_from_args(SimpleNamespace(), 1024, OFFICIAL_SOURCE_QWEN_V1)
    assert cfg["out_mlp"] is False
    assert cfg["out_hidden"] is None


def test_official_source_geometry_is_untouched() -> None:
    """The knobs must not have disturbed the frozen official geometry."""
    cfg = reader_config_from_args(
        SimpleNamespace(zero_init_out=False), 1024, OFFICIAL_SOURCE_QWEN_V1
    )
    assert cfg["d_source"] == 2560
    assert cfg["d_mem"] == 2560
    assert cfg["hc"] == 4
    assert cfg["kernel_size"] == 4
    assert cfg["dilation"] == 3
    assert cfg["freeze_source"] is True
