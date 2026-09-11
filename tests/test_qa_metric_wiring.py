"""Static wiring guards for the round-156 metric fixes in ``run_phase0.py``.

The round-156 patch adds two keyword arguments at CLI call sites
(``space_variant`` for the spaced gold-NLL reading) and relies on
``_qa_exact_match`` computing the new token-level EM.  Both are easy to wire
into the *wrong* call: ``_qa_exact_match`` and ``_qa_gold_nll`` share the
``head_mask`` keyword and sit a few lines apart, so a mechanical patch inserts
``space_variant`` into the former and the failure only shows up at runtime on
the GPU box, hours later.

These tests parse the file instead of importing it (importing needs torch) and
assert the arguments land on the intended functions only.
"""

from __future__ import annotations

import ast
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_phase0.py"


def _parse() -> ast.Module:
    return ast.parse(SCRIPT.read_text(encoding="utf-8"))


def _calls(tree: ast.Module, name: str) -> list[ast.Call]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            is_match = (isinstance(func, ast.Name) and func.id == name) or (
                isinstance(func, ast.Attribute) and func.attr == name
            )
            if is_match:
                found.append(node)
    return found


def _keywords(call: ast.Call) -> set[str]:
    return {kw.arg for kw in call.keywords if kw.arg}


def test_space_variant_is_wired_to_gold_nll_only() -> None:
    tree = _parse()
    gold_calls = _calls(tree, "_qa_gold_nll")
    assert gold_calls, "no _qa_gold_nll call sites found"
    for call in gold_calls:
        assert "space_variant" in _keywords(call), (
            f"line {call.lineno}: _qa_gold_nll call is missing space_variant"
        )

    for call in _calls(tree, "_qa_exact_match"):
        assert "space_variant" not in _keywords(call), (
            f"line {call.lineno}: space_variant passed to _qa_exact_match, which "
            "does not accept it"
        )


def test_gold_nll_signature_accepts_space_variant() -> None:
    tree = _parse()
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    target = next((f for f in funcs if f.name == "_qa_gold_nll"), None)
    assert target is not None
    args = [a.arg for a in target.args.args]
    assert "space_variant" in args, args


def test_exact_match_reports_token_level_em_and_format_fingerprint() -> None:
    tree = _parse()
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "_token_level_hit" in names, "token-level EM helper is not called"
    assert "_format_fingerprint" in names, "format fingerprint is not called"
    source = SCRIPT.read_text(encoding="utf-8")
    assert "qa_em_token_mean" in source
    assert "qa_fmt_scaffold_rate" in source
    assert "qa_fmt_empty_rate" in source


def test_token_level_hit_does_not_match_across_a_boundary() -> None:
    """`Nouser` must not satisfy the answer `no` at token level."""
    tree = _parse()
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    helper = funcs.get("_token_level_hit")
    assert helper is not None
    # The helper must compare whole decoded windows for equality, not containment.
    joins = [
        n for n in ast.walk(helper) if isinstance(n, ast.Compare)
    ]
    ops = [type(op) for cmp in joins for op in cmp.ops]
    assert ast.In not in ops, (
        "_token_level_hit must not use substring containment; that is the "
        "round-156 bug that inflated a scaffolding arm by 33 points"
    )


def test_gold_nll_per_chunk_body_is_inside_the_offset_loop() -> None:
    """Every chunk must be scored, not just the last one.

    Regression for f1685cf: the ``for group in groups`` refactor that kept the
    raw and spaced gold-NLL variants in separate chunkings re-indented the
    ``for offset in range(...)`` header one level deeper but left the 57-line
    per-chunk body behind.  The body therefore ran once per variant group, on
    whatever ``batch`` the loop happened to leave behind, and
    ``--qa-gold-nll`` scored only the last ``--qa-batch-size`` items -- silently,
    because the metrics still looked like ordinary numbers.  Every artifact on
    the box carrying ``nll_spaced`` reported ``qa_n = 4.0`` for a 1500-item eval.

    Purely static, so it catches the mistake before a GPU run rather than after:
    the forward pass must be *inside* the offset loop.
    """
    tree = _parse()
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    fn = funcs.get("_qa_gold_nll")
    assert fn is not None

    offset_loops = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "offset"
    ]
    assert len(offset_loops) == 1, f"expected one offset loop, found {len(offset_loops)}"
    loop = offset_loops[0]

    names_in_loop = {n.id for n in ast.walk(loop) if isinstance(n, ast.Name)}
    for required in ("ids_np", "mask_np", "batch"):
        assert required in names_in_loop, (
            f"{required!r} is not built inside the offset loop, so the per-chunk "
            "body has escaped it and only one chunk will be scored"
        )

    forwards = [
        call
        for call in ast.walk(loop)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "no_grad"
    ]
    assert forwards, (
        "the model forward is not inside the offset loop; --qa-gold-nll would "
        "score only the final batch"
    )


def test_gold_nll_metric_counts_every_item() -> None:
    """``qa_n`` must be recomputed from what was actually scored.

    A guard against the same class of failure surviving a future refactor in a
    different shape: the number of scored items must come from the answers list,
    not from whatever a loop variable happens to hold.
    """
    tree = _parse()
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    fn = funcs.get("_qa_gold_nll")
    assert fn is not None
    source = ast.unparse(fn)
    assert 'metrics["qa_n"] = float(len(answers))' in source.replace("'", '"'), (
        "qa_n must be len(answers); deriving it from a loop variable is how the "
        "f1685cf bug stayed invisible"
    )
