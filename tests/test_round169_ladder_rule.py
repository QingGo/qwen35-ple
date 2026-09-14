"""Regression tests for the phase-2 ladder's stopping rule.

The bug these pin, which actually happened on the round-169 run:

``run_point`` was invoked as ``D1=$(run_point ...)``, and it called ``say`` on
stdout.  So ``D1`` held the whole progress trace followed by the number:

    [01:06:22] START train-lr3.162e-4  (mem 52 GB, gpu 0%)
    ...
    -0.06379010709223071

The frozen rule then asked ``awk -v a="$D1" 'BEGIN{exit !(a+0 > b+0)}'`` with
``b = -0.0027166``.  The container's awk coerces that leading ``[`` to 0, and
``0 > -0.00272`` is true, so the ladder climbed to a second lr point after a
first point that had already LOST to lr=1e-4.  The correct comparison stops.

(The same expression errors out under BWK awk, so the rule did not merely read a
wrong number -- it read a *different* number on a different machine.  A decision
rule must not depend on that.)

These tests extract the shipped functions from the script itself rather than
re-implementing them, so they fail if the guard is ever removed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_round169_phase2.sh"

# The real values from the round-169 ladder.
DELTA_1E4 = "-0.0027166083372815774"
DELTA_316E4 = "-0.06379010709223071"
LEAKED = (
    "[01:06:22] START train-lr3.162e-4  (mem 52 GB, gpu 0%)\n"
    "[02:13:34] END   train-lr3.162e-4 rc=0 wall=4032s\n"
    f"{DELTA_316E4}   (lr=1e-4 was {DELTA_1E4}; positive delta means training helped)"
)


def _fn(name: str) -> str:
    src = SCRIPT.read_text()
    m = re.search(rf"^{re.escape(name)}\(\) \{{.*\}}$", src, re.MULTILINE)
    assert m, f"{name}() not found in {SCRIPT.name}"
    return m.group(0)


def _bash(body: str, **env: str) -> subprocess.CompletedProcess:
    script = "\n".join([_fn("is_number"), _fn("better_than"), body])
    return subprocess.run(
        ["bash", "-c", script], check=False, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", **env}
    )


def test_is_number_accepts_a_real_delta():
    for value in (DELTA_1E4, DELTA_316E4, "0", "1e-4", "-0.5"):
        r = _bash('is_number "$V"', V=value)
        assert r.returncode == 0, f"{value!r} should be numeric: {r.stderr}"


def test_is_number_rejects_the_leaked_progress_trace():
    # This is the whole bug: without the guard the rule asks awk to compare a log.
    r = _bash('is_number "$V"', V=LEAKED)
    assert r.returncode != 0


def test_is_number_rejects_empty_and_garbage():
    for value in ("", "-", "n/a", "delta=?  -0.06"):
        r = _bash('is_number "$V"', V=value)
        assert r.returncode != 0, f"{value!r} must not pass as a number"


def test_the_real_ladder_result_stops_the_ladder():
    # Point 1 came out at -0.06379 against a -0.00272 reference; it LOST, and the
    # frozen rule's else-branch is "stop".
    r = _bash('if better_than "$A" "$B"; then echo CLIMB; else echo STOP; fi',
              A=DELTA_316E4, B=DELTA_1E4)
    assert r.stdout.strip() == "STOP"


def test_a_genuinely_improving_delta_climbs():
    r = _bash('if better_than "$A" "$B"; then echo CLIMB; else echo STOP; fi',
              A="0.004", B=DELTA_1E4)
    assert r.stdout.strip() == "CLIMB"


def test_the_leaked_value_is_refused_rather_than_compared():
    # End-to-end shape of the fixed call site: guard first, decide second.
    body = (
        'if ! is_number "$A"; then echo REFUSED; '
        'elif better_than "$A" "$B"; then echo CLIMB; else echo STOP; fi'
    )
    assert _bash(body, A=LEAKED, B=DELTA_1E4).stdout.strip() == "REFUSED"
    assert _bash(body, A=DELTA_316E4, B=DELTA_1E4).stdout.strip() == "STOP"


def test_the_script_no_longer_prints_progress_on_stdout():
    # say() must append to the log file.  If it ever goes back to stdout, the
    # captured delta silently becomes a log excerpt again.
    src = SCRIPT.read_text()
    m = re.search(r"^say\(\) \{.*\}$", src, re.MULTILINE)
    assert m, "say() not found"
    assert ">>" in m.group(0), f"say() must append to a file, got: {m.group(0)}"


def test_failure_and_step_state_is_file_backed_not_a_subshell_array():
    # FAILED+=()/STEPS+=() inside run_point (called via $(...)) were discarded,
    # so a killed arm was reported as "all steps ok".
    src = SCRIPT.read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "FAILED+=(" not in code
    assert "STEPS+=(" not in code
    for fn in ("fail", "step", "nfail"):
        assert re.search(rf"^{fn}\(\) ", code, re.MULTILINE), f"{fn}() missing"


def test_the_failure_count_is_correct(tmp_path):
    pass_file = tmp_path / "f"
    body = f'FAILFILE={pass_file}\n' + _fn("nfail") + "\necho $(nfail)"
    assert subprocess.run(["bash", "-c", body], check=False, capture_output=True, text=True).stdout.strip() == "0"
    pass_file.write_text("train-a\nbank-a\ntrain-a\n")
    r = subprocess.run(["bash", "-c", body], check=False, capture_output=True, text=True)
    assert r.stdout.strip() == "2", "duplicate entries must collapse"


def test_the_queue_is_killed_before_waiting_for_its_jobs():
    # Ordering bug: with wait_quiet first, a job the queue starts while we wait
    # is waited for forever, and the pkill that could stop it is queued behind
    # it.  It launched a 4-epoch code arm three seconds before the handoff.
    # Match the CALL, not the word: the comment above it explains the bug by
    # name, which is exactly the kind of self-reference that made this test fail
    # the first time.
    src = SCRIPT.read_text()
    section = src[src.index("# ---- 2.") : src.index("# ---- 3.")]
    lines = [ln.strip() for ln in section.splitlines()]
    kill_at = lines.index("pkill -f '[r]un_round169_overnight.sh' 2>/dev/null || true")
    wait_at = lines.index("wait_quiet")
    assert kill_at < wait_at, "the kill must precede the wait"
    for pat in ("[r]ound169_row_snapshot", "[r]ound169_train_rows", "[r]ound169_eval_rows"):
        assert pat in section, f"{pat} must be killed in the handoff section"
