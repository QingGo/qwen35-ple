#!/usr/bin/env python3
"""Extract standalone SVG files from diagram-design HTML sources."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"

PAIRS = [
    ("system_architecture_dd.html", "system_architecture_dd.svg"),
    ("inference_pipeline_dd.html", "inference_pipeline_dd.svg"),
]

for html_name, svg_name in PAIRS:
    html = (FIG_DIR / html_name).read_text(encoding="utf-8")
    match = re.search(r"<svg.*?</svg>", html, re.DOTALL)
    if not match:
        raise SystemExit(f"no <svg> found in {html_name}")
    svg = '<?xml version="1.0" encoding="UTF-8"?>\n' + match.group(0)
    (FIG_DIR / svg_name).write_text(svg, encoding="utf-8")
    print(f"wrote {svg_name}")
