"""Model-engine orchestration for qwen35-ple.

This package wraps calls into engram-peft. It does not implement the PLE layer;
it only constructs configurations and selects the EngramDB table source.
"""

from __future__ import annotations
