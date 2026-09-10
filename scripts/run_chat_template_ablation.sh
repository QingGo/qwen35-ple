#!/usr/bin/env bash
# Compare completion-style prompts vs the tokenizer's native chat template.
#
# This is a small, low-impact ablation (300 held-out items) that answers:
#   does not applying the chat template hurt format compliance / scores?
#
# Arms:
#   raw-no-reader
#   chat-no-reader
#   raw-sft-mixed50
#   chat-sft-mixed50
#   chat-sft-mixed50-purecode
set -uo pipefail
ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
OUT="$ROOT/outputs/chat-template-ablation"
LOG="$ROOT/logs/chat-template-ablation.log"
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p "$OUT" "$ROOT/logs"

log() { echo "=== [chat-ablation] $* $(date -Is) ===" | tee -a "$LOG"; }
run_logged() {
  local logfile="$1"; shift
  echo "--- [$(date -Is)] $*" >>"$logfile"
  "$@" >>"$logfile" 2>&1
  local rc=$?
  echo "--- [$(date -Is)] rc=$rc" >>"$logfile"
  return $rc
}

# 300-item subset: 100/task, fixed order from the standard eval file.
if [[ ! -f data/qa-standard/eval-300.jsonl ]]; then
  "$PY" - <<'PY'
import json, collections
rows=[json.loads(l) for l in open("data/qa-standard/eval.jsonl") if l.strip()]
by=collections.defaultdict(list)
for r in rows: by[r["task"]].append(r)
out=[]
for t in ["boolq","triviaqa","nq"]:
    out.extend(by[t][:100])
with open("data/qa-standard/eval-300.jsonl","w") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False)+"\n")
print("wrote eval-300", len(out))
PY
fi

PROMPT=$'Question: {question}\nAnswer:'
BOOLQ_PROMPT=$'Question: {question}\nAnswer with one word, Yes or No:'

run_arm() {
  # run_arm <name> <modes> <reader-or-empty> <chat:0|1>
  local name="$1" modes="$2" reader="$3" chat="$4"
  local dir="$OUT/$name"
  mkdir -p "$dir"
  if [[ -f "$dir/DONE" ]]; then log "skip $name"; return 0; fi
  local load_args=()
  local chat_args=()
  if [[ -n "$reader" ]]; then load_args=(--load-reader "$reader"); fi
  if [[ "$chat" == "1" ]]; then chat_args=(--qa-chat-template); fi
  log "arm $name modes=$modes reader=$reader chat=$chat"
  run_logged "$ROOT/logs/chat-ablation-$name.log" \
    "$PY" -u scripts/run_phase0.py \
      --live-store --tokens-npy data/phase1/PURE_WIKI/tokens.npy \
      --rows-dir /dev/shm/qwen38-rows \
      --model-dir "$ROOT/models/qwen38_ple" \
      --model "$ROOT/models/Qwen3.5-0.8B" \
      --reader official --layer 2 --device cuda --bridge-mlp --out-mlp \
      --official-reader-path data/official_ple_reader.pt \
      "${load_args[@]}" \
      --steps 0 --seq-len 128 --lr 1e-4 --seeds 0 --modes "$modes" \
      --qa --qa-exact-match --qa-max-new-tokens 32 \
      --qa-batch-size 8 --qa-batch-max-tokens 2048 \
      --qa-prompt-template "$PROMPT" \
      --qa-boolq-prompt-template "$BOOLQ_PROMPT" \
      "${chat_args[@]}" \
      --qa-file data/qa-standard/eval-300.jsonl \
      --resume --partial-dir "$dir/partial" --backup-dir "$dir/backup" \
      --output "$dir/phase1.json"
  if grep -q '"summary"' "$dir/phase1.json" 2>/dev/null; then
    "$PY" - "$dir" <<'PY' >>"$ROOT/logs/chat-template-ablation.log" 2>&1
import json, sys, collections
from pathlib import Path
from qwen35_ple.eval.answers import score_answer_v2
d=Path(sys.argv[1])
data=json.load(open(d/"phase1.json"))
summary=data.get("summary",{})
mode=next(iter(summary))
answers=summary[mode]["details"][0]["qa_exact"]["answers"]
by=collections.defaultdict(list)
for r in answers:
    s=score_answer_v2(str(r.get("generated","")), str(r.get("answer","")), task=r.get("task"))
    metric="extracted_exact" if r.get("task")=="boolq" else "extracted_contains"
    by[r.get("task","?")].append(int(bool(s[metric])))
metrics={t: sum(v)/len(v) for t,v in by.items()}
metrics["overall"]=sum(sum(v) for v in by.values())/sum(len(v) for v in by.values())
json.dump(metrics, open(d/"metrics.json","w"), indent=2)
lines=["# Chat-template ablation: "+d.name,"","| Task | Accuracy |","|---|---:|"]
for t,v in metrics.items(): lines.append(f"| {t} | {v:.4f} |")
(d/"metrics.md").write_text("\n".join(lines)+"\n")
print(d.name, metrics)
PY
    touch "$dir/DONE"
  fi
}

run_arm "raw-no-reader" "no-reader" "" "0"
run_arm "chat-no-reader" "no-reader" "" "1"
run_arm "raw-sft-mixed50" "real" "$ROOT/outputs/sft-mixed50/reader-PURE_WIKI-real-seed0.pt" "0"
run_arm "chat-sft-mixed50" "real" "$ROOT/outputs/sft-mixed50/reader-PURE_WIKI-real-seed0.pt" "1"
run_arm "chat-sft-mixed50-purecode" "real" "$ROOT/outputs/sft-mixed50-purecode/reader-PURE_CODE-real-seed0.pt" "1"

# Combined markdown.
"$PY" - <<'PY' >>"$ROOT/logs/chat-template-ablation.log" 2>&1
import json
from pathlib import Path
out=Path("/root/autodl-tmp/qwen35-ple/outputs/chat-template-ablation")
rows=[]
for p in sorted(out.glob("*/metrics.json")):
    rows.append((p.parent.name, json.load(open(p))))
lines=["# Chat-template ablation (300 held-out items)","","| Arm | BoolQ | TriviaQA | NQ | Overall |","|---|---:|---:|---:|---:|"]
for name,m in rows:
    lines.append(f"| {name} | {m.get('boolq',float('nan')):.4f} | {m.get('triviaqa',float('nan')):.4f} | {m.get('nq',float('nan')):.4f} | {m.get('overall',float('nan')):.4f} |")
(out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
print("\n".join(lines))
PY

log "CHAT ABLATION DONE"
echo done > "$OUT/DONE"
