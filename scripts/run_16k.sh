#!/bin/bash
# Train one seed of the 16K Selective Attention model and run the evaluation set, with the same
# commands the Dilution-Attention 209M runs used (its block 17): train to the step-64,000
# checkpoint of the 128,174-step schedule, common-slice loss, NIAH inside the trained context
# (20 placements) and past it to 64K / 128K / 256K (10 placements). The 384K / 512K NIAH and the
# PG-19 runs are separate (README, "Reproducing").
#   bash scripts/run_16k.sh 1337 [2357 7331 ...]                  # the recommended BOS recipe
#   ARM=all_selective_nope bash scripts/run_16k.sh 1337            # without BOS
# Resumes a seed that stopped part-way and skips results that already exist. Writes to runs_repro/ and
# results_repro/ so the published run records and results are never touched (RUNS / RESULTS override);
# training counts as done only when the run logged its stop at step 64,000 AND its checkpoint is there.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
ARM=${ARM:-all_selective_nope_bos}
RUNS=${RUNS:-runs_repro/scale_16384}
RESULTS=${RESULTS:-results_repro}
mkdir -p "$RUNS" "$RESULTS/eval" "$RESULTS/niah" runs_repro
STOP=64000
L16="256 512 896 1024 1920 3072 3840 6144 7936 12288 16000"
log() { echo "[run_16k] $(date -Is) $*"; }
reached() { python - "$1" "$STOP" <<'PY'
import json, os, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1] + "/metrics.jsonl", encoding="utf-8") if l.strip()]
except FileNotFoundError:
    raise SystemExit(1)
stopped = any(r.get("event") == "stopped" and r.get("step", 0) >= int(sys.argv[2]) for r in rows)
raise SystemExit(0 if stopped and os.path.isfile(sys.argv[1] + "/latest.pt") else 1)
PY
}
for SEED in "$@"; do
  out=$RUNS/${ARM}_seed${SEED}
  if reached "$out"; then
    log "seed $SEED already at step $STOP"
  else
    resume=""; [ -f "$out/latest.pt" ] && resume="--resume $out/latest.pt"
    log "training seed $SEED to step $STOP ${resume:+(resuming)}"
    python -m selective_attention.train --config configs/scale_16384/$ARM.yaml --seed "$SEED" --output-dir "$out" \
      --stop-at-step $STOP $resume >> runs_repro/train_${ARM}_seed${SEED}.log 2>&1 || log "trainer exited non-zero"
  fi
  reached "$out" || { log "seed $SEED did not reach step $STOP; skipping its evaluations"; continue; }
  python scripts/report_run.py "$out" | head -1 | while read -r l; do log "$l"; done
  name="scale1x__${ARM}_seed${SEED}"
  if [ ! -f "$RESULTS/eval/$name.json" ]; then
    python scripts/eval_checkpoint.py "$out" --batches 64 --batch-size 2 --output "$RESULTS/eval/$name.json" 2>&1 \
      | grep -v USDT | tail -1 | while read -r l; do log "eval: $l"; done
  fi
  if [ ! -f "$RESULTS/niah/$name.json" ]; then
    python -m selective_attention.niah_eval --checkpoint "$out/latest.pt" --context-length $L16 --samples 20 --num-needles 8 \
      --output "$RESULTS/niah/$name.json" >> runs_repro/eval_${ARM}_seed${SEED}.log 2>&1 \
      && log "wrote $RESULTS/niah/$name.json" || log "NIAH FAILED: $name"
  fi
  for spec in "extrap64k 66000 16000 24000 32000 48000 64000" "extrap128k 132000 64000 96000 128000" \
              "extrap256k 260000 192000 256000"; do
    set -- $spec; tag=$1; override=$2; shift 2
    ename="${tag}__scale_16384_${ARM}_seed${SEED}"
    [ -f "$RESULTS/niah/$ename.json" ] && continue
    python -m selective_attention.niah_eval --checkpoint "$out/latest.pt" --context-length "$@" --context-override "$override" \
      --samples 10 --num-needles 8 --output "$RESULTS/niah/$ename.json" >> runs_repro/eval_${ARM}_seed${SEED}.log 2>&1 \
      && log "wrote $RESULTS/niah/$ename.json" || log "EXTRAP FAILED: $ename"
  done
done
log "done"
