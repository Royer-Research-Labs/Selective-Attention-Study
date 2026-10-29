"""One-paragraph result report for a finished (or running) run directory.

    python scripts/report_run.py runs/validate_s27/all_dilution_seed1337 [--ref runs/12l_8192_600m/all_dilution_seed1337]

Prints last step, throughput, val loss at the last eval and at 10k/20k/30k
(with the reference's values and the difference when --ref is given), the
stability verdict (grad median/p99, val rising 10k->20k, completeness) and the
NIAH accuracies if results/niah/<suite>__<name>.json exists.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def load(run):
    p = Path(run) / "metrics.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def evals(rows):
    return {r["step"]: r["loss"] for r in rows if r.get("event") == "evaluation"}


def niah_summary(run):
    run = Path(run)
    name = f"{run.parent.name}__{run.name}"
    p = Path("results/niah") / f"{name}.json"
    if not p.exists():
        return None
    j = json.load(p.open(encoding="utf-8"))
    if isinstance(j, dict) and "results" in j:
        j = j["results"]
    if isinstance(j, list):
        return " ".join(f"{x.get('context_length')}:{x.get('accuracy', float('nan')):.2f}" for x in j)
    if isinstance(j, dict):
        parts = []
        for k, v in j.items():
            acc = v.get("accuracy") if isinstance(v, dict) else v
            if isinstance(acc, (int, float)):
                parts.append(f"{k}:{acc:.2f}")
        return " ".join(parts)
    return str(j)[:200]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--ref")
    a = ap.parse_args()
    rows = load(a.run)
    if not rows:
        print(f"{a.run}: no metrics yet"); return 0
    ev = evals(rows)
    tr = [r for r in rows if r.get("event") == "train"]
    max_steps = next((r["schedule_max_steps"] for r in rows if r.get("schedule_max_steps")), None)
    if max_steps is None:
        import re
        cfg = Path(a.run) / "resolved_config.yaml"
        if cfg.exists():
            m = re.search(r"max_steps:\s*(\d+)", cfg.read_text(encoding="utf-8"))
            max_steps = int(m.group(1)) if m else None
    last = max(ev) if ev else 0
    tps = statistics.median([r["tokens_per_second"] for r in tr[-40:]]) if tr else float("nan")
    grads = [r["grad_norm"] for r in tr if "grad_norm" in r]
    gmed = statistics.median(grads) if grads else float("nan")
    g99 = sorted(grads)[int(0.99 * (len(grads) - 1))] if grads else float("nan")
    complete = max_steps is not None and last >= max_steps
    rising = 10000 in ev and 20000 in ev and ev[20000] > ev[10000]
    unstable = gmed > 2 or g99 > 20 or rising or not complete
    ref = evals(load(a.ref)) if a.ref and load(a.ref) else {}
    line = f"{a.run}: step {last}/{max_steps} {'complete' if complete else 'INCOMPLETE'} | {tps/1000:.1f}K tok/s | grad med {gmed:.2f} p99 {g99:.2f} | {'UNSTABLE' if unstable else 'stable'}"
    print(line)
    pts = [s for s in (10000, 20000, 30000, last) if s in ev]
    vals = []
    for s in pts:
        v = ev[s]
        if s in ref:
            vals.append(f"{s}: {v:.4f} (ref {ref[s]:.4f}, {v - ref[s]:+.4f})")
        else:
            vals.append(f"{s}: {v:.4f}")
    print("  val " + " | ".join(vals))
    n = niah_summary(a.run)
    if n:
        print("  niah " + n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
