"""Judge whether trained arms are stable.

    python scripts/judge_stability.py runs/12l_8192_600m hybrid_softmax4 hybrid_dilution4

Prints one line per arm and exits 0 if every arm is stable, 1 if any is
unstable or missing. Unstable = grad-norm median > 2.0 or p99 > 20 over
steps >= 1000, or val loss at 20k above val at 10k, or no `complete` event.
"""
import json, statistics, sys
from pathlib import Path

def judge(run_dir: Path):
    mp = run_dir / "metrics.jsonl"
    if not mp.is_file():
        return False, "missing"
    ev = [json.loads(l) for l in mp.open(encoding="utf-8") if l.strip()]
    g = sorted(e["grad_norm"] for e in ev if e.get("event") == "train" and e["step"] >= 1000)
    val = {e["step"]: e["loss"] for e in ev if e.get("event") == "evaluation"}
    done = [e for e in ev if e.get("event") == "complete"]
    if not g or not done:
        return False, "incomplete"
    med, p99 = statistics.median(g), g[max(0, int(0.99 * len(g)) - 1)]
    rising = 10000 in val and 20000 in val and val[20000] > val[10000]
    ok = med <= 2.0 and p99 <= 20 and not rising
    best = done[-1]["best_val_loss"]
    return ok, f"grad med {med:.2f} p99 {p99:.1f} rising={rising} best {best:.4f}"

if __name__ == "__main__":
    root, arms = Path(sys.argv[1]), sys.argv[2:]
    bad = 0
    for arm in arms:
        ok, why = judge(root / f"{arm}_seed1337")
        print(f"{'STABLE  ' if ok else 'UNSTABLE'} {arm}: {why}", flush=True)
        bad += not ok
    sys.exit(1 if bad else 0)
