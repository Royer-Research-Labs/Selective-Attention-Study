"""Figures for the README, built from the result files.

    python scripts/make_figures.py [--reference results/reference] [--out docs/figures]

Writes retrieval_16k.png, bos_blind_spot.png, pg19_16k.png, kernel_cost.png, pg19_long.png,
niah_hard.png and context_use.png. Colours are the
first four slots of a categorical palette in its fixed order (one per model family, the same in
every figure), checked with a colour-vision-deficiency validator (adjacent-pair CVD dE >= 9.1);
Selective Attention without BOS is the same blue, dashed, or its lighter ramp step in bar charts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a3a29d"
BLUE_LIGHT = "#86b6ef"   # blue ramp step 250: the no-BOS variant of the recommended design in bar charts
FAMILIES = [  # label, arm, repository, colour, dashed
    ("Selective + BOS, recommended", "all_selective_nope_bos", "here", BLUE, False),
    ("Dilution", "all_dilution_nope", "ref", ORANGE, False),
    ("alt6 (softmax + RoPE / dilution), unscaled", "alt6_softmax_rope_dilution", "ref", AQUA, False),
    ("Softmax + RoPE, unscaled", "all_softmax_rope", "ref", YELLOW, False),
    ("Selective, no BOS", "all_selective_nope", "here", BLUE, True),
]
SEEDS = (1337, 2357, 7331)


def style(ax, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="semibold", pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def niah(results: Path, arm: str, seed: int) -> dict[int, float]:
    acc = {}
    for name in [f"scale1x__{arm}_seed{seed}"] + [f"{t}__scale_16384_{arm}_seed{seed}"
                                                  for t in ("extrap64k", "extrap128k", "extrap256k", "extrap512k")]:
        rows = None
        for sub in ("niah_v4", "niah"):
            rows = load(results / sub / f"{name}.json")
            if rows:
                break
        for r in rows or []:
            acc.setdefault(int(r["context_length"]), float(r["accuracy"]))
    return acc


def save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def retrieval(roots, out):
    xs = [3840, 7936, 12288, 16000, 24000, 32000, 48000, 64000, 96000, 128000, 192000, 256000, 384000, 512000]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    style(ax, "Needle retrieval inside and beyond the 16K training length",
          "209M models, 1x Chinchilla; line = mean over seeds, band = seed range; chance = 1/7")
    ax.axvspan(xs[0] * 0.9, 16384, color="#f1f0ec", zorder=0)
    ax.text(5200, 0.06, "inside training length", fontsize=8.5, color=INK2)
    ax.axhline(1 / 7, color=NEUTRAL, linewidth=1)
    ax.text(xs[-1] * 1.05, 1 / 7, "chance", fontsize=8.5, color=INK2, va="center")
    for label, arm, repo, colour, dashed in FAMILIES:
        per_seed = [niah(roots[repo], arm, s) for s in SEEDS]
        per_seed = [p for p in per_seed if p]
        pts = [x for x in xs if sum(x in p for p in per_seed) >= 2]
        if not pts:
            continue
        vals = np.array([[p[x] for p in per_seed if x in p] for x in pts], dtype=object)
        mean = np.array([np.mean(v) for v in vals])
        lo, hi = np.array([min(v) for v in vals]), np.array([max(v) for v in vals])
        if not dashed:
            ax.fill_between(pts, lo, hi, color=colour, alpha=0.10, linewidth=0)
        ax.plot(pts, mean, color=colour, linewidth=2, linestyle=(0, (4, 3)) if dashed else "-",
                solid_capstyle="round", label=f"{label} ({len(per_seed)} seeds)")
        ax.plot(pts[-1:], mean[-1:], "o", color=colour, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xscale("log")
    ax.set_xticks([4096, 16384, 65536, 131072, 262144, 524288])
    ax.set_xticklabels(["4K", "16K", "64K", "128K", "256K", "512K"])
    ax.set_xlim(xs[0] * 0.9, xs[-1] * 1.25)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("input length (tokens)", color=INK2, fontsize=9)
    ax.set_ylabel("retrieval accuracy", color=INK2, fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8.5, frameon=False, labelcolor=INK)
    save(fig, out / "retrieval_16k.png")


def depth0(path, model, length=1920):
    """Depth-0 placements retrieved, from a scripts/investigate/niah_investigate.py result."""
    rows = load(path)["models"][model]["niah"][str(length)]["rows"]
    return sum(r["pass"] for r in rows if r["depth"] == 0.0)


def blind_spot(out, here):
    seeds = ["1337", "2357", "7331"]
    inv = here / "investigate"
    no_bos = [depth0(inv / "early_needles.json", f"s{s}") for s in seeds]
    bos = [depth0(inv / f"early_needles_bos_s{s}.json", f"bos_s{s}") for s in seeds]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    style(ax, "Needle in the first sentence of the input",
          "depth-0 placements retrieved, of 7, at 1,920 tokens; every other depth is retrieved")
    x = np.arange(len(seeds))
    w = 0.32
    for off, vals, colour, label in ((-w / 2 - 0.01, no_bos, BLUE_LIGHT, "No BOS"), (w / 2 + 0.01, bos, BLUE, "BOS at every window start")):
        bars = ax.bar(x + off, vals, width=w, color=colour, label=label, zorder=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v}/7", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylim(0, 7.8)
    ax.set_yticks([0, 7])
    ax.grid(False, axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5, frameon=False, labelcolor=INK, ncol=2)
    save(fig, out / "bos_blind_spot.png")


def pg19(roots, out):
    ranges = ["0.5", "1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K"]
    mids = np.array([384, 768, 1536, 3072, 6144, 12288, 24576, 49152, 98304])
    sel = load(roots["here"] / "longppl/pg19_209m_16k_1x_selective.json")["models"]
    bos3 = load(roots["here"] / "longppl/pg19_209m_16k_1x_selective_bos_s2357.json")
    if bos3:
        sel.update(bos3["models"])
    ref1 = load(roots["ref"] / "longppl/pg19_209m_16k_1x.json")["models"]
    ref2 = load(roots["ref"] / "longppl/pg19_209m_16k_2x.json")["models"]
    s2 = {**load(roots["here"] / "longppl/pg19_209m_16k_2x_selective_s1337.json")["models"]}
    extra = load(roots["here"] / "longppl/pg19_209m_16k_2x_selective_s2357.json")
    if extra:
        s2 = {"s1337": s2["selective_s1337"], "s2357": extra["models"]["selective_s2357"]}

    def series(models, names):
        return np.array([models[n]["curve"] for n in names if n in models])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.25})
    style(a1, "Long books (PG-19), 1x Chinchilla", "loss by position, 29 books; past 16K is beyond training")
    style(a2, "After 2x Chinchilla", "same books; selective without BOS")
    groups1 = [("Selective + BOS, recommended", series(sel, ["selective_bos_s1337", "selective_bos_s2357", "selective_bos_s7331"]), BLUE, False),
               ("Dilution", series(ref1, ["dilution_nope_s2357"]), ORANGE, False),
               ("alt6 (softmax + RoPE / dilution)", series(ref1, ["alt6_s1337", "alt6_s2357"]), AQUA, False),
               ("Softmax + RoPE, NTK past 16K", series(ref1, ["softmax_rope_s1337", "softmax_rope_s2357"]), YELLOW, False),
               ("Selective, no BOS", series(sel, ["selective_s1337", "selective_s2357", "selective_s7331"]), BLUE, True)]
    groups2 = [("Selective (2 seeds)", series(s2, ["s1337", "s2357"]), BLUE, False),
               ("Dilution (2 seeds)", series(ref2, ["dilution_nope_s1337", "dilution_nope_s2357"]), ORANGE, False)]
    for ax, groups, ylim in ((a1, groups1, (3.3, 4.3)), (a2, groups2, (3.1, 3.9))):
        ax.axvspan(16384, 140000, color="#f1f0ec", zorder=0)
        for label, arr, colour, dashed in groups:
            if arr.size == 0:
                continue
            mean = arr.mean(0)
            if len(arr) > 1 and not dashed:
                ax.fill_between(mids, arr.min(0), arr.max(0), color=colour, alpha=0.10, linewidth=0)
            ax.plot(mids, np.minimum(mean, ylim[1]), color=colour, linewidth=2,
                    linestyle=(0, (4, 3)) if dashed else "-", label=label)
            if mean.max() > ylim[1]:
                j = int(np.argmax(mean > ylim[1]))
                ax.annotate(f"softmax + RoPE: {mean[-2]:.1f} / {mean[-1]:.1f}" "\n" "at 32-64K / 64-128K (off scale)",
                            xy=(mids[j] * 0.97, ylim[1] - 0.01), xytext=(mids[j] * 0.08, ylim[1] - 0.16),
                            fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        ax.set_xscale("log")
        ax.set_xticks(mids)
        ax.set_xticklabels(ranges, fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_xlabel("position in book (end of range, tokens)", color=INK2, fontsize=9)
        ax.set_ylabel("loss (nats / token)", color=INK2, fontsize=9)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=False, labelcolor=INK)
    save(fig, out / "pg19_16k.png")


def pg19_long(out):
    """Loss relative to the model's own 8-16K loss, out to 512K (scripts/prepare_pg19_long.py books)."""
    def curves(names):
        rows = []
        for n in names:
            d = load(ROOT / f"results/longppl/pg19long_{n}.json")
            if d:
                c = np.array(list(d["models"].values())[0]["curve"][5:])
                rows.append(c - c[0])
        return np.array(rows)
    mids = np.array([12288, 24576, 49152, 98304, 196608, 393216])
    labels = ["8-16K", "16-32K", "32-64K", "64-128K", "128-256K", "256-512K"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, gridspec_kw={"wspace": 0.08})
    panels = [("1x Chinchilla", [("Selective + BOS (3 seeds)", curves(["selective_bos_s1337_1x", "selective_bos_s2357_1x", "selective_bos_s7331_1x"]), BLUE),
                                 ("Dilution (seed 2357)", curves(["dilution_nope_s2357_1x"]), ORANGE)]),
              ("2x Chinchilla", [("Selective, no BOS (2 seeds)", curves(["selective_s1337_2x", "selective_s2357_2x"]), BLUE),
                                 ("Dilution (2 seeds)", curves(["dilution_nope_s1337_2x", "dilution_nope_s2357_2x"]), ORANGE)])]
    for ax, (title, groups) in zip(axes, panels):
        style(ax, f"Long books to 512K tokens, {title}",
              "loss minus the model's own 8-16K loss; 16 books of 512K+ tokens" if ax is axes[0] else "same books")
        ax.axhline(0, color=NEUTRAL, linewidth=1)
        for label, arr, colour in groups:
            if arr.size == 0:
                continue
            if len(arr) > 1:
                ax.fill_between(mids, arr.min(0), arr.max(0), color=colour, alpha=0.10, linewidth=0)
            ax.plot(mids, arr.mean(0), color=colour, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        ax.set_xscale("log")
        ax.set_xticks(mids)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("position in book (tokens); trained on 16K", color=INK2, fontsize=9)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK)
    axes[0].set_ylabel("change in loss (nats / token)", color=INK2, fontsize=9)
    save(fig, out / "pg19_long.png")


def kernel_cost(out):
    data = load(ROOT / "results/bench/layer_fwd_bwd_v2.json")
    rows = data["rows"]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    style(ax, "One attention layer, forward + backward",
          "batch 2, 12 heads x 64, bf16, RTX PRO 6000; milliseconds, log scale")
    for op, label, colour in (("selective", "Selective kernel", BLUE), ("dilution", "Dilution kernel", ORANGE),
                              ("softmax", "Softmax (flash, PyTorch SDPA)", YELLOW)):
        pts = sorted((r["seq"], r["fwd_ms"] + r["bwd_ms"]) for r in rows if r["op"] == op)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=colour, linewidth=2, label=label, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=2)
        nudge = {"selective": 0.88, "dilution": 1.12}.get(op, 1.0)
        ax.text(xs[-1] * 1.08, ys[-1] * nudge, f"{ys[-1]:.1f} ms", fontsize=8.5, color=INK, va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([2048, 4096, 8192, 16384])
    ax.set_xticklabels(["2K", "4K", "8K", "16K"])
    ax.set_xlim(1800, 26000)
    ax.set_yticks([0.5, 1, 3, 10, 30])
    ax.set_yticklabels(["0.5", "1", "3", "10", "30"])
    ax.minorticks_off()
    ax.set_xlabel("sequence length (tokens)", color=INK2, fontsize=9)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK)
    save(fig, out / "kernel_cost.png")


def hard_mode(here, out):
    """Niah hard mode (natural filler; 0 or 3 keyed distractors), results/investigate/niah_hard_d*.json."""
    xs = [4096, 16000, 32000, 64000, 128000]
    groups = [("Selective + BOS", ["sel_bos_s1337", "sel_bos_s2357", "sel_bos_s7331"], BLUE),
              ("Dilution", ["dilution_s2357"], ORANGE),
              ("Softmax + RoPE, NTK past 16K", ["softmax_rope_s1337", "softmax_rope_s2357"], YELLOW)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    panels = [("d0", "One fact in natural text", "no distractors"),
              ("d3", "One fact among four", "3 distractor facts about other keys")]
    for ax, (tag, title, sub) in zip(axes, panels):
        data = load(here / f"investigate/niah_hard_{tag}.json")["models"]
        style(ax, title, sub)
        ax.axvspan(xs[0] * 0.85, 16384, color="#f1f0ec", zorder=0)
        ax.axhline(1 / 7, color=NEUTRAL, linewidth=1)
        ax.text(xs[-1] * 1.16, 1 / 7, "chance", fontsize=8, color=INK2, va="center")
        if tag == "d3":
            ax.axhline(0.25, color=NEUTRAL, linewidth=1, linestyle=(0, (2, 2)))
            ax.text(xs[-1] * 1.08, 0.25, "a planted fact\nat random", fontsize=8, color=INK2, va="center")
        for label, names, colour in groups:
            runs = [[data[n]["by_length"][str(x)]["accuracy"] for x in xs] for n in names if n in data]
            arr = np.array(runs)
            if len(runs) > 1:
                ax.fill_between(xs, arr.min(0), arr.max(0), color=colour, alpha=0.12, linewidth=0)
            ax.plot(xs, arr.mean(0), color=colour, linewidth=2, solid_capstyle="round",
                    label=f"{label} ({len(runs)} seed{'s' if len(runs) > 1 else ''})")
            ax.plot(xs[-1:], arr.mean(0)[-1:], "o", color=colour, markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=2)
        ax.set_xscale("log")
        ax.set_xticks([4096, 16384, 32768, 65536, 131072])
        ax.set_xticklabels(["4K", "16K", "32K", "64K", "128K"])
        ax.set_xlim(xs[0] * 0.85, xs[-1] * 1.5)
        ax.set_ylim(0, 1.04)
        ax.set_xlabel("input length (tokens)", color=INK2, fontsize=9)
    axes[0].set_ylabel("retrieval accuracy", color=INK2, fontsize=9)
    axes[0].text(4400, 0.04, "inside training length", fontsize=8, color=INK2)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.16), ncol=3, fontsize=8.5, frameon=False,
                   labelcolor=INK)
    fig.subplots_adjust(wspace=0.12)
    save(fig, out / "niah_hard.png")


def context_use(here, out):
    """Full-prefix minus 16K-rolling-window loss on the 512K PG-19 books (scripts/investigate/rolling_window.py)."""
    d = load(here / "investigate/rolling_window_512k_1x.json")
    rs = [(lo, hi) for lo, hi in d["ranges"] if lo >= 16384]
    idx = [d["ranges"].index([lo, hi]) if [lo, hi] in d["ranges"] else d["ranges"].index((lo, hi)) for lo, hi in rs]
    xs = [np.sqrt(lo * hi) for lo, hi in rs]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    style(ax, "Does the whole book help, or only the last 16K tokens?",
          "PG-19, 16 books of 512K tokens; full-prefix loss minus 16K-window loss on the same tokens")
    ax.axhline(0, color=NEUTRAL, linewidth=1)
    ax.text(xs[-1] * 1.12, 0.012, "full prefix worse", fontsize=8, color=INK2, va="bottom")
    ax.text(xs[-1] * 1.12, -0.012, "full prefix better", fontsize=8, color=INK2, va="top")
    groups = [("Selective + BOS", ["sel_bos_s1337", "sel_bos_s2357", "sel_bos_s7331"], BLUE),
              ("Dilution", ["dilution_s2357"], ORANGE)]
    for label, names, colour in groups:
        arr = np.array([[d["models"][n]["full_minus_window"][i] for i in idx] for n in names])
        if len(names) > 1:
            ax.fill_between(xs, arr.min(0), arr.max(0), color=colour, alpha=0.12, linewidth=0)
        ax.plot(xs, arr.mean(0), color=colour, linewidth=2, solid_capstyle="round",
                label=f"{label} ({len(names)} seed{'s' if len(names) > 1 else ''})")
        ax.plot(xs[-1:], arr.mean(0)[-1:], "o", color=colour, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
        se = np.array([[d["models"][n]["full_minus_window_se"][i] for i in idx] for n in names]).mean(0)
        shift = 0.97 if colour == BLUE else 1.03     # keep the two series' bars apart
        ax.errorbar([x * shift for x in xs], arr.mean(0), yerr=se, fmt="none", ecolor=colour, elinewidth=1,
                    capsize=2.5, alpha=0.7)
    ax.text(xs[0] * 0.78, -0.105, "bars: standard error across books (mean over seeds)", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lo // 1024}-{hi // 1024}K" for lo, hi in rs])
    ax.minorticks_off()
    ax.set_xlim(xs[0] * 0.75, xs[-1] * 1.9)
    ax.set_xlabel("position range (tokens)", color=INK2, fontsize=9)
    ax.set_ylabel("loss difference (nats / token)", color=INK2, fontsize=9)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK)
    save(fig, out / "context_use.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", default=str(ROOT / "results/reference"),
                    help="the published Dilution-Attention results (copied here, see SOURCE.json; or "
                         "the results/ directory of a checkout of that repository)")
    ap.add_argument("--out", default=str(ROOT / "docs/figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    roots = {"here": ROOT / "results", "ref": Path(a.reference)}
    retrieval(roots, out)
    blind_spot(out, roots["here"])
    pg19(roots, out)
    kernel_cost(out)
    pg19_long(out)
    hard_mode(roots["here"], out)
    context_use(roots["here"], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
