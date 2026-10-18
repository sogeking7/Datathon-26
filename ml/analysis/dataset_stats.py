"""Dataset facts that shape the evaluation protocol: windows per video, where labelled timecodes fall,
class counts, alternative-type videos, and confidence intervals for the headline metrics of ml/README.md.

    python ml/analysis/dataset_stats.py
"""

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_labels, load_setups  # noqa: E402

import evalkit  # noqa: E402

FPS, WIN, STRIDE = 2.0, 20.0, 10.0


def windows(duration, win=WIN, stride=STRIDE, fps=FPS):
    """Copy of app.vj.runtime.VideoJev.windows (runtime imports torch, which the laptop venv lacks)."""
    if duration <= win * 1.3:
        return [(0.0, duration)]
    starts = list(np.arange(0.0, max(duration - win, 0.0) + 1e-6, stride))
    if starts[-1] + win < duration:
        starts.append(math.floor((duration - win) * fps) / fps)
    return [(float(s), float(min(s + win, duration))) for s in starts]


# Normals the first release flags (ml/README.md §6), used for the duration-vs-FP check.
FP_FIRST_RELEASE = {"Х-5.mov", "Х-14.mov", "Х-16.mov", "Х-23.mov", "Х-29.mov", "Х-31.mov", "Х-39.mov"}


def fisher_one_sided(a, b, c, d):
    from math import comb

    n = a + b + c + d
    return sum(comb(a + b, x) * comb(c + d, a + c - x) / comb(n, a + c)
               for x in range(a, min(a + b, a + c) + 1) if a + c - x <= c + d)


def main():
    videos = load_labels()
    setups = load_setups()
    print(f"{'file':10} {'dur':>6} {'nwin':>4} {'setup':5} verdict      type                     t   windows with t")
    norm, pos = [], []
    for v in videos:
        w = windows(v["duration_sec"])
        t = v.get("timecode_sec")
        ev = v["events"][0] if v.get("events") else {}
        cont = [i for i, (a, b) in enumerate(w) if t is not None and a <= t < b]
        print(f"{v['file']:10} {v['duration_sec']:6} {len(w):4} {setups.get(v['file'], '?'):5} {v['verdict']:12} "
              f"{str(ev.get('type_code')):24} {str(t):>4} {cont}  {'FP' if v['file'] in FP_FIRST_RELEASE else ''}")
        (pos if v["verdict"] == "violation" else norm).append((v["file"], len(w), v["file"] in FP_FIRST_RELEASE))
    long = [r for r in norm if r[1] >= 5]
    short = [r for r in norm if r[1] < 5]
    a, c = sum(r[2] for r in long), sum(r[2] for r in short)
    print(f"\nnormals with >=5 windows: FP {a}/{len(long)}; with <5 windows: FP {c}/{len(short)}; "
          f"one-sided Fisher p = {fisher_one_sided(a, len(long) - a, c, len(short) - c):.3f}")
    print("windows: total", sum(r[1] for r in norm + pos), "normal", sum(r[1] for r in norm), "violation", sum(r[1] for r in pos))
    print("violation videos with >1 window:", sum(r[1] > 1 for r in pos), "of", len(pos))
    print("\nWilson 95% CI for ml/README.md headline numbers:")
    for name, k, n in [("recall 16/21", 16, 21), ("fp_rate 7/19", 7, 19), ("verdict 28/40", 28, 40),
                       ("type 9/21", 9, 21), ("tc within 2s 5/9", 5, 9)]:
        lo, hi = evalkit.wilson(k, n)
        print(f"  {name:18} {k / n:.3f}  [{lo:.3f}, {hi:.3f}]")
    print("\nevents per type:", Counter(e["type_code"] for v in videos for e in v.get("events", [])))
    print("alt types:", [(v["file"], e["type_code"], e.get("alt_type_codes")) for v in videos
                         for e in v.get("events", []) if e.get("alt_type_codes")])
    by_setup = Counter((setups.get(v["file"], "?"), v["verdict"]) for v in videos)
    print("setups (violation/normal):", {s: (by_setup[(s, "violation")], by_setup[(s, "no_violation")])
                                        for s in sorted({x for x, _ in by_setup})})


if __name__ == "__main__":
    main()
