"""Run one decision-logic experiment on cached scores and append it to the experiment registry.

Example (baseline, reproduces ml/calibrate.py on the production head's LOVO scores):

    python ml/experiment.py --scores runs/head/scores_cv.jsonl --exp-id E00 \
        --hypothesis "baseline: max over windows, F2 thresholds, LOVO" --registry reports/experiments.csv

Variants: --agg top2|zsidak|sidak, --calib bias, --split setup (leave-one-camera-session-out), --split rgkf
(repeated group 5-fold), --objective recall_at_fp --alpha 0.3, --no-per-type, --shrink 0.1,
--max-events 2, --fixed-thresholds ml/artifacts/thresholds.json (score the shipped thresholds),
--compare runs/E00_pred.jsonl (paired bootstrap against a saved prediction file).

Everything here is numpy-only and runs on a laptop in seconds; the only GPU-dependent input is the
scores file produced by ml/score_dataset.py / ml/train_head.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import date
from pathlib import Path

from common import load_labels, load_setups

import decision
import evalkit

REGISTRY_COLUMNS = [
    "exp_id", "date", "hypothesis", "scores", "policy", "split", "objective", "alpha", "per_type", "shrink",
    "max_events", "n_videos", "recall", "recall_ci", "fp_rate", "fp_rate_ci", "verdict_acc", "verdict_acc_ci",
    "balanced_acc", "type_acc_on_pos", "type_acc_acceptable", "tc_n", "tc_mae", "tc_within_2s",
    "event_f1_2s", "event_f1_5s", "roc_auc", "pr_auc", "runtime_sec", "decision", "notes",
]


def fmt(x, nd=3):
    if x is None:
        return ""
    if isinstance(x, tuple):
        return f"[{x[0]:.{nd}f}, {x[1]:.{nd}f}]"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def build_splits(kind: str, videos: list[dict], setup_of, k: int, repeats: int, seed: int):
    if kind == "lovo":
        return list(evalkit.lovo(videos))
    if kind == "setup":
        return list(evalkit.leave_one_group_out(videos, setup_of))
    if kind == "rgkf":
        return list(evalkit.repeated_group_kfold(videos, setup_of, k=k, repeats=repeats, seed=seed))
    if kind == "none":  # fit and evaluate on everything: optimistic, only for inspection
        return [(videos, videos)]
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True)
    p.add_argument("--aug", default="none")
    p.add_argument("--agg", default="max", choices=["max", "top2", "sidak", "zsidak"])
    p.add_argument("--calib", default="none", choices=["none", "bias"])
    p.add_argument("--rank", default="margin", choices=["margin", "raw"], help="order flagged types by margin (baseline) or raw max probability")
    p.add_argument("--events-from", default="self", choices=["self", "max"], help="max: type/timecode from the baseline max rule when it fires")
    p.add_argument("--max-events", type=int, default=1)
    p.add_argument("--split", default="lovo", choices=["lovo", "setup", "rgkf", "none"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--objective", default="f2_type", choices=["f2_type", "recall_at_fp", "youden", "balanced"])
    p.add_argument("--alpha", type=float, default=0.3, help="FP-rate budget for --objective recall_at_fp")
    p.add_argument("--no-per-type", action="store_true")
    p.add_argument("--shrink", type=float, default=None, help="max |per-type threshold - global|")
    p.add_argument("--fixed-thresholds", help="thresholds.json to score instead of fitting (e.g. production)")
    p.add_argument("--train-scores", help="scores_cv_train.jsonl: fit thresholds per fold on the fold head's own training scores (lovo/setup splits)")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--exp-id", required=True)
    p.add_argument("--hypothesis", default="")
    p.add_argument("--notes", default="")
    p.add_argument("--decision", default="investigate", help="keep | reject | investigate")
    p.add_argument("--registry", default=str(Path(__file__).resolve().parent.parent / "reports" / "experiments.csv"))
    p.add_argument("--save-pred", help="write per-video predictions (jsonl) for later paired comparison")
    p.add_argument("--compare", help="prediction jsonl of a reference experiment for a paired bootstrap")
    p.add_argument("--open-set", action="store_true", help="also run leave-one-type-out open-set probes")
    p.add_argument("--platt", action="store_true", help="nested 2-parameter Platt scaling of the video margin -> Brier/log-loss/ECE (E07)")
    p.add_argument("--simulate-min-confidence", type=float, default=None,
                   help="drop events whose raw score is below this value after the decision, as the first-release "
                        "analyzer did with MIN_CONFIDENCE=0.5 (only meaningful with --agg max/top2 raw probabilities)")
    a = p.parse_args()

    t0 = time.perf_counter()
    by_video = decision.load_scores(a.scores, a.aug)
    videos = [v for v in load_labels() if v["file"] in by_video]
    missing = [v["file"] for v in load_labels() if v["file"] not in by_video]
    if missing:
        print(f"warning: {len(missing)} labelled videos without scores: {missing}")
    setups = load_setups()
    setup_of = lambda v: setups.get(v["file"], "S?")  # noqa: E731
    policy = decision.Policy(agg=a.agg, calib=a.calib, max_events=a.max_events, rank=a.rank, events_from=a.events_from)
    splits = build_splits(a.split, videos, setup_of, a.k, a.repeats, a.seed)
    print("split:", a.split, evalkit.split_report(splits))
    fixed = json.loads(Path(a.fixed_thresholds).read_text()) if a.fixed_thresholds else None

    if a.split == "rgkf":
        # repeated folds give several predictions per video; keep the per-repeat metrics and the
        # pooled majority prediction for the registry row
        per_repeat, pooled = [], {}
        n_folds = a.k
        for r in range(a.repeats):
            chunk = splits[r * n_folds:(r + 1) * n_folds]
            pred, margins, used = decision.nested_evaluate(videos, by_video, policy, chunk, a.objective, a.alpha,
                                                           not a.no_per_type, a.shrink, fixed)
            m, _ = evalkit.video_metrics(videos, pred)
            per_repeat.append(m)
            for f, ev in pred.items():
                pooled.setdefault(f, []).append(ev)
        keys = ["recall", "fp_rate", "verdict_acc", "balanced_acc", "type_acc_on_pos"]
        print("repeated group k-fold, mean ± sd over repeats:")
        for k in keys:
            vals = [m[k] for m in per_repeat if m[k] is not None]
            print(f"  {k:16} {sum(vals) / len(vals):.3f} ± {(sum((x - sum(vals) / len(vals)) ** 2 for x in vals) / max(len(vals) - 1, 1)) ** 0.5:.3f}")
        # majority vote per video for the registry row
        predictions = {}
        for f, evs in pooled.items():
            flagged = sum(bool(e) for e in evs) * 2 > len(evs)
            predictions[f] = max(evs, key=lambda e: (bool(e), e[0][2] if e else 0)) if flagged else []
        margins = None  # no single margin per video across repeats
    else:
        train_scores, fold_key = None, None
        if a.train_scores:
            if a.split not in ("lovo", "setup"):
                raise SystemExit("--train-scores works with --split lovo or setup")
            train_scores = decision.load_train_scores(a.train_scores, a.aug)
            fold_key = (lambda test: test[0]["file"]) if a.split == "lovo" else (lambda test: setup_of(test[0]))
        out = decision.nested_evaluate(videos, by_video, policy, splits, a.objective, a.alpha,
                                       not a.no_per_type, a.shrink, fixed, train_scores, fold_key, platt=a.platt)
        predictions, margins, used = out[:3]
        probs = out[3] if a.platt else None

    if a.simulate_min_confidence is not None:
        predictions = {f: [e for e in evs if e[2] >= a.simulate_min_confidence] for f, evs in predictions.items()}
    metrics, rows = evalkit.video_metrics(videos, predictions)
    ci = evalkit.bootstrap_ci(videos, predictions, n_boot=a.bootstrap, seed=a.seed)
    probs = locals().get("probs")
    sm = evalkit.score_metrics(videos, margins, probs) if margins is not None else {"roc_auc": None, "pr_auc": None}
    if probs:
        bins = {}
        for v in videos:
            b = min(int(probs[v["file"]] * 5), 4)
            bins.setdefault(b, []).append(int(evalkit.is_violation(v)))
        print("Platt reliability (bin: n, observed rate):", {f"{b/5:.1f}-{(b+1)/5:.1f}": (len(ys), round(sum(ys) / len(ys), 2)) for b, ys in sorted(bins.items())})
    runtime = time.perf_counter() - t0

    for r in rows:
        print(f"{r['file']:10} {r['verdict']:12} label={','.join(r['label']) or '-':24} pred={r['pred']}"
              f"{'  type_ok' if r['type_ok'] else ''}{'  tc_err=' + str(r['tc_err']) if r['tc_err'] is not None else ''}")
    print("\nmetrics:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()}, ensure_ascii=False))
    print("95% bootstrap CI:", {k: fmt(v) for k, v in ci.items()})
    print("wilson: recall", fmt(evalkit.wilson(metrics["tp"], metrics["n_pos"])),
          "fp_rate", fmt(evalkit.wilson(metrics["fp"], metrics["n_neg"])))
    print("score metrics:", sm)
    print("by setup:", json.dumps(evalkit.breakdown(videos, predictions, setup_of), ensure_ascii=False))
    print("by windows:", json.dumps(evalkit.breakdown(videos, predictions, lambda v: "multi-window" if len(by_video[v["file"]]) > 1 else "single-window"), ensure_ascii=False))
    print("confusion:", {k: dict(c) for k, c in evalkit.confusion_by_type(videos, predictions).items()})
    if a.split != "rgkf":
        print("thresholds used (first fold):", next(iter(used.values())))

    if a.open_set:
        print("\nopen-set probes (leave-one-type-out, other gates only):")
        for code in decision.CODES:
            print(" ", decision.open_set_recall(videos, by_video, policy, code, a.objective, a.alpha, not a.no_per_type, a.shrink))

    if a.compare:
        ref = {json.loads(l)["file"]: [tuple(e) for e in json.loads(l)["events"]] for l in Path(a.compare).read_text().splitlines() if l.strip()}
        pb = evalkit.paired_bootstrap(videos, ref, predictions, n_boot=a.bootstrap, seed=a.seed)
        print("\npaired vs", a.compare)
        for k, v in pb.items():
            print(f"  Δ{k:16} {v['mean']:+.3f} CI {fmt(v['ci'])}  P(new better)={v['p_b_better']:.2f}")
        print("  flips:", evalkit.flips(videos, ref, predictions))

    if a.save_pred:
        with Path(a.save_pred).open("w") as f:
            for v in videos:
                f.write(json.dumps(dict(file=v["file"], events=predictions[v["file"]]), ensure_ascii=False) + "\n")

    row = dict(
        exp_id=a.exp_id, date=date.today().isoformat(), hypothesis=a.hypothesis, scores=a.scores,
        policy=policy.label(), split=a.split if a.split != "rgkf" else f"rgkf k={a.k} r={a.repeats}",
        objective=(a.objective if not fixed else f"fixed:{a.fixed_thresholds}") + (" +trainscores" if a.train_scores else ""), alpha=a.alpha,
        per_type=not a.no_per_type, shrink=a.shrink, max_events=a.max_events, n_videos=metrics["videos"],
        recall=fmt(metrics["recall"]), recall_ci=fmt(ci.get("recall")), fp_rate=fmt(metrics["fp_rate"]),
        fp_rate_ci=fmt(ci.get("fp_rate")), verdict_acc=fmt(metrics["verdict_acc"]),
        verdict_acc_ci=fmt(ci.get("verdict_acc")), balanced_acc=fmt(metrics["balanced_acc"]),
        type_acc_on_pos=fmt(metrics["type_acc_on_pos"]), type_acc_acceptable=fmt(metrics["type_acc_acceptable"]),
        tc_n=metrics["tc_n"], tc_mae=fmt(metrics["tc_mae"], 2), tc_within_2s=fmt(metrics["tc_within_2s"]),
        event_f1_2s=fmt(metrics["event_f1_2s"]), event_f1_5s=fmt(metrics["event_f1_5s"]),
        roc_auc=fmt(sm["roc_auc"]), pr_auc=fmt(sm["pr_auc"]), runtime_sec=f"{runtime:.1f}",
        decision=a.decision, notes=a.notes + (f" | Platt: brier={sm['brier']:.3f} logloss={sm['log_loss']:.3f} ece={sm['ece']:.3f}" if probs else ""),
    )
    reg = Path(a.registry)
    reg.parent.mkdir(parents=True, exist_ok=True)
    new = not reg.exists()
    with reg.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"\nappended {a.exp_id} to {reg}")


if __name__ == "__main__":
    main()
