"""Evaluation toolkit for the 40-video dataset (numpy only, no torch, runs on a laptop).

Why a separate module: calibrate.py reports point estimates on leave-one-video-out (LOVO) folds.
With 21 positives and 19 normals one video moves recall by ~5 points, twin recordings (a violation
and its normal re-take from the same camera session) sit on both sides of a LOVO split, and one
timecode per video does not define an interval. This module adds:

- confidence intervals: Wilson for proportions, bootstrap over videos (stratified by verdict),
  paired bootstrap for A/B comparisons on the same videos;
- group-aware splits: leave-one-setup-out (camera session), repeated group k-fold,
  leave-one-type-out for open-set experiments;
- event-level matching with temporal tolerances and alternative-type-aware type accuracy;
- score-level metrics: ROC-AUC, average precision, Brier, log loss, ECE.

Predictions are `{file: [(code, timecode_sec, score), ...]}`, events ordered as the service would
output them (highest margin first). Labels come from common.load_labels().
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator

import numpy as np

Event = tuple[str, float, float]  # (type code, timecode_sec, score)
Predictions = dict[str, list[Event]]


# --- labels -------------------------------------------------------------------------------------


def is_violation(video: dict) -> bool:
    return video["verdict"] == "violation"


def label_events(video: dict) -> list[dict]:
    return list(video.get("events", [])) if is_violation(video) else []


def acceptable_codes(video: dict) -> dict[str, float]:
    """code -> labelled timecode for the primary type of each event and its alternative types."""
    out: dict[str, float] = {}
    for e in label_events(video):
        for code in [e["type_code"], *e.get("alt_type_codes", [])]:
            out.setdefault(code, e["timecode_sec"])
    return out


def primary_codes(video: dict) -> dict[str, float]:
    return {e["type_code"]: e["timecode_sec"] for e in label_events(video)}


# --- event matching -------------------------------------------------------------------------------


def match_events(pred: list[Event], truth: list[dict], tol: float, with_alt: bool = True):
    """Greedy one-to-one matching of predicted events to labelled events.

    A prediction matches a labelled event when the type is the event's primary type (or one of its
    alternative types when with_alt) and |t_pred - t_label| <= tol. Predictions are matched in
    descending score order, each to the nearest free event. Returns (matches, unmatched_pred,
    unmatched_truth) where matches are (pred_index, truth_index, abs_error).
    """
    used: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for i in sorted(range(len(pred)), key=lambda i: -pred[i][2]):
        code, tc, _ = pred[i]
        best = None
        for j, e in enumerate(truth):
            if j in used:
                continue
            codes = {e["type_code"]} | (set(e.get("alt_type_codes", [])) if with_alt else set())
            if code not in codes:
                continue
            err = abs(tc - e["timecode_sec"])
            if err <= tol and (best is None or err < best[1]):
                best = (j, err)
        if best is not None:
            used.add(best[0])
            matches.append((i, best[0], best[1]))
    unmatched_pred = [i for i in range(len(pred)) if i not in {m[0] for m in matches}]
    unmatched_truth = [j for j in range(len(truth)) if j not in used]
    return matches, unmatched_pred, unmatched_truth


# --- video-level metrics --------------------------------------------------------------------------


def _safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


def _fbeta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / max(b2 * precision + recall, 1e-12)


def video_metrics(videos: list[dict], predictions: Predictions, tols: Iterable[float] = (1.0, 2.0, 5.0)):
    """Verdict, type, timecode and event-level metrics. Returns (metrics, per-video rows).

    Baseline-compatible keys (same definitions as ml/calibrate.py): recall, fp_rate, verdict_acc,
    type_acc_on_pos (first event's type == a primary labelled type), tc_mae and tc_within_2s over
    videos whose first event's type is correct. Extended keys are documented inline.
    """
    tols = tuple(tols)
    n_pos = n_neg = tp = fp = 0
    type_primary = type_accept = 0
    tc_err: list[float] = []
    tc_err_accept: list[float] = []
    ev = {t: Counter() for t in tols}
    multi_total = multi_full = 0
    rows = []
    for v in videos:
        pred = list(predictions.get(v["file"], []))
        pos = is_violation(v)
        flagged = bool(pred)
        n_pos += pos
        n_neg += not pos
        tp += pos and flagged
        fp += (not pos) and flagged
        events = label_events(v)
        first = pred[0] if pred else None
        row = dict(file=v["file"], verdict=v["verdict"], flagged=flagged, label=primary_codes(v),
                   pred=[(c, round(t, 1), round(s, 3)) for c, t, s in pred], type_ok=None, tc_err=None)
        if pos and first is not None:
            prim, acc = primary_codes(v), acceptable_codes(v)
            row["type_ok"] = first[0] in prim
            row["type_acceptable"] = first[0] in acc
            if first[0] in prim:
                type_primary += 1
                tc_err.append(abs(first[1] - prim[first[0]]))
                row["tc_err"] = round(tc_err[-1], 2)
            if first[0] in acc:
                type_accept += 1
                tc_err_accept.append(abs(first[1] - acc[first[0]]))
        for t in tols:
            m, up, ut = match_events(pred, events, t)
            ev[t].update(tp=len(m), fp=len(up), fn=len(ut))
        if len(events) > 1:
            multi_total += 1
            m, _, _ = match_events(pred, events, 10.0)
            multi_full += len(m) == len(events)
        rows.append(row)

    n = len(videos)
    tn, fn = n_neg - fp, n_pos - tp
    recall = tp / max(n_pos, 1)
    fp_rate = fp / max(n_neg, 1)
    precision = tp / max(tp + fp, 1)
    out: dict = dict(
        videos=n, n_pos=n_pos, n_neg=n_neg, tp=tp, fp=fp, fn=fn, tn=tn,
        recall=recall, fp_rate=fp_rate, precision=precision,
        f1=_fbeta(precision, recall, 1.0), f2=_fbeta(precision, recall, 2.0),
        balanced_acc=(recall + (1 - fp_rate)) / 2,
        verdict_acc=(tp + tn) / max(n, 1),
        # type: denominators as in calibrate.py (all positives) and among detected positives
        type_acc_on_pos=type_primary / max(n_pos, 1),
        type_acc_acceptable=type_accept / max(n_pos, 1),
        type_acc_on_detected=_safe_div(type_primary, tp),
        # timecode over videos whose first event's type is the primary labelled type
        tc_n=len(tc_err),
        tc_mae=_safe_div(sum(tc_err), len(tc_err)),
        tc_median=float(np.median(tc_err)) if tc_err else None,
        tc_within_1s=_safe_div(sum(e <= 1 for e in tc_err), len(tc_err)),
        tc_within_2s=_safe_div(sum(e <= 2 for e in tc_err), len(tc_err)),
        tc_within_5s=_safe_div(sum(e <= 5 for e in tc_err), len(tc_err)),
        # the same with alternative types accepted (labels.json alt_type_codes)
        tc_within_2s_acceptable=_safe_div(sum(e <= 2 for e in tc_err_accept), len(tc_err_accept)),
        multi_event_videos=multi_total,
        multi_event_full_recall=_safe_div(multi_full, multi_total),
    )
    for t in tols:
        c = ev[t]
        p_ = c["tp"] / max(c["tp"] + c["fp"], 1)
        r_ = c["tp"] / max(c["tp"] + c["fn"], 1)
        key = f"{t:g}s"
        out[f"event_precision_{key}"] = p_
        out[f"event_recall_{key}"] = r_
        out[f"event_f1_{key}"] = _fbeta(p_, r_, 1.0)
    return out, rows


def confusion_by_type(videos: list[dict], predictions: Predictions) -> dict[str, Counter]:
    """primary labelled type (or 'normal') -> Counter of predicted first-event types ('none' = not flagged)."""
    out: dict[str, Counter] = defaultdict(Counter)
    for v in videos:
        pred = predictions.get(v["file"], [])
        key = ",".join(sorted(primary_codes(v))) or "normal"
        out[key][pred[0][0] if pred else "none"] += 1
    return dict(out)


def breakdown(videos: list[dict], predictions: Predictions, key_of: Callable[[dict], str],
              keys=("videos", "n_pos", "n_neg", "recall", "fp_rate", "verdict_acc", "type_acc_on_pos")) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for v in videos:
        groups[key_of(v)].append(v)
    out = {}
    for g in sorted(groups):
        m, _ = video_metrics(groups[g], predictions)
        out[g] = {k: m[k] for k in keys}
    return out


# --- uncertainty ----------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _resample(videos: list[dict], rng: random.Random) -> list[dict]:
    pos = [v for v in videos if is_violation(v)]
    neg = [v for v in videos if not is_violation(v)]
    return [rng.choice(pos) for _ in pos] + [rng.choice(neg) for _ in neg]


LOWER_IS_BETTER = {"fp_rate", "tc_mae"}
BOOT_KEYS = ("recall", "fp_rate", "verdict_acc", "balanced_acc", "f2", "type_acc_on_pos",
             "type_acc_acceptable", "tc_within_2s", "event_f1_2s", "event_f1_5s")


def bootstrap_ci(videos: list[dict], predictions: Predictions, n_boot: int = 2000, seed: int = 0,
                 keys: Iterable[str] = BOOT_KEYS, level: float = 0.95) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap over videos, resampling positives and normals separately.

    Out-of-fold predictions are fixed per video, so this captures sampling uncertainty of the video
    set, not the variance of refitting. It cannot capture the bias of selecting prompts and
    hyperparameters on the same 40 videos (see model_audit.md)."""
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        m, _ = video_metrics(_resample(videos, rng), predictions)
        for k in keys:
            if m.get(k) is not None:
                samples[k].append(m[k])
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {k: (float(np.percentile(s, lo)), float(np.percentile(s, hi))) for k, s in samples.items() if s}


def paired_bootstrap(videos: list[dict], pred_a: Predictions, pred_b: Predictions, n_boot: int = 2000,
                     seed: int = 0, keys: Iterable[str] = BOOT_KEYS) -> dict[str, dict]:
    """Bootstrap of metric(B) - metric(A) on the same resampled videos; also the share of resamples with B > A."""
    rng = random.Random(seed)
    diffs: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        vs = _resample(videos, rng)
        ma, _ = video_metrics(vs, pred_a)
        mb, _ = video_metrics(vs, pred_b)
        for k in keys:
            if ma.get(k) is not None and mb.get(k) is not None:
                diffs[k].append(mb[k] - ma[k])
    out = {}
    for k, d in diffs.items():
        arr = np.asarray(d)
        sign = -1.0 if k in LOWER_IS_BETTER else 1.0  # FP rate: B is better when its FP rate is lower
        out[k] = dict(mean=float(arr.mean()), ci=(float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))),
                      p_b_better=float((sign * arr > 0).mean()), p_a_better=float((sign * arr < 0).mean()))
    return out


def flips(videos: list[dict], pred_a: Predictions, pred_b: Predictions) -> dict[str, list[str]]:
    """Videos whose verdict correctness changes between A and B (for McNemar-style reporting)."""
    out = {"fixed_by_b": [], "broken_by_b": []}
    for v in videos:
        ok_a = bool(pred_a.get(v["file"])) == is_violation(v)
        ok_b = bool(pred_b.get(v["file"])) == is_violation(v)
        if ok_b and not ok_a:
            out["fixed_by_b"].append(v["file"])
        if ok_a and not ok_b:
            out["broken_by_b"].append(v["file"])
    return out


# --- score-level metrics (video anomaly score vs verdict) ----------------------------------------


def roc_auc(y: list[int], s: list[float]) -> float | None:
    y_arr, s_arr = np.asarray(y), np.asarray(s, dtype=float)
    pos, neg = s_arr[y_arr == 1], s_arr[y_arr == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # Mann-Whitney U with ties counted as half
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def average_precision(y: list[int], s: list[float]) -> float | None:
    y_arr, s_arr = np.asarray(y), np.asarray(s, dtype=float)
    if y_arr.sum() == 0:
        return None
    order = np.argsort(-s_arr, kind="stable")
    y_sorted = y_arr[order]
    hits = np.cumsum(y_sorted)
    precision_at = hits / np.arange(1, len(y_sorted) + 1)
    return float((precision_at * y_sorted).sum() / y_arr.sum())


def brier(y: list[int], p: list[float]) -> float:
    y_arr, p_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p_arr - y_arr) ** 2))


def log_loss(y: list[int], p: list[float], eps: float = 1e-6) -> float:
    y_arr, p_arr = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_arr * np.log(p_arr) + (1 - y_arr) * np.log(1 - p_arr)))


def ece(y: list[int], p: list[float], bins: int = 5) -> float:
    """Expected calibration error with equal-width bins (few bins: 40 videos)."""
    y_arr, p_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p_arr >= lo) & (p_arr < hi if hi < 1 else p_arr <= hi)
        if sel.any():
            total += sel.mean() * abs(p_arr[sel].mean() - y_arr[sel].mean())
    return float(total)


def score_metrics(videos: list[dict], video_score: dict[str, float], video_prob: dict[str, float] | None = None) -> dict:
    """Ranking and calibration quality of a per-video anomaly score (and optional probability)."""
    y = [int(is_violation(v)) for v in videos]
    s = [video_score[v["file"]] for v in videos]
    out = dict(roc_auc=roc_auc(y, s), pr_auc=average_precision(y, s))
    if video_prob is not None:
        p = [video_prob[v["file"]] for v in videos]
        out.update(brier=brier(y, p), log_loss=log_loss(y, p), ece=ece(y, p))
    return out


# --- splits ---------------------------------------------------------------------------------------

Split = tuple[list[dict], list[dict]]


def lovo(videos: list[dict]) -> Iterator[Split]:
    for v in videos:
        yield [u for u in videos if u is not v], [v]


def leave_one_group_out(videos: list[dict], group_of: Callable[[dict], str]) -> Iterator[Split]:
    for g in sorted({group_of(v) for v in videos}):
        test = [v for v in videos if group_of(v) == g]
        yield [v for v in videos if group_of(v) != g], test


def repeated_group_kfold(videos: list[dict], group_of: Callable[[dict], str], k: int = 5,
                         repeats: int = 10, seed: int = 0) -> Iterator[Split]:
    """Groups are shuffled and dealt round-robin into k folds; repeated with different seeds."""
    groups = sorted({group_of(v) for v in videos})
    for r in range(repeats):
        rng = random.Random(seed + r)
        order = groups[:]
        rng.shuffle(order)
        folds = [set(order[i::k]) for i in range(k)]
        for fold in folds:
            test = [v for v in videos if group_of(v) in fold]
            if test:
                yield [v for v in videos if group_of(v) not in fold], test


def leave_one_type_out(videos: list[dict], code: str) -> tuple[list[dict], list[dict], set[str]]:
    """Train without any video showing `code` (primary or alternative); test on all videos.

    The videos removed from training are the 'unknown type' set: at test time, the question is whether
    the system still flags them (open-set recall) and how, since no gate was calibrated for them."""
    involved = {v["file"] for v in videos if code in acceptable_codes(v)}
    train = [v for v in videos if v["file"] not in involved]
    return train, list(videos), involved


def split_report(splits: Iterable[Split]) -> dict:
    """Sanity numbers for a split scheme: folds, mean test size, positives per test fold."""
    sizes, pos = [], []
    for train, test in splits:
        assert not {v["file"] for v in train} & {v["file"] for v in test}, "train/test overlap"
        sizes.append(len(test))
        pos.append(sum(is_violation(v) for v in test))
    return dict(folds=len(sizes), mean_test=float(np.mean(sizes)) if sizes else 0.0,
                min_test_pos=min(pos) if pos else 0, max_test_pos=max(pos) if pos else 0)
