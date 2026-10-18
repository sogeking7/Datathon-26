"""Video-level decision policies over per-window gate scores (rows of scores.jsonl / scores_cv.jsonl).

The production rule (app/vj/detector.py) is: per type, take the maximum "yes" probability over
windows; flag types above a per-type threshold; output the top-`max_events` by margin. This module
keeps that rule as `Policy()` (the baseline) and adds alternatives that can be evaluated offline on
cached scores, with calibration statistics fitted on the *training* fold only:

- agg="top2": mean of the two best windows instead of the maximum (less sensitive to one spiky window);
- agg="sidak": per-window empirical p-value against normal training windows, combined over the N
  windows of a video with the Šidák correction 1 - (1 - p_min)^N. The maximum of N noisy window scores
  grows with N, so a fixed threshold flags long normal videos more often; a p-value per video is
  length-invariant under the null. With ~60 reference windows the empirical p-value cannot go below
  1/61, which caps the score of long videos (a 14-window video cannot exceed (60/61)^14 = 0.79), so
  prefer agg="zsidak": the same combination with a Gaussian tail fitted to the logits of normal
  training windows (continuous, no resolution floor);
- calib="bias": subtract, per type, the median logit of normal training windows before aggregation
  (few-shot batch calibration: Zhao et al. 2021, Zhou et al. ICLR 2024), so a question that answers
  "yes" on every video does not dominate;
- max_events > 1 with different types: several violations per video (Х-4 has two).

Threshold fitting reuses the coordinate-ascent search of ml/calibrate.py (global threshold, then one
per-type correction at a time), with selectable objectives and optional shrinkage of per-type
corrections towards the global value (1-5 positives per type overfit easily).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import json

import numpy as np

import evalkit

from app.vj.detector import TypeScore, decide, pick_timecode
from app.vj.queries import BY_CODE, QUERIES

CODES = [q.code for q in QUERIES]
GRID = [round(0.05 * i, 2) for i in range(1, 20)]
# Thresholds for p-value aggregations live near 1: score = (1 - p_min)^N, so 0.99 means "p = 0.01 per video".
PGRID = [round(1 - x, 4) for x in (0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001)]
PVALUE_AGGS = ("sidak", "zsidak")


def grid_for(policy: "Policy") -> list[float]:
    return PGRID if policy.agg in PVALUE_AGGS else GRID


@dataclass(frozen=True)
class Policy:
    agg: str = "max"  # max | top2 | sidak | zsidak
    calib: str = "none"  # none | bias
    max_events: int = 1
    rank: str = "margin"  # margin (baseline: score - threshold) | raw (max raw yes-probability over windows)
    events_from: str = "self"  # self | max: take type/timecode from the baseline max policy (own thresholds) when it fires
    name: str = ""

    def label(self) -> str:
        return self.name or (f"{self.agg}+{self.calib}+e{self.max_events}" + ("+rankraw" if self.rank == "raw" else "")
                             + ("+eventsmax" if self.events_from == "max" else ""))


BASELINE = Policy(name="baseline")


# --- scores ---------------------------------------------------------------------------------------


def load_train_scores(path: str | Path, aug: str = "none") -> dict[str, dict[str, list[dict]]]:
    """scores_cv_train.jsonl -> {fold: {file: [rows]}}."""
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("aug", "none") == aug:
            out[r["fold"]][r["file"]].append(r)
    for fold in out.values():
        for rows in fold.values():
            rows.sort(key=lambda r: r["start"])
    return {k: dict(v) for k, v in out.items()}


def load_scores(path: str | Path, aug: str = "none") -> dict[str, list[dict]]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("aug", "none") == aug:
            by_video[r["file"]].append(r)
    for rows in by_video.values():
        rows.sort(key=lambda r: r["start"])
    return dict(by_video)


def yes_prob(row: dict, code: str) -> float:
    return float(row["gate"][code][BY_CODE[code].yes_index])


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class Calibration:
    """Per-type statistics of normal training windows."""

    bias: dict[str, float] = field(default_factory=dict)  # median logit of the yes-probability
    normal: dict[str, np.ndarray] = field(default_factory=dict)  # sorted yes-probabilities
    mu: dict[str, float] = field(default_factory=dict)  # mean logit of normal windows
    sigma: dict[str, float] = field(default_factory=dict)  # std of the logit of normal windows


def fit_calibration(train_videos: list[dict], by_video: dict[str, list[dict]]) -> Calibration:
    cal = Calibration()
    windows = [r for v in train_videos if not evalkit.is_violation(v) for r in by_video.get(v["file"], [])]
    for code in CODES:
        vals = [yes_prob(r, code) for r in windows if code in r["gate"]]
        if vals:
            lg = np.asarray([logit(p) for p in vals])
            cal.bias[code] = float(np.median(lg))
            cal.normal[code] = np.sort(np.asarray(vals))
            cal.mu[code] = float(lg.mean())
            cal.sigma[code] = float(max(lg.std(ddof=1) if len(lg) > 1 else 1.0, 1e-3))
    return cal


def window_scores(rows: list[dict], code: str, policy: Policy, cal: Calibration) -> list[tuple[tuple[float, float], float]]:
    """Per-window yes-probabilities of one type. calib="bias" shifts the logit by the normal-window median;
    p-value aggregations already compare against the normal-window distribution, so the shift is skipped there."""
    out = []
    for r in rows:
        if code not in r["gate"]:
            continue
        p = yes_prob(r, code)
        if policy.calib == "bias" and code in cal.bias and policy.agg not in PVALUE_AGGS:
            p = sigmoid(logit(p) - cal.bias[code])
        out.append(((r["start"], r["end"]), p))
    return out


def _pvalue(s: float, normal: np.ndarray) -> float:
    """Empirical upper-tail p-value with add-one smoothing: (#normal >= s + 1) / (n + 1)."""
    ge = len(normal) - int(np.searchsorted(normal, s, side="left"))
    return (ge + 1) / (len(normal) + 1)


def _gauss_pvalue(s: float, mu: float, sigma: float) -> float:
    """Upper-tail p-value of logit(s) under a Gaussian fitted to normal-window logits; floored at 1e-9."""
    z = (logit(s) - mu) / sigma
    return max(0.5 * math.erfc(z / math.sqrt(2)), 1e-9)


def type_scores(rows: list[dict], policy: Policy, cal: Calibration) -> dict[str, TypeScore]:
    out: dict[str, TypeScore] = {}
    for code in CODES:
        ws = window_scores(rows, code, policy, cal)
        if not ws:
            continue
        if policy.agg == "max":
            win, s = max(ws, key=lambda x: x[1])
        elif policy.agg == "top2":
            top = sorted(ws, key=lambda x: -x[1])[:2]
            win, s = top[0][0], float(np.mean([t[1] for t in top]))
        elif policy.agg in ("sidak", "zsidak"):
            if policy.agg == "sidak" and code in cal.normal and len(cal.normal[code]):
                pv = [(w, _pvalue(x, cal.normal[code])) for w, x in ws]
            elif policy.agg == "zsidak" and code in cal.mu:
                pv = [(w, _gauss_pvalue(x, cal.mu[code], cal.sigma[code])) for w, x in ws]
            else:  # no calibration data: fall back to the maximum
                pv = None
            if pv is None:
                win, s = max(ws, key=lambda x: x[1])
            else:
                win, p_min = min(pv, key=lambda x: x[1])
                s = (1 - p_min) ** len(ws)  # = 1 - P(at least one of N normal windows exceeds the best one)
        else:
            raise ValueError(policy.agg)
        out[code] = TypeScore(code, float(s), win)
    return out


def predict(rows: list[dict], thresholds: dict, policy: Policy, cal: Calibration):
    """Same contract as calibrate.predict: (type scores, [(code, timecode, score), ...])."""
    scores = type_scores(rows, policy, cal)
    if policy.rank == "raw":
        # Flag with the policy's score, but order (and cap) the flagged types by the strongest raw evidence:
        # p-value scores of several types all sit near 1 and their margins are not a meaningful ranking.
        all_hits = decide(scores, {**thresholds, "max_events": len(scores)})
        raw = {c: max(yes_prob(r, c) for r in rows if c in r["gate"]) for c in scores}
        hits = sorted(all_hits, key=lambda h: -raw[h.code])[: policy.max_events]
    else:
        hits = decide(scores, {**thresholds, "max_events": policy.max_events})
    events = []
    for h in hits:
        row = next(r for r in rows if (r["start"], r["end"]) == h.window)
        g = row["ground"][h.code]
        tc, _ = pick_timecode(h.code, g["times"], g["probs"])
        events.append((h.code, float(tc), h.score))
    return scores, events


def threshold_for(thresholds: dict, code: str) -> float:
    return thresholds.get("per_type", {}).get(code, thresholds["default"])


def video_margin(scores: dict[str, TypeScore], thresholds: dict) -> float:
    """Length- and type-comparable anomaly score of a video: best margin over the per-type threshold."""
    return max((s.score - threshold_for(thresholds, c) for c, s in scores.items()), default=-1.0)


# --- threshold fitting ------------------------------------------------------------------------------


def evaluate(videos, by_video, thr_for, policy: Policy, cal: Calibration):
    predictions = {}
    for v in videos:
        _, events = predict(by_video[v["file"]], thr_for(v), policy, cal)
        predictions[v["file"]] = events
    return evalkit.video_metrics(videos, predictions)


def objective_value(m: dict, objective: str, alpha: float = 0.3) -> float:
    if objective == "f2_type":  # ml/calibrate.py: F2 with precision proxied by 1 - FP rate, plus type bonus
        p, r = 1 - m["fp_rate"], m["recall"]
        f = 5 * p * r / max(4 * p + r, 1e-9)
        return f + 0.25 * m["type_acc_on_pos"]
    if objective == "recall_at_fp":  # highest recall subject to FP rate <= alpha; infeasible points rank below
        feasible = m["fp_rate"] <= alpha + 1e-9
        base = m["recall"] if feasible else m["recall"] - 2 - m["fp_rate"]
        return base + 0.01 * (1 - m["fp_rate"]) + 0.1 * m["type_acc_on_pos"]
    if objective == "youden":
        return m["recall"] - m["fp_rate"] + 0.1 * m["type_acc_on_pos"]
    if objective == "balanced":
        return m["balanced_acc"]
    raise ValueError(objective)


def fit_thresholds(videos, by_video, policy: Policy, cal: Calibration, objective: str = "f2_type",
                   alpha: float = 0.3, per_type: bool = True, shrink: float | None = None,
                   codes=CODES, grid=None) -> dict:
    """Coordinate ascent as in ml/calibrate.py; ties keep the earlier (simpler) candidate.
    The grid follows the policy (GRID for probabilities, PGRID for p-value scores); `shrink` is in grid units
    for p-value policies (number of grid steps), in probability units otherwise."""
    grid = grid or grid_for(policy)

    def near(t, centre):
        if shrink is None:
            return True
        if policy.agg in PVALUE_AGGS:
            return abs(grid.index(t) - grid.index(centre)) <= int(round(shrink * 10))
        return abs(t - centre) <= shrink + 1e-9

    def value(th):
        m, _ = evaluate(videos, by_video, lambda v: th, policy, cal)
        return objective_value(m, objective, alpha)

    best = max(({"default": t, "per_type": {}, "max_events": policy.max_events} for t in grid), key=value)
    if per_type:
        for code in codes:
            local = [t for t in grid if near(t, best["default"])]
            cands = [best] + [{**best, "per_type": {**best["per_type"], code: t}} for t in local]
            best = max(cands, key=value)
    return best


def fit_platt(margins: list[float], labels: list[int], l2: float = 1e-2, iters: int = 100) -> tuple[float, float]:
    """Two-parameter logistic P(violation) = sigmoid(a*margin + b) by Newton's method with a small L2 penalty.
    Platt scaling is the data-efficient calibrator for ~40 points (Niculescu-Mizil & Caruana 2005)."""
    x = np.asarray(margins, dtype=float)
    y = np.asarray(labels, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = a * x + b
        p = 1 / (1 + np.exp(-z))
        g = np.array([np.sum((p - y) * x) + l2 * a, np.sum(p - y)])
        w = p * (1 - p)
        h = np.array([[np.sum(w * x * x) + l2, np.sum(w * x)], [np.sum(w * x), np.sum(w) + 1e-9]])
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def nested_evaluate(videos, by_video, policy: Policy, splits, objective: str = "f2_type", alpha: float = 0.3,
                    per_type: bool = True, shrink: float | None = None, fixed_thresholds: dict | None = None,
                    train_scores: dict | None = None, fold_key=None, platt: bool = False):
    """For each (train, test): fit calibration and thresholds on train, predict test.

    Returns predictions, per-video anomaly margins (for AUC), and the thresholds used per video.
    With fixed_thresholds the threshold search is skipped (e.g. to score the production thresholds.json).
    train_scores: {fold key: by_video of the TRAINING videos scored by that fold's own head}
    (train_head.py scores_cv_train.jsonl); fold_key(test) names the fold. Then calibration and thresholds are
    fitted on scores of the head that is actually applied to the test videos — the protocol that matches
    deployment (train on everything, fit thresholds on the deployed head's own scores)."""
    predictions, margins, used = {}, {}, {}
    probs: dict[str, float] = {}
    base = Policy(max_events=policy.max_events)
    for train, test in splits:
        fit_scores = by_video
        if train_scores is not None:
            key = fold_key(test)
            if key not in train_scores:
                raise KeyError(f"no training scores for fold {key!r}")
            fit_scores = train_scores[key]
        cal = fit_calibration(train, fit_scores)
        th = fixed_thresholds or fit_thresholds(train, fit_scores, policy, cal, objective, alpha, per_type, shrink)
        th_base = (fit_thresholds(train, fit_scores, base, cal, objective, alpha, per_type, shrink)
                   if policy.events_from == "max" else None)
        ab = None
        if platt:
            tm = [video_margin(type_scores(fit_scores[u["file"]], policy, cal), th) for u in train]
            ab = fit_platt(tm, [int(evalkit.is_violation(u)) for u in train])
        for v in test:
            scores, events = predict(by_video[v["file"]], th, policy, cal)
            if events and th_base is not None:
                # verdict from this policy; type and timecode from the baseline rule when it also fires
                _, base_events = predict(by_video[v["file"]], th_base, base, cal)
                events = base_events or events
            predictions[v["file"]] = events
            margins[v["file"]] = video_margin(scores, th)
            used[v["file"]] = th
            if ab is not None:
                probs[v["file"]] = sigmoid(ab[0] * margins[v["file"]] + ab[1])
    if platt:
        return predictions, margins, used, probs
    return predictions, margins, used


def open_set_recall(videos, by_video, policy: Policy, code: str, objective: str = "f2_type", alpha: float = 0.3,
                    per_type: bool = True, shrink: float | None = None) -> dict:
    """Leave-one-type-out on cached scores: calibrate the other gates without videos of `code`,
    then measure how often the held-out type's videos are still flagged (by any gate) and the FP rate
    on the normals of the training fold (optimistic, in-sample) — a cheap probe of open-set behaviour
    of the current gates before any catch-all query exists."""
    train, test, unknown = evalkit.leave_one_type_out(videos, code)
    other_codes = [c for c in CODES if c != code]
    cal = fit_calibration(train, by_video)
    restricted = replace(policy)
    th = fit_thresholds(train, by_video, restricted, cal, objective, alpha, per_type, shrink, codes=other_codes)
    flagged, by_gate = 0, defaultdict(int)
    for v in test:
        if v["file"] not in unknown:
            continue
        scores, events = predict(by_video[v["file"]], th, restricted, cal)
        events = [e for e in events if e[0] != code]  # the held-out gate does not exist
        if events:
            flagged += 1
            by_gate[events[0][0]] += 1
    m, _ = evaluate(train, by_video, lambda v: th, restricted, cal)
    return dict(code=code, unknown_videos=len(unknown), flagged_by_other_gates=flagged,
                open_set_recall=flagged / max(len(unknown), 1), flagged_as=dict(by_gate),
                train_fp_rate=m["fp_rate"], train_recall=m["recall"])
