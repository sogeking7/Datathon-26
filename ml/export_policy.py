"""Export a decision policy fitted on all labelled videos into a thresholds.json the service can load.

The first-release file holds {"default", "per_type", "max_events"} for the max-over-windows rule.
Policy "zsidak+eventsmax" (experiment E03e, reports/experiments.csv) adds what the detector needs to
reproduce it at inference time:

  python ml/export_policy.py runs/head/scores_cv.jsonl --policy zsidak+eventsmax --out runs/head/thresholds_v2.json

Output keys: policy, default/per_type (thresholds on the Šidák score (1 - p_min)^N), calibration
{code: {mu, sigma}} of normal-window logits, base {default, per_type} for the max rule used to pick
type and timecode, max_events. Fitting on all 40 videos is the production fit (optimistic in-sample);
the honest numbers are the nested cross-validation rows in the registry.
"""

import argparse
import json
from pathlib import Path

from common import load_labels

import decision


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scores")
    p.add_argument("--policy", default="zsidak+eventsmax", choices=["baseline", "zsidak+eventsmax"])
    p.add_argument("--objective", default="f2_type")
    p.add_argument("--max-events", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    by_video = decision.load_scores(a.scores)
    videos = [v for v in load_labels() if v["file"] in by_video]
    cal = decision.fit_calibration(videos, by_video)
    if a.policy == "baseline":
        th = decision.fit_thresholds(videos, by_video, decision.Policy(max_events=a.max_events), cal, a.objective)
        out = {**th, "policy": "max"}
    else:
        pol = decision.Policy(agg="zsidak", events_from="max", max_events=a.max_events)
        th = decision.fit_thresholds(videos, by_video, pol, cal, a.objective)
        base = decision.fit_thresholds(videos, by_video, decision.Policy(max_events=a.max_events), cal, a.objective)
        out = dict(policy="zsidak+eventsmax", default=th["default"], per_type=th["per_type"], max_events=a.max_events,
                   calibration={c: {"mu": cal.mu[c], "sigma": cal.sigma[c]} for c in cal.mu},
                   base={"default": base["default"], "per_type": base["per_type"]})
    m, _ = decision.evaluate(videos, by_video, lambda v: th, decision.Policy() if a.policy == "baseline"
                             else decision.Policy(agg="zsidak", events_from="max", max_events=a.max_events), cal)
    print("fit on all videos (optimistic):", {k: round(m[k], 3) for k in ("recall", "fp_rate", "verdict_acc", "type_acc_on_pos")})
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
