# Final improvement report — 2026-10-18

All numbers are on the 40 unique labelled videos (21 violations, 19 normals). Each one comes from an executed row of
`reports/experiments.csv`, which holds intervals, the exact configuration and the decision. Rows marked *in-sample*
score the deployed head on videos it was trained on and are optimistic. Differences of one to three videos are within
the run-to-run noise of head training (F16). The decision rule (experiment plan §1) keeps a change only if the paired
bootstrap gives P(better) ≥ 0.9 on balanced accuracy, recall drops by at most one video, and leave-one-setup-out does
not get worse.

## 1. Original baseline

| Quantity | Value | Row |
|---|---|---|
| Model | VideoJev-4B, FP16 copy on one T4, frozen backbone, 2.6 M-parameter head fine-tuned on cached hidden states | `ml/README.md` |
| Decision | max over 20 s windows (stride 10 s), per-type thresholds (0.45; UNATTENDED_CASH 0.25, NO_COUNTER 0.70, UNAUTHORIZED_PERSON 0.05, CASH_EXPOSED 0.15), one event | `ml/artifacts/thresholds.json` |
| Verdict, nested LOVO, thresholds fitted on fold-head scores | recall 76% [57–95], FP 37% [16–58], accuracy 70%, balanced 0.697 | **E00** |
| Same protocol, leave-one-setup-out thresholds | 76% / 42% / 67.5%, balanced 0.670 | E00-setup |
| Head and thresholds without the test session | 81% / 58% / 62.5%, balanced 0.615 | **E06h** |
| **What the live service did until today** (head_v1 + production thresholds + MIN_CONFIDENCE filter, in-sample) | recall 67%, FP 21% | **D0-live-before** |
| Same pair without the filter (in-sample) | recall 76%, FP 53% | E14h-none |

Four defects separated the live service from the evaluated rule:
- **F1:** the confidence filter dropped hits with scores between the threshold and 0.5.
- **F15:** the thresholds were tuned on 40 fold heads, not on the deployed head.
- **Grounding context (new today, see §2):** live timecodes came from a different question context than offline.
- **Evidence frames (new today, see §2):** jobs failed on Х-23-like files when an evidence frame could not be read.

## 2. Changes implemented and deployed

| Change | Files | Evidence |
|---|---|---|
| Confidence contract fix. A detection that passes its threshold is no longer dropped by `analyzer.MIN_CONFIDENCE`. Confidence is now the margin mapped to [0.5, 1]. | `service/app/vj/detector.py` | test `test_confidence_from_margin_survives_analyzer_filter`; E01a vs E01b |
| **Thresholds refit on the deployed head's own scores** (max rule), which removes the F15 mismatch | `ml/artifacts/thresholds_head_v1_refit.json` (`ml/export_policy.py runs/head_v1/scores_augs.jsonl --policy baseline`) | D1-config-A |
| **Grounding asked in the scored context.** VideoJev packs all questions into one causal context. Offline scoring, head training and the localization metrics asked each grounding question after the 8 gate questions. The service asked it alone, so the curve was different. Live Х-20 returned 0.2 s; offline gives 4.25 s; the label is 5 s. `detector.scoring_questions()` is now shared by the service and `ml/score_dataset.py`. | `service/app/vj/detector.py`, `ml/score_dataset.py` | test `test_grounding_is_asked_in_the_scored_context`; live timecodes now equal offline to 0.05 s (§4) |
| **Evidence frame robustness.** Seeking by msec fails after about 11 s on part of the .mov files (Х-23), and the whole job failed with "Не удалось прочитать кадр". Fallbacks: seek by frame index, then sequential read. An unreadable frame is now skipped instead of failing the analysis. | `service/app/video.py`, `service/app/analyzer.py` | tests `test_grab_frame_falls_back_when_msec_seek_fails`, `test_unreadable_evidence_frame_does_not_fail_analysis`; Х-23 failed live on 10-17 and at 05:39 today, and passed after the fix |
| Paired bootstrap reported P(better) for FP rate with the wrong sign (it counted a higher FP rate as better). Decisions were unaffected, since they use balanced accuracy. | `ml/evalkit.py` | test `test_paired_bootstrap_fp_rate_lower_is_better` |
| Earlier today: deployment-consistent nested protocol (`scores_cv_train.jsonl`, `--train-scores`), batched trainer, `zsidak+eventsmax` policy in the service (selectable, **not deployed**), evaluation toolkit | `ml/`, `service/app/vj/detector.py` | rows N-*, E03e, unit tests |

Nothing changed in the prompts, windows, model or head weights (`/srv/models/head.pt` md5 `c7d63317…`, the release
head).

## 3. Before / after (executed rows)

### 3.1 Deployed configurations, scored on the deployed head (in-sample)

| Row | Configuration | Recall | FP | Accuracy | Balanced | Type | ±2 s |
|---|---|---|---|---|---|---|---|
| D0-live-before | live until 2026-10-18: head_v1 + `thresholds.json` + MIN_CONFIDENCE filter | 67% | 21% | 72.5% | 0.728 | 62% | 62% |
| E14h-none | same pair without the filter | 76% | 53% | 62.5% | 0.618 | 62% | 62% |
| **D1-config-A** | **deployed now: head_v1 + `thresholds_head_v1_refit.json` + confidence fix** | **86%** | **32%** | **77.5%** | **0.771** | **76%** | 50% |
| D2-config-B | head_v1 + `thresholds_v2.json` (zsidak+eventsmax) | 95% | 53% | 72.5% | 0.713 | 76% | 50% |

D2 vs D1, paired: balanced −0.056, P(B better) = 0.20. B fixes 4 videos (Х-8, Х-14, Х-32, Х-34) and breaks 6 (Х-3, Х-12, Х-19, Х-31, Х-37, Х-39).
D1 is fitted on these same scores, so its absolute numbers are optimistic. The honest estimate of the recipe is N-base below.

### 3.2 Honest estimates: nested, thresholds on each fold head's own training scores (deployment recipe)

Every head below was retrained on the T4 today with the same recipe as production: 40 fold heads, plus thresholds
fitted on each fold head's own training scores. Reference row: **N-base**.

| Row | Head / policy | Recall | FP | Accuracy | Balanced | P(bal better) vs ref | Leave-one-setup-out balanced | Decision |
|---|---|---|---|---|---|---|---|---|
| **N-base** | default head, max rule (= recipe of config A) | 76% | 53% | 62.5% | **0.618** | ref | 0.670 (N-base-setup) | **keep** |
| N-base-z | default head, zsidak+eventsmax (= recipe of B) | 71% | 58% | 57.5% | 0.568 | 0.23 | 0.742 (N-base-z-setup, thresholds only) | reject |
| N-soup | 3-seed soup + feature noise | 71% | 47% | 62.5% | 0.620 | 0.52 | 0.618 | reject |
| N-soup-z | soup + zsidak | 71% | 53% | 60% | 0.594 | 0.35 vs N-soup | 0.687 | reject |
| N-alt | alt-type windows masked | 62% | 53% | 55% | 0.546 | 0.18 | 0.565 | reject |
| N-alt-z | alt-masked + zsidak | 76% | 58% | 60% | 0.591 | 0.72 vs N-alt | 0.660 | reject |
| N-altsoup | alt-masked 3-seed soup | 71% | 47% | 62.5% | 0.620 | 0.52 | 0.618 | reject |
| N-altsoup-z | alt-masked soup + zsidak | 81% | 58% | 62.5% | 0.615 | 0.44 vs N-altsoup | 0.639 | reject |

The setup column uses LOVO fold scores with thresholds fitted leave-one-setup-out, so the head still saw the session.
The only true leave-one-setup-out pair is E06h (max, balanced 0.615) vs E06h+E03e (zsidak, 0.563).

### 3.3 Choosing between A and B

| Evidence | A (max) | B (zsidak+eventsmax) | Rule outcome |
|---|---|---|---|
| Fold-head LOVO (E00 vs E03e) | 0.697 | 0.768 | P = 0.79 < 0.9 → not kept |
| Deployment-recipe LOVO (N-base vs N-base-z) | 0.618 | 0.568 | B worse, recall −1 |
| Thresholds-only setup split (N-base-setup vs N-base-z-setup) | 0.670 | 0.742 | B better (P = 0.84) |
| **True leave-one-setup-out, session-blind head (E06h vs E06h+E03e)** | **0.615** | **0.563** | **B worse: zsidak does not help on an unseen session** |
| In-sample on the deployed head (D1 vs D2) | 0.771 | 0.713 | P = 0.20 |

B looks better only where the head has already seen the camera session (E03e, setup split with fold scores). Under
the recipe that is actually deployed (N-base-z), and on a session the head never saw (E06h+E03e), it is worse. The
Gaussian calibration of normal windows does not transfer to a new session. The jury films from a different angle,
so this is the case that matters. **Chosen: configuration A.** No retrained head (C) passes the rule: the best,
N-soup and N-altsoup, tie N-base on balanced accuracy (P = 0.52), lose one violation, and are worse on the setup split.

### 3.4 Honest summary of the deployed configuration

Two figures are the honest ones to quote:
- **Familiar camera** (N-base): recall 76% [57–95], FP 53% [32–74], accuracy 62.5%, type 43%, timecode within 2 s 56%.
- **New session** (E06h): recall 81%, FP 58%, accuracy 62.5%.

Today's gain is mostly in the delivered behaviour, not in the model:
- the service now runs the rule that was evaluated;
- it returns the timecodes that were evaluated;
- it no longer fails on videos whose late frames cannot be seeked.

Against the old live behaviour (D0), recall rises from 67% to 86% and FP from 21% to 32% on the same deployed head
(in-sample).

## 4. Live verification (public API, 2026-10-18)

Each video was posted through `http://3.83.152.128/api/v1/analyze` with the X-API-Key from `/opt/kassa/gpu.env`, via
the frontend proxy on :80. The client ran on the host, so upload time is about 0. The final run was made with the GPU
free of training jobs. Logs are in `~/videojev/verify_api_A_*.log` on the host.

| Video | Label | Verdict | Type | Timecode (label) | Confidence | Total time | Offline D1 prediction |
|---|---|---|---|---|---|---|---|
| Х-20 (11.6 s) | violation, SAFE_OPEN @ 5 s | violation ✓ | SAFE_OPEN ✓ | 4.2 s (5 s) ✓ | 0.84 | **14.1 s** | SAFE_OPEN 4.25 s, score 0.821 |
| Х-23 (18.3 s) | normal (hard) | violation ✗ | UNATTENDED_CASH | 14.2 s | 0.61 | **23.1 s** | UNATTENDED_CASH 14.25 s, 0.727 |
| Х-14 (142.5 s, 14 windows) | normal | violation ✗ | NO_COUNTER | 10.2 s | 0.66 | **154.8 s** (< 300 s) | NO_COUNTER 10.25 s, 0.621 |

Live output equals config A's offline predictions on all three videos (type, window, score, timecode to 0.05 s).
Х-23 and Х-14 are two of the six in-sample false positives that config A is known to produce (D1: 6 of 19 normals).
They are model errors, not deployment errors.

Earlier runs today, recorded for completeness:

| Run (UTC) | Х-20 | Х-23 | Х-14 |
|---|---|---|---|
| 05:39, config A before the frame fix, GPU shared with 3 training jobs | 18.3 s, timecode 0.2 s | failed: "Не удалось прочитать кадр на 11.2 с" | 218 s |
| 05:46, after the frame fix | 18.3 s, timecode 0.2 s | 31.4 s, done | 219 s |
| 05:53, after the grounding-context fix | 18.2 s, timecode 4.2 s | 31.3 s, timecode 14.2 s | — |

Grounding with the full question list costs about 1–2 s per detected event under the same load (05:46 vs 05:53 runs, both with training on the GPU: Х-20 7.9 → 9.1 s, Х-23 12.8 → 14.5 s). Х-14
takes 138 s in the gates on a free T4 and about 192 s when the GPU is shared. The 280 s job timeout still counts queue
time (F9), so two long uploads back to back can still fail.

## 5. Deployment state and rollback

The live configuration and its rollback copies:

| Item | Value |
|---|---|
| Container | `kassa`, image `kassa-gpu` (built 2026-10-18 from `service/`, also in `/opt/kassa/src`), `--network kassa-net`, `-v /srv/models:/models:ro -v /opt/kassa/data:/data`, `--env-file /opt/kassa/gpu.env` |
| Head | `/srv/models/head.pt` = release head_v1 (unchanged) |
| Thresholds | `VIDEOJEV_THRESHOLDS=/models/thresholds_head_v1_refit.json` = `{"policy": "max", "default": 0.45, "per_type": {"UNATTENDED_CASH": 0.65, "DOOR_OPEN": 0.40}, "max_events": 1}` |
| Rollback copies | `/srv/models/head.pt.v1`, `/srv/models/thresholds.json.v1` (and the original `thresholds.json`), `/opt/kassa/gpu.env.v1`, image tag `kassa-gpu:v1-20261017`, stopped container `kassa-v1-old`, `/opt/kassa/src.bak-20261018` |

## 6. Remaining problems

1. **False positives on normals stay high:** 53% on the familiar-camera recipe (N-base) and 58% on a new session
   (E06h). The hard normals involve an empty desk or a closed door (Х-3, Х-19, Х-23, Х-29, Х-31, Х-37). They fire
   UNATTENDED_CASH or DOOR_OPEN under every policy. Long normals fire NO_COUNTER (Х-14).
2. **Head-training variance (F16).** Identical runs flip 11 of 40 verdicts. Soups did not reduce it measurably at
   n = 40.
3. **Session dependence of the head (E06h).** Only more recording sessions, such as self-recorded clips from other
   angles, can fix this.
4. **Closed set.** Generic "needs review" queries (E11) were not run.
5. **Timecode accuracy rests on 9 labelled videos.** The coarse-to-fine pass (E13) was not run.
6. **Confidence is only margin-mapped.** Platt calibration (E07) was measured (Brier 0.20, ECE 0.05) but not wired
   into the service.
7. **Queue time and the timeout (F9).** The job timeout counts queue time, so set `WORKERS=1` or show the queue
   position for back-to-back long uploads.

## 7. Reproduction

```bash
scripts/fetch_runs.sh ubuntu@<host> /home/ubuntu/videojev/runs ~/Downloads/vockey.pem
H=runs/n_base; python ml/experiment.py --scores $H/scores_cv.jsonl --train-scores $H/scores_cv_train.jsonl --exp-id N-base --save-pred runs/pred/N-base.jsonl
python ml/experiment.py --scores $H/scores_cv.jsonl --train-scores $H/scores_cv_train.jsonl --agg zsidak --events-from max --exp-id N-base-z --compare runs/pred/N-base.jsonl
python ml/experiment.py --scores runs/head_v1/scores_augs.jsonl --fixed-thresholds ml/artifacts/thresholds_head_v1_refit.json --exp-id D1-config-A
python ml/export_policy.py runs/head_v1/scores_augs.jsonl --policy baseline --out ml/artifacts/thresholds_head_v1_refit.json
```
To retrain a head on the host, run this with `[flags]` from `train_args.json` of each `runs/n_*` (6–15 min on a free T4):
`docker run --rm --gpus all -v ~/videojev:/work -v /srv/models:/models:ro kassa-gpu python /work/proj_audit/ml/train_head.py --cache /work/runs/full/cache --model /models/videojev-4b-fp16 --scores /work/runs/full/scores.jsonl --out /work/runs/<name> [flags]`
