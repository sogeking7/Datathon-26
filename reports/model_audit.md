# Model audit — ForteBank Datathon 26 cash-discipline detector

Date: 2026-10-18. Scope: everything in this repository at commit `2bba908` plus the uncommitted
`reports/`, `research_notes/`, `design/` folders. Audited by reading the code, the dataset files, the
labels, the VideoJev-4B remote code and paper, and by running local analyses on the 42 videos
(`ml/analysis/`). What could **not** be done in this session: run the model. The GPU server that holds
the FP16 model copy, the hidden-state cache and the `scores*.jsonl` files was not reachable from this
session (the session's permission policy blocked the SSH command as a production read), and the laptop
has no CUDA. Every number below is either read from the code/labels/videos or quoted from
`ml/README.md` with the uncertainty it deserves.

## 1. System as built

```
.mov ─► PyAV decode on an exact 2 fps grid, native 464×848 (app/vj/runtime.py decode)
     ─► windows: whole clip if ≤ 26 s, else 20 s windows with 10 s stride (runtime.windows)
     ─► VideoJev-4B (Qwen3.5-4B hybrid backbone, FP16 copy for the T4, SDPA patched to EFFICIENT_ATTENTION)
          encode_video once per window; one joint forward for the 8 gate questions (processing.prepare_query_groups
          packs all questions that fit the 16 384-token context)
     ─► gate = single-choice question per violation type; option A = violation, B = normal behaviour, C = did not arise
          score(type) = max over windows of P(A)            (detector.gate_scores)
     ─► per-type thresholds from ml/artifacts/thresholds.json, top-1 by margin (max_events = 1)  (detector.decide)
     ─► for the chosen type: grounding query on the best window → timecode = peak (events) or onset at half-peak (states)
     ─► rules.CATALOG → title, rationale, verified NBRK clauses; evidence frames via OpenCV; JSON per the ToR §10
```

| Component | Where | Facts |
|---|---|---|
| Backbone | `ZhuYuhan/VideoJev-4B` rev `2fc6218`, Apache-2.0 | Qwen3.5-4B: 32 text layers (24 Gated-DeltaNet linear-attention + 8 full-attention, `full_attention_interval=4`), hidden 2560; ViT 24 layers / 1024; patch 16, spatial merge 2, temporal patch 2 → one token per 32×32 px per frame pair; last layer replaced by an attention-only readout; `CosineDecisionHead` (q,k: 2560→512, scale 10, learned scalar bias) scores `<|option_end|>` states against the `<|decide|>` state ([paper](https://arxiv.org/abs/2610.18615), §1 of `research_report.md`) |
| T4 adaptations | `ml/prepare_model.py`, `runtime._patch_sdpa` | bfloat16→float16 copy; GQA disabled in SDPA so the memory-efficient kernel is used; peak 10–11 GB; Triton kernels of flash-linear-attention compiled at warm-up (~3 min, cached in `/data/triton`) |
| Window budget | `runtime.prepare_window` | context 16 384, visual budget 13 107 tokens; a 40-frame window at 464×848 is ≈7–8 k tokens, so the native resolution is kept (no downscale), which matters for door/safe/banknotes that are tens of pixels wide |
| Queries | `app/vj/queries.py` | 8 types; 3 options each except PERSONAL_ITEMS (2); English; grounding description per type; `state=True` for SAFE_OPEN, DOOR_OPEN |
| Head fine-tuning | `ml/train_head.py` | frozen backbone; hidden states cached once (`score_dataset.py --hidden`); head initialised from the pretrained one; CE on gates with positive weight (neg/pos)^0.5, BCE on grounding cells (±1.5 s around the timecode, [t, t+3] for states) ×0.5; L2 anchor 1e-3 to the pretrained weights; AdamW lr 1e-4, 8 epochs, batch 32; augmentations (hflip, persp, none) as extra examples; LOVO by video (seed = fold index); final head on all videos |
| Thresholds | `ml/calibrate.py` → `ml/artifacts/thresholds.json` | grid 0.05…0.95; coordinate ascent: global threshold then one per-type correction at a time; objective F2 (precision proxied by 1−FP rate) + 0.25·type accuracy; nested LOVO for the report, fit on all 40 for production: default 0.45, UNATTENDED_CASH 0.25, NO_COUNTER 0.70, UNAUTHORIZED_PERSON 0.05, CASH_EXPOSED_TO_VISITOR 0.15 |
| Service | `service/app/*` | FastAPI; thread pool (WORKERS=2) but one GPU lock; job JSON on disk; 280 s job timeout; `.mov/.mp4/...` accepted; 300 MB cap; API key; warm-up on a synthetic clip; frontend proxies `/api/v1` |
| Output | `analyzer.analyze` | verdict, violations[type, type_code, timecode_sec, description (observation + rationale + clause), confidence, rules[], evidence_frames (1–3 JPEG base64)] |

## 2. Data and labels

**Files.** 42 `.mov`, all H.264 464×848 portrait at 30 fps, no audio, 11.6–142.5 s (median ≈ 22 s), total 280 MB. Two
byte-identical pairs (`Х-4`=`Х-47`, `Х-5`=`Х-48`, md5 verified) → **40 unique videos**: 21 violation videos with 22
labelled events (`Х-4` carries two: CASH_EXPOSED_TO_VISITOR at 57 s and NO_COUNTER at 68 s) and 19 normals.
`common.DUPLICATES` merges them correctly.

**Organizer labels** (`open/open_labels.xlsx`): verdict + one timecode per violation video, no type, no interval.
`analysis/labels.json` adds the team's type per event with a self-assessed confidence: **high 4, medium 11, low 7**
of 22. Six events carry alternative types (`alt_type_codes`): X-2 (PERSONAL_ITEMS|NO_COUNTER), Х-11 and Х-22
(UNATTENDED_CASH|SAFE_OPEN,DOOR_OPEN), Х-24 (UNATTENDED_CASH|DOOR_OPEN,PERSONAL_ITEMS), Х-28 (DOOR_OPEN|UNATTENDED_CASH),
Х-34 (DOOR_OPEN|CASH_EXPOSED_TO_VISITOR). A third of the type labels are therefore hypotheses; the "type accuracy"
ceiling is itself uncertain and any type metric should be reported both strictly and with alternatives accepted
(`evalkit.video_metrics`: `type_acc_on_pos` vs `type_acc_acceptable`).

**Class balance per type** (events): UNATTENDED_CASH 5, NO_COUNTER 4, DOOR_OPEN 4, PERSONAL_ITEMS 3,
CASH_EXPOSED_TO_VISITOR 2, UNAUTHORIZED_PERSON 2, CIT_PROCEDURE 1, SAFE_OPEN 1. Per-type recall numbers are therefore
0/1–5/5 counts, not rates (a 0/4 is compatible with a true recall of ~60% at 95% confidence).

**Camera sessions** (`ml/analysis/scene_groups.py`, median-background correlation ≥ 0.92 → `analysis/setups.json`):

| Setup | Videos (violation / normal) | Notes |
|---|---|---|
| S0 | X-1, X-2, Х-4 / Х-3, Х-5 (3/2) | two desks visible; the long clips |
| S1 | Х-13, Х-15, Х-17, Х-18, Х-20, Х-22 / Х-10, Х-16, Х-19, Х-21, Х-23 (6/5) | largest session |
| S2 | Х-11 / Х-8, Х-12, Х-14 (1/3) | darker (mean gray 122–126 vs 133–143); **camera physically moves** during Х-11 and Х-12 (first-vs-last background correlation drops by 0.50; all other videos ≤ 0.07) |
| S3 | Х-24, Х-26, Х-28 / Х-25, Х-27, Х-29 (3/3) | |
| S4 | Х-30, Х-32 / Х-31 (2/1) | |
| S5 | Х-33, Х-40 / Х-45 (2/1) | |
| S6 | Х-34, Х-38 / Х-35, Х-37, Х-39 (2/3) | |
| S7 | Х-44, Х-46 / Х-43 (2/1) | |

Within a session the violation and its normal counterpart are re-takes of the same scene with the same actors
(X-2/Х-3, Х-16/Х-17, Х-18/Х-19, Х-20/Х-21, Х-22/Х-23, Х-24/Х-25, Х-26/Х-27, Х-28/Х-29, Х-38/Х-39 …). Under
leave-one-video-out the twin of every held-out video is in the training fold. That is the single most important
reason the LOVO numbers overstate what to expect from the jury's "different angle" video.

**Windows.** With the production windowing the 40 videos yield **98 windows** (62 from normals, 36 from violations);
**27 videos are a single window** (18 violation, 9 normal); only 3 violation videos (Х-4, Х-13, Х-38) and 10 normals have
several windows; Х-14 alone has 14. "Thousands of frames" never become independent samples here: the effective
sample size for every verdict metric is 40, for type accuracy 21, for the timecode metric 9.

## 3. Reported performance and what it means

`ml/README.md` (first release, LOVO on the 40 unique videos; thresholds refit inside each fold):

| Metric | Reported | Count | Wilson 95% CI |
|---|---|---|---|
| Recall (violations flagged) | 76% | 16/21 | 55–89% |
| False positives on normals | 37% | 7/19 | 19–59% |
| Verdict accuracy | 70% | 28/40 | 55–82% |
| Type correct (of all positives) | 43% | 9/21 | 25–64% |
| Timecode within ±2 s (of type-correct) | 56% | 5/9 | 27–81% |

Zero-shot + thresholds: 86% recall / 63% FP; stacking: 81% / 47%; "always violation": 100% / 100%.
Latency on the T4 (team measurement): 11–33 s for 12–20 s clips, 143–157 s for the 142 s clip through the public API.

Two sources of optimism are not in those intervals: (a) the twin-session leakage above; (b) selection on the
same 40 videos — at least seven prompt/window/decision variants in `ml/README.md` §6 were compared on the same
LOVO protocol, and the head hyper-parameters (epochs, lr, anchor) were chosen by looking at it. There is no
untouched evaluation set. Expect the jury videos to land in the lower half of the intervals unless the
generalization measures in `generalization_report.md` are taken.

## 4. Findings

Severity: **H** likely changes jury scores, **M** matters for robustness or honesty of the evaluation, **L** hygiene.
Findings F1–F14 were made by reading; F15–F17 by the experiments of the same day (registry rows in `experiments.csv`).

### F1 (H, bug, fixed in this session) — the deployed decision rule was not the evaluated one
`analyzer.analyze` dropped every detection with `confidence < MIN_CONFIDENCE = 0.5`, and `VideoJevDetector` wrote the
raw gate probability into `confidence`. Production thresholds are 0.25 / 0.05 / 0.15 / 0.45 / 0.70, so any hit with a
score in [threshold, 0.5) passed `decide()` and was silently removed by the analyzer; only NO_COUNTER (0.70) was
unaffected. `calibrate.py` never applied that filter, so the 76% / 37% figures describe a rule the service did not run.
Fix: `vj.detector.confidence_from_margin` maps the margin above the per-type threshold to [0.5, 1] (threshold → 0.5,
score 1 → 1), the decision now survives the analyzer, the raw gate scores are logged per request, and
`tests/test_vj_logic.py::test_confidence_from_margin_survives_analyzer_filter` guards it. Effect size, measured (E01a vs E01b
in `experiments.csv`, production thresholds applied to the out-of-fold scores): the filter removed 6 of 20 flagged
violation videos (recall 95% → 67% for that in-sample rule) and 2 of 6 false positives (32% → 21%). The deployed
service was therefore missing roughly one violation in three that its own evaluated rule would have reported.

### F2 (H, label noise in training) — alternative types were trained as hard negatives
`common.window_targets` returns 0 for every type that is not the event's primary type. For the six videos with
`alt_type_codes` the head therefore learned "DOOR_OPEN = no" on Х-11, Х-22 and Х-24 although the door is visibly open
there (the labels say so), "SAFE_OPEN = no" on Х-11/Х-22, "NO_COUNTER = no" on X-2, "PERSONAL_ITEMS = no" on Х-24,
"UNATTENDED_CASH = no" on Х-28 and "CASH_EXPOSED_TO_VISITOR = no" on Х-34. DOOR_OPEN has 4 positives and received 3
contradictory negatives; it is one of the two types at 0/4. Option implemented: `window_targets(..., alt_policy=)` with
`ignore` (mask) and `positive`; `train_head.py --alt-policy`. **Measured (E02a/E02b): rejected.** Masking those windows
raises the false-positive rate from 37% to 58% with no recall gain, and treating them as positives gives 53%. The
"contradictory" negatives are in fact the hardest negatives the head has (a cashier leaving with the door still in
view), and removing them costs more than the label noise did. Finding stands as a label-quality issue; the training
policy stays as released.

### F3 (H, evaluation validity) — LOVO is optimistic for the jury condition and has no error bars
See §2–3. Implemented: `ml/evalkit.py` (Wilson and bootstrap intervals, paired bootstrap, leave-one-setup-out,
repeated group k-fold, leave-one-type-out, event-level matching with tolerances, alternative-type-aware type accuracy)
and `ml/experiment.py` (registry rows with intervals). Recommended protocol in `experiment_plan.md` §3.
Also minor: the threshold LOVO in `calibrate.py` consumes `scores_cv.jsonl`, whose other-video scores come from heads
that *did* see the held-out video, so the threshold search is not strictly nested with respect to head training
(small optimistic bias, documented, not fixed: fixing it costs 40× the head training).

### F4 (M, aggregation) — max over windows favours long videos
The video score is the maximum of per-window probabilities over 8 types. The maximum of N noisy scores grows with N.
All three normals longer than 70 s (Х-5, Х-14, Х-39: 7, 14 and 7 windows) are among the 7 false positives; the other
four FPs (Х-16, Х-23, Х-29, Х-31) are single-window "hard normals" where the cashier leaves an empty desk or closes
the door. 3/5 FP among normals with ≥5 windows vs 4/14 among the rest (one-sided Fisher p = 0.24: suggestive, not
significant at n = 19). Implemented alternatives in `ml/decision.py`: `top2` mean, and length-invariant p-value
aggregation (`zsidak`: Gaussian tail on the logits of normal training windows, Šidák-combined over the N windows;
`sidak`: empirical variant with a documented resolution floor). **Measured (E03e):** the Šidák verdict with type and
timecode from the baseline rule raises recall from 76% to 90% at the same 37% false-positive rate, on nested LOVO,
leave-one-setup-out (acc 0.75 vs 0.675) and repeated group 5-fold (0.755 ± 0.037 vs 0.675). It clears all three long
normals (Х-5, Х-14, Х-39) and finds X-1, Х-32, Х-44, while flagging Х-3, Х-19, Х-37 — so the hard-normal mechanism
("leaves with an empty desk") remains and needs a different cure. Shipped as `ml/artifacts/thresholds_v2.json`
(`detector.choose`).

### F5 (M, overfit thresholds) — per-type thresholds of 0.05 and 0.15 come from two positives each
The coordinate ascent can push a type's threshold to the grid edge to catch one more positive; nested LOVO reports
the consequence honestly but the variance is large. Implemented: `--objective recall_at_fp --alpha`, `--no-per-type`,
`--shrink δ` (per-type corrections limited to ±δ around the global threshold). Verify with `E05`.

### F6 (M, calibration) — "уверенность" in the UI is an uncalibrated softmax
VideoJev reports a mean ECE of 8.65 points on QA benchmarks with over-confident probabilities (paper §6); our gates
are 2–3-option questions outside its training distribution (4–5 options) and the head was re-trained with class
weights, so the probability is not a frequency. Implemented: Brier/log-loss/ECE in `evalkit.score_metrics`.
Plan: Platt scaling of the out-of-fold video margin (`E07`); report confidence bands rather than percentages.

### F7 (M, multi-event) — one event per video by construction
`max_events = 1` in production; Х-4 has two labelled events of different types. `Policy(max_events=2)` and
event-level metrics exist; verify the FP cost with `E08`. Same-type repeats (two separate exits with cash on the desk)
are not representable at all; the labels do not contain such a case, so this stays a known limitation.

### F8 (M, temporal resolution) — 20 s windows and single-timecode supervision
At 2 fps a hand-to-hand transfer lasting 1–2 s is 2–4 frames inside a 40-frame window; the gate must notice it among
38 unrelated frames. The team measured that 10 s / 5 s windows raise type accuracy (43 → 52%) and timecode accuracy
(56 → 67% within 2 s) at the price of a second pass that pushes the 142 s video to ≈4 min. The grounding target is a
±1.5 s band around a point label, and the "state onset at half peak" rule has no ground truth to validate against
(the organizer's timecode for states is "when the state becomes visible", also a guess). The timecode metric rests on
9 videos. Plan: coarse-to-fine localization only for the flagged type (`E13`), timecode tolerance curves at
1/2/5 s (already in `evalkit`).

### F9 (M, throughput and the 5-minute limit)
Per window: one vision encode + one joint forward for 8 gates, then one re-encode of the best window for grounding
(the window is rebuilt because pixels are freed; caching the features of the best window would save one encode).
142 s = 14 windows ≈ 145–157 s end-to-end. Risks: (i) the 280 s job timeout counts queue time, and the GPU lock
serialises jobs while `WORKERS=2` accepts two, so two long uploads back-to-back can fail the second; (ii) a teammate's
container on the same GPU multiplies latency (documented in `ml/README.md`). Options: `WORKERS=1` with an explicit
queue position in the status, or adaptive stride for long videos (stride 15 s above 60 s costs ≈30% of recall
resolution but no verdict information for state types), or skipping grounding when no type passes (already the case).
Latency profiling per stage is `E16`.

### F10 (M, open set) — unknown violation types are missed by design
Eight gates, exact catalogue mapping, no catch-all, no abstention. The ToR says unseen types may appear (FAQ 9).
Plan and comparison of options in `generalization_report.md` §5 and `E10`/`E11`.

### F11 (L, prompts) — format shift from VideoJev's training
Training QA used 4–5 letter-prefixed options with order permutation; our gates use 2–3 options with "A" always the
violation. The paper notes later queries in a joint forward can attend to earlier query text (no answer leakage, but
context differs with question order). Cheap GPU checks: permute option order and gate order and measure score
stability (`E12`).

### F12 (L, regulatory mapping)
`rules.py` quotes were checked against the adilet copies in `analysis/sources/` — good. Two soft spots the description
should not overstate: DOOR_OPEN relies on premises-design clauses (NBRK 14 п. 8/16), and food/drink has no public norm
(labels.json notes it; the PERSONAL_ITEMS rationale cites п. 12 which covers personal belongings and phones). Wording
in `OBSERVATIONS`/`rationale` already avoids legal conclusions; keep it that way and mark these two as "indirect".

### F13 (L, reproducibility)
Model outputs (`runs/*/scores.jsonl`, the hidden-state cache, `head.pt`) exist only on the server; nothing in git or in
a fetch script. Added `scripts/fetch_runs.sh` (rsync) and `runs/` to `.gitignore`; consider attaching `head.pt` and
`scores_cv.jsonl` to the private `dataset-v1` release so the evaluation can be reproduced on a laptop in seconds.

### F14 (L, robustness of inputs)
`video.probe` (OpenCV, frame count / fps) and `runtime.decode` (PyAV, PTS grid) agree on these files; variable-frame-rate
or audio-bearing `.mov` from a different recorder has not been tested. The warm-up clip is 464×848 like the dataset;
a jury video with another aspect ratio changes the token budget (handled by the binary search in `prepare_window`,
but untested). Add one test video of a different resolution to the smoke tests before submission.

### F15 (H, deployment) — the deployed head and the deployed thresholds were never evaluated together
`thresholds.json` was fitted on `scores_cv.jsonl`, i.e. on the scores of 40 *fold* heads (each trained on 39 videos,
seed = fold index), while the deployed `head.pt` is a 41st head trained on all 40 videos with seed 0. Re-scoring the
cache with the deployed head (E14h-none, in-sample) gives recall 76% and FP 53% under the production thresholds, whereas
the fold heads the thresholds were tuned on give 95% / 32% (E01a). Mean score shift between the two is small
(|ΔP| 0.04 on normals) but the per-type threshold for UNATTENDED_CASH (0.25, chosen on two-positive statistics) sits
exactly where the deployed head puts several hard normals; refitting on the deployed head's own scores moves it to 0.65.
Root cause is head-training variance (F16). Remedy implemented: `train_head.py` now also writes `scores_cv_train.jsonl`
(each fold's training videos scored by that fold's head) and `experiment.py --train-scores` fits thresholds on them, so
the cross-validated estimate matches the deployment recipe "train on everything, fit thresholds on the deployed head's
own scores" (`ml/export_policy.py`). **Results (rows `N*`):** under that protocol the released configuration scores
recall 76% / FP 53% / accuracy 62.5% (N00) against 76% / 32% / 72.5% when thresholds come from other folds' out-of-fold
scores (N00-old): in-sample threshold fitting on a head that has memorised its 39 training videos sets thresholds too
low for unseen videos. Neither protocol matches deployment perfectly — the cross-head protocol assumes the deployed head
behaves like the fold heads (it does not, F16), the in-sample protocol over-fits thresholds — so the realistic
expectation for the live system is **62–70% verdict accuracy, 76% recall, 37–53% false positives** on familiar cameras.
The 3-seed soup did not change this within noise (N09: 71 / 47 / 62.5).

### F16 (H, variance) — head training is not reproducible at the verdict level
Re-running the released configuration (same cache, same hyper-parameters, batched trainer) reproduces the aggregate
numbers (recall 76%, FP 32%, accuracy 72.5%) but flips 11 of 40 verdicts against the production head (E00r2; mean
|ΔP(A)| 0.05, max 0.62). Any claimed difference of one to three videos between two head variants is therefore inside
run-to-run noise. Mitigation under test: 3-seed weight averaging with feature-space noise (`--seeds 3 --ensemble soup`,
rows N09/N02s).

### F17 (M, measured robustness) — geometry is handled by augmentation, the session is not
Zero-shot gates with fixed thresholds lose 10 (flip) and 15 (perspective) accuracy points on the transformed copies of the
40 videos (E14zs); the fine-tuned head, trained with those augmentations, loses at most 3 (E14h). But a head trained
without the test camera session (true leave-one-setup-out, E06h) goes from 37% to 58% false positives at 81% recall:
the head learns session appearance, and no geometric augmentation covers "a room it has never seen". This is the
number to quote for the jury's different-angle condition until self-recorded clips exist.

## 5. Answers to the mandatory audit questions

| # | Question | Answer (evidence) |
|---|---|---|
| 1 | Fine-tuning approach appropriate for the data size? | Yes in kind: frozen 4B backbone, 2.6 M-parameter head anchored to its pretrained weights, class weights, feature caching. Not in detail: alt-type negatives (F2), per-type thresholds from 1–5 positives (F5), hyper-parameters chosen on the evaluation folds (F3). Backbone LoRA would learn these 8 sessions; not recommended (see research report). |
| 2 | Overfitting? | The head moved the operating point (recall 86→76%, FP 63→37%) rather than adding discrimination; "fit on all" vs LOVO numbers are not reported side by side. Cannot be quantified without the score files; `experiment.py --split none` vs `--split lovo` and `--split setup` give the three numbers. |
| 3 | Train/validation leaks or duplicates? | Byte duplicates handled. Twin recordings from the same session (8 sessions, 9+ twin pairs) are split by LOVO — a leak with respect to the jury condition, not with respect to the data as given. Augmented copies of the held-out video are correctly excluded. Threshold search not nested w.r.t. head training (minor). |
| 4 | Metrics implemented correctly? | Yes for what they define (verdict recall/FP, first-event type, timecode error on type-correct videos). Missing: intervals, event-level matching, alternative types, calibration, per-session breakdown — added in `evalkit`. `tc_within_2s` uses ≤ 2 s; the ToR gives no tolerance ("a specific frame"), so report 1/2/5 s. |
| 5 | Validation representative of unseen videos? | Partly. Same cameras, same actors, same room; the jury adds a new angle and possibly new types. Leave-one-setup-out is the closest proxy available in the data; a self-recorded different-angle set is the only honest check. |
| 6 | Learning actions or artifacts? | Unknown without ablations. Risk indicators: FPs on hard normals (semantics), length effect (statistics), head trained with the twin scene in the fold (scene memorisation possible). Leave-one-setup-out and the test-time transform suite in `generalization_report.md` are the tests. |
| 7 | Does temporal downsampling discard short actions? | 2 fps keeps 2–4 frames of a 1–2 s handover; 20 s windows dilute them (F8). The team's 10 s-window result (+9 type points) says yes for NO_COUNTER/CIT-like events. |
| 8 | Reliably distinguish normal from violation? | No: verdict accuracy 70% [55, 82]; the hardest normals are the ones that look like violations by construction (cashier leaves, door closes). |
| 9 | Calibrated confidence? | No (F6). |
| 10 | More than one event per video? | Not in production (F7); supported by the code paths and metrics now. |
| 11 | Events outside the taxonomy? | Missed by design (F10). |
| 12 | What limits throughput? | Visual encoding and the joint forward per 20 s window on a T4 (~10 s per window), serialised by one GPU lock; re-encoding the best window for grounding; Triton warm-up on cold start; a co-tenant container. |

## 6. Reproduction status

Executed here: service tests (55 pass), new toolkit tests (16 pass), dataset analyses
(`ml/analysis/scene_groups.py`, `ml/analysis/dataset_stats.py`). Not executed: anything requiring model outputs.
To reproduce the baseline on the GPU host (paths as in `ml/README.md` §4):

```bash
python ml/calibrate.py runs/head/scores_cv.jsonl                      # first-release numbers
python ml/experiment.py --scores runs/head/scores_cv.jsonl --exp-id E00 --hypothesis "baseline" --save-pred runs/E00_pred.jsonl
```

`ml/run_experiments.sh` chains every offline experiment of `experiment_plan.md`; `scripts/fetch_runs.sh` copies the
score files and cache to a laptop so that all of them run without a GPU.
