# Experiment plan — ranked hypotheses, evaluation protocol, resource estimates

Companion to `model_audit.md` (findings F1–F15) and `research_report.md`. IDs are reused in
`reports/experiments.csv` and `ml/run_experiments.sh`. Status on 2026-10-18 (evening): tier A ran in full on
the fetched score files (E00–E10), head retrains ran on the T4 (E02, E06h, E09 pending as the nested soup),
E14 ran as far as the cached augmentations allow; E11–E13, E15–E16 did not run. Outcomes are in
`reports/experiments.csv` and summarised in `final_improvement_report.md`; this file keeps the original ranking
with a status column so the reasoning stays visible.

| ID | Status | One-line outcome |
|---|---|---|
| E00 | done | reproduces 76 / 37 / 70 / 43 / 56 exactly; margin ROC-AUC 0.77 |
| E01 | done | deployed filter cost 6 of 20 flagged violations in-sample (recall 0.95→0.67) |
| E02 | done, rejected | masking alt-type negatives raises FP 0.37→0.58; the "noisy" negatives are useful hard negatives |
| E03 | done, kept (E03e) | Šidák verdict + baseline type/timecode: recall 0.76→0.90 at equal FP on LOVO, setup and group k-fold; does not hold when the head itself never saw the session (E06h+E03e) |
| E04 | done, rejected | bias subtraction: FP 0.58 |
| E05 | done, rejected | single/shrunk thresholds and recall@FP objective all worse out of fold |
| E06 | done | thresholds-only setup split costs 1 video; head trained without the session: FP 0.37→0.58, acc 0.625 — the honest new-camera figure |
| E07 | not run | Platt scaling of the margin |
| E08 | done, rejected | two events: event-F1 down, Х-4's second event still missed |
| E09 | running | nested 3-seed soup (n_soup, n_alt_soup on the host) |
| E10 | done | sibling gates flag 0–60% of a held-out type's videos |
| E11–E13 | not run | need new GPU passes |
| E14 | partial | zero-shot gates lose 10–15 acc points under flip/perspective; the augmented-trained head loses ≤3 — training with augmentations worked for geometry |
| E15–E16 | not run | |
| New: E00r2 | done | identical config re-run flips 11/40 verdicts: head-training noise is the dominant uncertainty |
| New: F15 | done | deployed head.pt + thresholds.json were never evaluated together: in-sample 76 / 53 instead of the 95 / 32 of the fold heads the thresholds were fitted on |

## 1. Ranking criteria

Expected effect on the jury criteria (detection 25, localization 15, classification 15, robustness 10), confidence in
the hypothesis, compute, implementation effort, regression risk. Cheap offline checks come first so that weak ideas die
before any GPU time is spent. A change is **kept** only if, on the same out-of-fold predictions, the paired bootstrap
gives P(better) ≥ 0.9 on balanced accuracy *and* recall does not drop by more than one video *and* leave-one-setup-out
does not get worse. With n = 40, differences under two videos are noise and are logged as "investigate".

## 2. Ranked experiments

### Tier A — offline on cached scores and hidden states (laptop or server CPU, seconds to minutes each)

| ID | Hypothesis | Change | Expected effect | Cost | Risk | Needs |
|---|---|---|---|---|---|---|
| **E00** | Reproduce the first-release baseline with the new toolkit and intervals | `experiment.py --scores runs/head/scores_cv.jsonl` (also zero-shot `runs/full/scores.jsonl`) | identical 76/37/70/43/56 point estimates; CIs; per-session breakdown | 1 min | none | `scores_cv.jsonl` |
| **E01** | The deployed service under-reports because of the MIN_CONFIDENCE filter (F1) | `--fixed-thresholds ml/artifacts/thresholds.json --simulate-min-confidence 0.5` vs without | quantifies what the jury would have measured before the fix | 1 min | none | same |
| **E02** | Alternative-type negatives hurt DOOR_OPEN/SAFE_OPEN (F2) | `train_head.py --alt-policy ignore` (and `positive`) → new `scores_cv.jsonl` → `experiment.py` | +1–2 videos on DOOR_OPEN/UNATTENDED_CASH recall, type accuracy up; FP unchanged | 50 min GPU-free on cached states (CPU: ~minutes per fold × 40; `--device cpu`) | low; label-policy choice is itself a hypothesis | hidden-state cache + torch |
| **E03** | Max-over-windows inflates FP on long videos (F4) | `--agg zsidak`, `--agg top2`, compare on long normals | 1–3 fewer FPs among Х-5/Х-14/Х-39 without losing Х-4/Х-13/Х-38 | 1 min | zsidak needs ≥ ~30 normal windows per fold (we have ~55–60) | `scores_cv.jsonl` |
| **E04** | Questions with a permanent "yes" bias dominate; per-type bias subtraction on normal windows helps (research §2) | `--calib bias` (with max and zsidak) | fewer FPs from DOOR_OPEN/UNATTENDED_CASH gates; recall kept | 1 min | bias estimated on 10–19 normals per fold | same |
| **E05** | Per-type thresholds overfit two-positive types (F5) | `--no-per-type`; `--shrink 0.1`; `--objective recall_at_fp --alpha 0.3` | similar point estimates, narrower spread across sessions (`--split setup`, `--split rgkf`) | 1 min each | may lose 1 recall video | same |
| **E06** | Leave-one-setup-out is materially worse than LOVO (F3): quantify the generalization gap | every kept variant with `--split setup` and `--split rgfk --k 5 --repeats 10` | expected drop of 5–15 points of verdict accuracy vs LOVO; this gap is the robustness budget | 5 min | none | same |
| **E07** | A 2-parameter Platt scaling of the out-of-fold video margin gives a usable confidence | fit logistic on `margins` from `nested_evaluate`, nested per fold; report Brier/ECE | ECE ≤ 0.1, honest UI bands | 30 min code | over-fitting with 40 points: fix slope sign, two params only | same |
| **E08** | Two events per video when two types clear their thresholds (F7) | `--max-events 2`, event-level metrics at 2/5/10 s | Х-4 second event found; cost = extra wrong types on positives, no new FPs on normals (normals already flagged or not) | 1 min | type-accuracy metric uses the first event only; report event-F1 | same |
| **E09** | Variance reduction of the head: 5-seed weight averaging and feature-space noise (research §4) | `train_head.py --seeds 5 --ensemble soup --feature-noise 0.05` | smoother thresholds, +0–1 video; mainly stability across folds | ~5× head training | none at inference (one head) | cache + torch |
| **E10** | The current gates already flag some unseen types through sibling gates (open-set probe, F10) | `experiment.py --open-set` (leave-one-type-out on cached scores) and `train_head.py --exclude-type` | a table: for each held-out type, share of its videos still flagged, by which gate | 2 min (+8 head retrains) | none | same |

### Tier B — new forward passes on the T4 (≈25 min per full pass over 40 videos at one augmentation; ≈10 s per 20 s window)

| ID | Hypothesis | Change | Expected effect | Cost | Risk | Needs |
|---|---|---|---|---|---|---|
| **E11** | A small set of generic, rule-shaped queries ("valuables accessible with no employee attending", "a non-employee in the staff zone", "an object changes hands across the desk", "a storage unit is left open") catches unseen types; shipped as a lower-confidence "needs review" tier | add queries in `queries.py` with `generic=True`; score zero-shot *and* with the head; evaluate with leave-one-type-out (train thresholds without type T, measure recall of T through generic gates vs FP on normals) | open-set recall > 0 for ≥ 4 of 8 held-out types at ≤ +1 FP | 1 pass (25 min) + 2 h code | generic queries say "yes" everywhere (the team saw this with atomic yes/no questions) — mitigated by the 3-option format with an explicit normal alternative and by `bias` calibration | GPU |
| **E12** | Scores depend on option order / gate order (F11) | re-score with options permuted (violation option as B/C) and gates shuffled; average logits (deterministic TTA) | stability (mean |Δp| < 0.05); if large, averaging over 2 orders becomes the default | 2 passes | none | GPU |
| **E13** | Coarse-to-fine localization for the flagged type only: 6 s clips, 2 s stride, 4 fps around the best window; timecode = state transition / peak of the fine curve (research §3) | new `detector` stage after `decide()`; metrics at 1/2/5 s | timecode within 2 s from 56% toward 70%+; +10–30 s per flagged video | 1 pass on positives + 3 h code | time budget on long videos (cap at one type, one peak) | GPU |
| **E14** | Label-preserving test-time transforms expose fragility (generalization §3) | re-score the unchanged system on: hflip, perspective, rotation+zoom, 0.5× downscale, H.264 CRF 35, Gaussian blur σ=2, 1 fps sampling; measure verdict flips and score drift per transform | a robustness table; transforms that flip > 10% of verdicts define the next augmentation/training target | 7 passes (~3 h) | none | GPU |
| **E15** | Feature caching for grounding and `WORKERS=1` remove the back-to-back timeout risk (F9) | keep the best window's features instead of re-encoding; queue position in job status; profile per stage | −5–10 s per flagged video; no job fails from queue time | 2 h code, 10 min measurement | none | GPU |
| **E16** | Backbone LoRA on the last 4 blocks over cached inputs (research §4) | only if E02–E09 leave recall < 85% and self-recorded data exists | unknown; high overfit risk | 1 day | high | GPU + more data |

### Tier C — data (people, no GPU)

| ID | Action | Why | Cost |
|---|---|---|---|
| **D1** | Record 10–20 clips from 2–3 new angles: each failing type (NO_COUNTER, DOOR_OPEN, SAFE_OPEN, CIT) with its hard-normal twin | the only honest held-out set for the jury condition; also the calibration set for thresholds | 3–4 h for two people + 1 h labelling |
| **D2** | Second-opinion labelling of the 7 low-confidence types (two annotators, record disagreement) | 1/3 of type labels are hypotheses; type accuracy cannot exceed label accuracy | 1 h |

## 3. Evaluation protocol (applies to every experiment)

1. **Unit of analysis is the video** (40 unique). Never count windows or frames as samples.
2. **Splits.** Primary: nested leave-one-video-out (comparability with the first release). Required alongside:
   leave-one-setup-out (8 folds, `analysis/setups.json`) and repeated group 5-fold (10 repeats). Thresholds and
   calibration statistics are fitted inside each training fold only (`decision.nested_evaluate`).
3. **Metrics.** Verdict: recall, FP rate, balanced accuracy, verdict accuracy, F2, ROC-AUC/PR-AUC of the video margin.
   Type: strict and alternative-accepting. Localization: MAE, within 1/2/5 s, event-level F1 at 2/5 s.
   Reliability: Brier, log loss, ECE (5 bins) once E07 exists. Efficiency: per-stage seconds per video on the T4,
   peak memory, timeouts.
4. **Uncertainty.** Wilson intervals for proportions, 2 000-sample stratified bootstrap over videos, paired bootstrap
   and the list of flipped videos for every A/B. Leave-one-type-out results are reported per type as counts.
5. **Decision rule.** See §1. Record every run in `reports/experiments.csv` with the exact command (the runner
   appends the row), including rejected ones.
6. **No contamination.** Jury videos are never used for anything. Self-recorded clips (D1) are a frozen test set:
   thresholds may be *checked* on them, not tuned, unless a second batch exists.

## 4. Resource summary

| Resource | Tier A | Tier B | Tier C |
|---|---|---|---|
| GPU | none (cached scores) | ≈6–8 T4-hours in total | none |
| CPU | minutes | — | — |
| People | 2–3 h to run and read results | 1–2 days engineering | 4–6 h |
| Blocking input | `runs/head/scores_cv.jsonl`, `runs/full/scores.jsonl`, `runs/full/cache` from the server | SSH to the T4 host | camera + two people |

## 5. Order of execution

E00 → E01 → E03/E04/E05 (same files, minutes) → E02 and E09 (head retrains) → E06 for every survivor → E08, E10 →
then GPU: E11 (open set) and E13 (timecode), E14 (robustness table), E15 (latency) → final selection → re-run the
kept configuration on all three splits → write `final_improvement_report.md` numbers.
