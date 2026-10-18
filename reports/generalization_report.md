# Generalization report — distribution shifts, OOD experiments, failure modes, unseen-type strategy

Date: 2026-10-18. Measured facts come from `ml/analysis/scene_groups.py` and `ml/analysis/dataset_stats.py` on the
42 dataset videos; model-dependent numbers in §3 come from executed rows of `reports/experiments.csv` (E06, E10, E14 and the nested
`N-*` rows of the final selection). E11 and E13 have not been run.

## 1. Known distribution shifts between training data and the jury's test

| Shift | Training data (measured) | Jury condition (ToR / FAQ) | Risk to the current system |
|---|---|---|---|
| Camera placement | 8 sessions, all top-down from above the desk, one camera; the same three rooms | "the final video is shot from a different angle" (FAQ 4) | high: the head and thresholds were selected with the twin scene in the training fold; nothing in the pipeline adapts per video |
| Scene / background | 3 room layouts; desk, safe, door and counting machine in fixed image regions per session | same type of room, position unknown | medium: the gates are semantic questions, not ROI-based (good); the head may have learned session-specific appearance (unknown) |
| Illumination | mean gray 133–143 in 7 sessions, 122–126 in S2 (Х-8/11/12/14) | unknown | low–medium; one darker session exists in training |
| Camera motion | the camera moves inside Х-11 and Х-12 (background correlation first/last 0.50 vs ≥ 0.93 elsewhere) | unknown | low; shows the recorder is hand-placed |
| Resolution, codec, frame rate | all 464×848, H.264, 30 fps, `Lavf63` encoder, no audio | ".mov, ~29 MB, fps ≥ 29, no audio" (FAQ 1, 25) | low for decoding (PyAV grid sampling is resolution-agnostic); untested for other aspect ratios (token budget search in `prepare_window`) |
| Duration | 11.6–142.5 s; 27 of 40 videos are one window | 10–145 s | medium: the max-over-windows rule behaves differently for 1 and 14 windows (audit F4) |
| Event types | 8 types, 1–5 examples each; 7 of 22 type labels self-rated low confidence | "types absent from the training data may appear" (FAQ 9) | high: closed set (audit F10) |
| Actors and props | same few actors; green CIT bag; one safe model | unknown | medium |
| Number of events | one labelled event per video except Х-4 | "several violations in one video are possible" (ToR §4) | medium: `max_events = 1` in production |

## 2. What the current protocol can and cannot see

- Leave-one-video-out keeps the twin recording of every test video in training; it measures "same camera, same
  actors, different take". Leave-one-setup-out (8 folds) measures "unseen session with the same recorder height";
  it is the closest in-data proxy for the jury's shift and is now available (`experiment.py --split setup`).
- Expected (not yet measured) gap between the two: 5–15 verdict-accuracy points. Whatever the gap is, it is the
  budget for robustness work, and any variant that widens it is rejected even if LOVO improves.
- Neither split covers a new angle. Only self-recorded clips from 2–3 other angles (plan D1) do. Until they exist,
  all numbers are "familiar-camera" numbers and should be labelled as such in the submission.

## 3. OOD experiments

Measured on 2026-10-18 (registry `experiments.csv`):

| Experiment | Condition | Recall | FP | Accuracy | Reading |
|---|---|---|---|---|---|
| E00 | nested LOVO, production head + thresholds refit per fold | 76% | 37% | 70% | familiar cameras, head saw the twin take |
| E00-setup | leave-one-setup-out for thresholds only (head still saw the session) | 76% | 42% | 67.5% | thresholds transfer across sessions at the cost of ~1 video |
| **E06h** | head trained without the test session, thresholds too | 81% | 58% | 62.5% | **the honest "new room" number**; FP +21 points |
| E14zs hflip / persp | zero-shot gates, fixed thresholds, transformed videos | 90% / 90% | 68% / 79% | 62.5% / 57.5% | geometry alone costs 10–15 points without augmentation training |
| E14h hflip / persp | fine-tuned head (trained with these augs), production thresholds | 81% / 81% | 58% / 53% | 62.5% / 65% | augmentation training neutralised geometry (≤ 3 points vs its own none-run) |
| E03e-setup | Šidák policy, thresholds-only setup split | 90% | 42% | 75% | gain holds across sessions when the head has seen them |
| E06h+E03e | Šidák policy on the session-blind head | 81% | 68% | 57.5% | the normal-window calibration does not transfer to an unseen session |
| N00 | thresholds fitted on each fold head's own training scores (deployment-consistent) | 76% | 53% | 62.5% | in-sample thresholds over-fit; realistic familiar-camera range 62–70% |
| N09 | 3-seed weight-averaged head, same protocol | 71% | 47% | 62.5% | variance reduction does not move the estimate within noise |
| E10 | leave-one-type-out, sibling gates only | — | — | — | 0/6 UNATTENDED_CASH … 3/5 NO_COUNTER videos of a held-out type still flagged |

Added in the final selection (2026-10-18). Every head was retrained today on the T4. Thresholds were fitted on the
fold head's own training scores (rows `N-*`), and the setup rows use the LOVO fold scores with thresholds fitted
leave-one-setup-out:

| Experiment | Condition | Recall | FP | Accuracy | Balanced | Reading |
|---|---|---|---|---|---|---|
| N-base | deployment recipe (train on all, thresholds on own scores), nested LOVO | 76% | 53% | 62.5% | 0.618 | honest familiar-camera estimate of the deployed recipe; E00's 0.697 used thresholds tuned on other heads (F15) |
| N-base-setup | same head, thresholds leave-one-setup-out | 76% | 42% | 67.5% | 0.670 | thresholds transfer across sessions |
| N-base-z-setup | + Šidák policy | 90% | 42% | 75% | 0.742 | Šidák gain appears only when the head has seen the session … |
| E06h vs E06h+E03e | session-blind head, max vs Šidák | 81% vs 81% | 58% vs 68% | 62.5% vs 57.5% | 0.615 vs 0.563 | … and reverses on an unseen session: not deployed |
| N-soup-setup / N-altsoup-setup | 3-seed soups (plain / alt-masked) | 76% / 76% | 53% / 53% | 62.5% / 62.5% | 0.618 / 0.618 | weight averaging does not improve session transfer (−0.052 vs N-base-setup) |
| N-alt-setup | alt-type windows masked | 76% | 63% | 57.5% | 0.565 | masking removes useful hard negatives; worse on new sessions |

Conclusion for the deployed configuration (A = release head + thresholds refit on its own scores, max rule): on the
in-data proxy for the jury's shift, the expected figure is E06h (recall ≈ 81%, FP ≈ 58%), not the LOVO numbers. None
of the tested policies or head variants narrows this gap.

Still to run (defined below): compression/blur/downscale/frame-rate transforms (need one GPU pass each), the generic
query tier (E11) and the timecode fine pass (E13).

### 3.1 Leave-one-setup-out and repeated group k-fold (E06)
For the baseline and every kept variant. Report per session: videos, recall, FP rate; the mean ± sd over 10 repeats of
group 5-fold shows how much the choice of held-out sessions moves the score.

### 3.2 Label-preserving test-time transforms (E14)
Score the unchanged production configuration on transformed copies of the 40 videos; count verdict flips and the mean
absolute change of the per-type max score. Transform recipes (all label-preserving for these violation types; the
horizontal flip is safe because no rule depends on left/right):

```bash
# geometry (same functions as ml/common.AUGS): hflip, rot_zoom, persp, flip_persp  → ml/score_dataset.py --augs ...
# resolution:   ffmpeg -i in.mov -vf scale=232:424 -c:v libx264 -crf 18 out_half.mov
# compression:  ffmpeg -i in.mov -c:v libx264 -crf 35 out_crf35.mov
# blur:         ffmpeg -i in.mov -vf gblur=sigma=2 -c:v libx264 -crf 18 out_blur.mov
# frame rate:   ml/score_dataset.py with runtime.FPS=1.0 (half the frames per window)
# brightness:   ffmpeg -i in.mov -vf eq=brightness=-0.15 -c:v libx264 -crf 18 out_dark.mov
```
Reading: a transform that flips more than 4 of 40 verdicts (10%) names a weakness; geometric flips point at scene
memorisation, compression/blur flips at small-object dependence (door, safe, banknotes), 1 fps flips at
temporal-resolution dependence (handovers, counting). Synthetic transforms cannot replace a real new angle; they
bound the problem from below.

### 3.3 Leave-one-type-out (E10, E11)
Hold out all videos showing type T (primary or alternative), fit thresholds and head on the rest, then ask: how
many of T's videos are still flagged (by which gate), at what false-alarm cost on normals. With the current gates
this is a cheap table from cached scores (`experiment.py --open-set`); with generic queries (E11) it becomes the
evaluation of the unseen-type tier. Expected pattern: UNATTENDED_CASH/DOOR_OPEN/SAFE_OPEN videos cross-fire among
themselves (they share "cashier leaves"), CIT and NO_COUNTER do not.

### 3.4 Length sensitivity (E03)
FP rate of normals by number of windows (1 / 2–4 / ≥ 5) for `max` vs `zsidak`/`top2`; recall of the three multi-window
violation videos must not drop.

## 4. Failure modes (first release) and their likely causes

| Observed | Videos | Likely cause | Lever |
|---|---|---|---|
| FP on hard normals: cashier leaves an empty desk, door closes behind | Х-23, Х-29, Х-31 (+Х-16) | the UNATTENDED_CASH/DOOR_OPEN gates respond to "cashier leaves" rather than to "cash remains"/"door stays open" | alt-type label fix (E02), bias calibration (E04), atomic contrast is already known not to work; real hard-normal clips from new angles (D1) |
| FP on long normals | Х-5, Х-14, Х-39 (all > 70 s) | max of 7–14 window scores | length-invariant aggregation (E03) |
| Misses on small-object states | DOOR_OPEN 0/4, SAFE_OPEN 0/1 | door/safe are tens of pixels; DOOR_OPEN got 3 contradictory negatives in training | E02; coarse-to-fine on the flagged window (E13) helps timecode, not detection; crops rejected (break on new angle) |
| Misses on manual counting | NO_COUNTER 0/4 | fine hand motion in 40-frame windows; machine vs hand is a subtle cue | 10 s windows raised type accuracy (team); E13; data (D1) |
| Wrong type on detected positives | type 43%, 7 low-confidence labels | shared head, 1–5 examples per type, label uncertainty | alt-accepting metric, second annotator (D2), E08 two events |
| Timecode off for states | tc ±2 s 56% on 9 videos | "onset at half peak" heuristic on a 20 s grounding curve; point labels | E13 fine pass with state-transition detection |

## 5. Strategy for unseen violation types

Three tiers, each with its own confidence band and wording in the API:

1. **Known types** (8 gates, catalogue rule, clause quoted) — as now, with the fixed confidence contract.
2. **Generic observations** (E11): a handful of rule-shaped, viewpoint-free questions in the same 3-option format
   ("Valuables are accessible while no employee is attending them / an employee is attending them / no valuables
   are visible"; "A person who is not the cashier is in the staff zone / only on the client side / nobody else";
   "An object is handed over across the desk without inspection / with inspection / no handover"; "A storage unit
   (safe, cabinet, drawer) is left open while unattended / closed after use / not used"). If a generic gate fires and
   no specific gate does, the service returns a violation verdict with `type = "Признак возможного нарушения (требует
   проверки)"`, the observation text of the generic gate, the timecode from its grounding curve, and **no legal
   clause** (criterion 4 forbids invented references; ToR §7 allows a human-review hint). The margin on generic
   gates is the open-set score; its threshold is set in leave-one-type-out so that the false-alarm cost on normals is
   explicit.
3. **Abstention band**: when the best margin of any gate is inside the Platt-calibrated uncertain band (E07) the
   verdict still follows the threshold (FAQ 15 penalises over-use of "insufficient data"), but the confidence shown is
   the calibrated probability, not the softmax.

Why not the alternatives: an embedding-distance anomaly score flags the new camera itself; a rule engine over
detector tracks is the right architecture for new rules but not buildable in the remaining time; learning a
classifier for "unknown" is impossible without unknowns. New rules later: add a `TypeQuery` and a `CATALOG` entry —
the model needs no retraining in zero-shot mode; the fine-tuned head, however, has only seen the 8 questions, so E11
must compare zero-shot vs head scores on the generic queries and the service may use the pretrained head for them.

## 6. Remaining limitations (honest list for the submission)

- All performance numbers are from familiar cameras; the different-angle robustness is unmeasured until D1.
- 40 videos: ±15 points on verdict accuracy; per-type results are counts.
- One event per video in production; same-type repeats not representable.
- Door/safe/manual-counting detection is weak at 464×848 from above; no zoom stage.
- Confidence is not yet a calibrated probability (E07 pending).
- Unknown types are missed until the generic tier (E11) ships; even then it is a hint, not a classification.
- 5-minute budget: 142 s videos take ≈150 s; two long uploads back-to-back can time out (audit F9).
