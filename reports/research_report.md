# Research report — what the literature says about our specific failure modes

Date: 2026-10-18. This report does not repeat the five surveys in `research_notes/` (few-shot and reference-based
anomaly methods, local video VLMs, fine-tuning and data strategies, inference-time techniques, industry practice) or
the synthesis in `reports/Методы улучшения видеодетекции нарушений.md`. It (1) verifies the facts about the model we
actually run, (2) fills the gaps those notes left — evaluation at n = 40, aggregation statistics, calibration,
open-set recognition, label noise — and (3) maps every recommendation to an experiment ID in `experiment_plan.md`.
Claims are tagged **[verified]** (read in the source), **[reported]** (taken from the team's notes or a secondary
source) or **[hypothesis]** (our inference for this task).

## 1. The model we run: VideoJev-4B

Source: Zhu et al., *VideoJev: System One Model for Video Understanding and Interaction*, arXiv:2610.18615
(Oct 2026), code github.com/MCG-NJU/VideoJev, weights `ZhuYuhan/VideoJev-4B`, Apache-2.0 (LICENSE and NOTICE in the
repo; backbone Qwen3.5-4B Apache-2.0). All **[verified]** unless noted.

- Candidate scoring, not generation: the sequence is video tokens → task instruction → question → candidates, each
  ended by `<|option_end|>`, then `<|decide|>`. A cosine head (two 2560→512 projections, fixed scale 10, learned
  bias) scores each candidate state against the decision state; softmax for choice questions, sigmoid per
  frame-pair for grounding. Our `train_head.py` re-implements exactly this head.
- Training: stage 1 trains only the head (30% of the data, head lr 3e-4), stage 2 the backbone + markers + head
  (lr 5e-6 for 4B) on TimeLens-100K/TimeLens2-93K for grounding and an internal ≈240 K-question QA corpus with 4–5
  options; options were permuted during training to reduce position bias; 64×H200, 2 fps, 144 K context.
  Our head-only fine-tuning is the published stage 1 recipe applied to 40 videos.
- Calibration: mean ECE 8.65 points (4B) on four QA benchmarks, over-confident; the authors suggest temperature
  scaling and did not apply it. Our UI shows the softmax as "уверенность" (audit F6).
- Joint inference: several questions share one video prefix in one causal forward; later questions see earlier
  question text. This is what `prepare_query_groups` does for our 8 gates; question order is a nuisance variable
  (audit F11, E12).
- Inference settings in the paper: 2 fps, ≤ 640×640 px per frame, 80% visual-token budget — matching
  `runtime.prepare_window`.
- Not in the paper: any out-of-domain evaluation, surveillance data, or small-object analysis beyond one qualitative
  failure (counting four books). Transfer of its benchmark numbers to a cash room is **[hypothesis]**.

Consequences. (a) The gate questions are in-distribution for the *format* (single-choice with letter prefixes) but
out-of-distribution in option count (2–3 vs 4–5) and content; (b) a new violation type is just a new question —
the architecture supports zero-shot queries by design, which is the basis of the open-set tier (E11); (c) there is
no published evidence that head-only training generalizes across camera placements; that is what E06 measures.

## 2. Evaluating with 40 videos

- Varoquaux, *Cross-validation failure: small sample sizes lead to large error bars*, NeuroImage 2018,
  arXiv:1706.07581 **[verified abstract]**: with ~100 samples the confidence interval of a cross-validated accuracy is
  about ±10 points; the standard error across folds under-estimates it; leave-one-out is unstable and biased, repeated
  random/group splits are preferred. Our 40 videos give ±15 points on verdict accuracy (Wilson), which is why
  `evalkit` reports intervals and repeated group k-fold rather than a single LOVO number.
- Cawley & Talbot, *On over-fitting in model selection and subsequent selection bias in performance evaluation*,
  JMLR 11 (2010), jmlr.org/papers/v11/cawley10a.html **[verified title/claim]**: selecting hyper-parameters or variants
  on the same folds that produce the reported score biases the score upward; nested cross-validation removes it.
  The first release compared ≥ 7 variants on one LOVO protocol (audit F3). Remedy: nested thresholds (done), a frozen
  self-recorded test set (D1), and reporting the number of variants tried.
- Group structure: standard practice in medical imaging and surveillance is to split by patient/camera, never by
  frame; our analogue is the camera session (`analysis/setups.json`). **[verified by data]** 8 sessions with
  violation/normal twins.
- Point labels: with one timecode per event, "temporal IoU" is undefined; tolerance curves (1/2/5 s) and event-level
  precision/recall at a tolerance are the standard substitutes (as in AI City Track 3 scoring, reported in
  `research_notes/industry_practice.md`).

Recommendation → E00, E06, protocol §3 of the plan.

## 3. Aggregation over windows and the length effect

- Weakly supervised VAD treats a video as a bag of snippets and aggregates with the maximum or the mean of the top-k
  snippet scores. Fixed-k selection ignores video length and the abnormality ratio: BN-WVAD (arXiv:2311.15367) and
  dynamic-k MIL in AR-Net (arXiv:2011.12077) **[reported]** replace the max by duration-aware k-max; RTFM
  (arXiv:2101.10030) shows separation peaks when k ≈ the number of abnormal snippets.
- Statistics: for N exchangeable window scores, the maximum exceeds a fixed threshold with probability
  1 − (1 − p)^N; converting each window to a p-value against normal windows and combining with the Šidák bound makes
  the video-level false-alarm rate independent of N **[verified, textbook]**. Implemented as `decision.Policy(agg="zsidak")`
  (Gaussian tail on normal-window logits; the empirical variant has a 1/(n+1) resolution floor with ~60 reference
  windows). Evidence in our data: all normals > 70 s are false positives (audit F4) — suggestive at n = 19.
- Temporal persistence (k-of-n, dwell timers) is the industry's main false-alarm lever (`industry_practice.md`), but
  27 of our 40 videos are a single window, so it only applies to long clips; implemented as `top2`.

Recommendation → E03; keep `max` if E03 is not better on leave-one-setup-out.

## 4. Calibration and thresholds with tiny data

- Per-question bias: Batch Calibration (Zhou et al., ICLR 2024, arXiv:2309.17249) subtracts the mean score over a
  batch of real inputs and works with ~10 unlabeled samples **[reported in research_notes/inference_time_techniques.md]**;
  the multimodal debiasing study (arXiv:2403.05262) shows content-free image inputs are unstable, so real normal windows
  are the right reference. Implemented as `calib="bias"` (median logit of normal training windows).
- Probability calibration: temperature scaling (Guo et al., ICML 2017, arXiv:1706.04599) and Platt scaling (two
  parameters) are the data-efficient choices; Niculescu-Mizil & Caruana (ICML 2005) found Platt safer than isotonic
  regression on small sets **[verified, classic]**. With 40 videos, calibrate the *video margin* with two parameters,
  nested in the folds (E07); never fit per-type calibration curves.
- Thresholds: the coordinate-ascent with per-type corrections (`calibrate.fit`) is a 9-parameter search on 39 videos;
  shrinkage toward the global threshold or a single threshold is the standard small-sample remedy
  **[hypothesis for our data]** (E05).
- Conformal risk control gives a formal false-negative bound only under exchangeability; a new camera breaks it
  (`finetuning_data_strategies.md` §6) — report empirical intervals instead.

## 5. Unseen violation types: closed-set, zero-shot, embedding anomaly, hybrid rules

Framing (not to be confused): *detecting that something unusual happened* vs *naming the legal violation*. The ToR
scores the verdict (25) separately from the type (15); an unknown event correctly flagged with an honest "needs
review, no matched rule" earns verdict points without inventing a rule (criterion 4 penalises invented references).

| Approach | Mechanism | Evidence | Fit for us | Experiment |
|---|---|---|---|---|
| Closed-set gates (current) | one question per type, thresholds | our LOVO numbers | misses unknown types by design; shared head may over-specialise to 8 questions | E10 (how often sibling gates still fire) |
| Zero-shot generic queries | rule-shaped descriptions scored by the same model (new text, no training); "needs review" tier | AnyAnomaly's text-defined anomalies, VERA's learned guiding questions (`fewshot_anomaly_methods.md`) **[reported]**; VideoJev scores arbitrary candidates **[verified]** | cheapest route; risk of "yes" bias (team's atomic-question result) → use the 3-option format with an explicit normal alternative and bias calibration | E11 |
| Open-set scoring on the closed-set classifier | max logit / margin as an unknown-vs-known score; abstain when all gates are weak but a generic gate is strong | Vaze et al., ICLR 2022, arXiv:2110.06207: a good closed-set classifier's max logit is a strong open-set baseline **[verified abstract]**; MSP (Hendrycks & Gimpel, arXiv:1610.02136) and energy scores (Liu et al., arXiv:2010.03759) **[verified, classic]** | free: `video_margin` already exists; selective prediction (Geifman & El-Yaniv, arXiv:1705.08500) formalises the "needs review" band | E07/E10 |
| Embedding anomaly (kNN to normals, V-JEPA 2 / SigLIP) | distance of a window to normal windows | zero-shot CLIP is weak on VAD (53 AUC, `fewshot_anomaly_methods.md`); a new camera is itself novel → FP **[reported + hypothesis]** | not as a verdict; at most a localization prior | deferred |
| Hybrid rules over visual predicates | detector/pose → states (door open, person count, cash visible) → deterministic rules; new type = new rule | industry pattern (`industry_practice.md`); AnomalyRuler-style rules transfer across views better than embeddings **[reported]**; the team's RT-DETR trial did not separate violation from normal | the right long-term architecture; too costly for the remaining hackathon time; keep the rule catalogue extensible (`rules.CATALOG` already is) | after the hackathon |

Recommendation: ship the generic-query tier with a distinct confidence band and no legal clause unless a specific
gate also fires (E11), and quantify it with leave-one-type-out, where the held-out type's videos are the "unknown"
set and the normals bound the false-alarm cost.

## 6. Label noise and partial labels

Six of 22 events list alternative types; the first release trained those as negatives (audit F2). In the
positive-unlabeled framing (Elkan & Noto, KDD 2008) an unlabeled instance is not a negative; masking or soft labels
are the standard treatment **[verified, classic]**. Expected effect is concentrated on DOOR_OPEN and SAFE_OPEN, the
types with the fewest positives → E02. Also: a third of type labels are self-rated "low"; a second annotator (D2) is
cheaper than any model change for the type criterion.

## 7. Variance reduction for the head

Model soups (Wortsman et al., ICML 2022, arXiv:2203.05482) **[verified abstract]**: averaging weights of models
fine-tuned from one initialisation improves accuracy and robustness at no inference cost — directly applicable to a
2.6 M-parameter head trained from the pretrained weights (`train_head.py --seeds 5 --ensemble soup`). Frozen-feature
augmentation (FroFA, arXiv:2403.10519, `finetuning_data_strategies.md`) **[reported]** motivates `--feature-noise`.
Deep ensembles (Lakshminarayanan et al., arXiv:1612.01474) would need several heads at inference; the soup is the
deployable form → E09.

## 8. Viewpoint robustness

The notes are consistent: geometric augmentation of one view helps less than genuine multi-view data
(SURREACT, multi-view recognition studies, `finetuning_data_strategies.md` §4) **[reported]**; test-time per-video
normalisation adapts to a new scene without labels (NOVA, arXiv:2609.06360, builds a per-video normality anchor
from early frames **[verified abstract]**; T3AL adapts per video for localization) but needs several windows — only 13
of our videos have them. Therefore: self-recorded clips (D1) and leave-one-setup-out (E06) are the measurement; the
test-time transform suite (E14) is the cheap proxy that tells which perturbations the system is blind to.

## 9. Localization

FV-Action-style coarse-to-fine yes/no scanning and state-transition detection (`inference_time_techniques.md` §5)
**[reported]** are the best-supported training-free routes; VideoJev's own grounding head already gives a dense curve,
so the fine stage is a second pass at 4 fps with 6 s clips around the best window for the flagged type only (E13).
Published numbers are on Charades-STA with 7–8B VLMs; transfer is **[hypothesis]**.

## 10. What we recommend not doing now

Backbone LoRA (overfits 8 sessions; weak published gains on anomaly QA even with 10⁵ samples), a second VLM as
verifier (memory and time on a T4, no evidence for this task), embedding-kNN as a verdict, synthetic video, fixed ROI
crops (break on the new angle; the team's OWLv2 trial failed). Details and licenses in the notes.

## 11. Verified vs hypothetical, in one list

Verified in sources: VideoJev architecture, training, license and ECE; Varoquaux/Cawley–Talbot on evaluation; Vaze et
al. on max-logit open-set; Guo/Platt on calibration; Šidák arithmetic; model soups; our dataset facts.
Reported by the team's notes (not re-read here): Batch Calibration numbers, AnyAnomaly/VERA/AnomalyRuler results,
FV-Action grounding gains, license table of alternative models.
Hypotheses to be tested by the plan: every expected-effect column in `experiment_plan.md`.
