# Inference-time techniques for VLM/video-LLM violation detection (accuracy, calibration, temporal localization)

Scope note: this covers VideoJev-4B-style zero-shot or lightly tuned scoring of short overhead cash-room videos on one T4. Most cited evidence comes from public benchmarks (Charades-STA, MME, VideoMME, HR-Bench, NLU tasks). None of it is from bank or CCTV procedure footage, so every expected-impact number for our setting is an **inference**, labeled as such. T4 runtime and hour estimates are my engineering estimates, not measured values. Per-call latency of VideoJev-4B on the T4 should be measured first (one 20 s / 40-frame window call). Below that latency is called **L**.

---

## 1. Prompt and question engineering (contrastive options, atomic decomposition, consistency checks, evidence frames)

### Takeaway
There is public evidence that (a) binary yes/no probing with first-token probabilities beats asking a VLM for free-form outputs, and (b) breaking claims into atomic, verifiable sub-checks reduces hallucination. MLLMs are poor at *generating* good sub-questions themselves, so the decomposition should be hand-written per violation type and combined by explicit rules. Comparing a direct answer with a decomposed answer is a published reliability signal.

### Cited Findings
- Pelican breaks a visual claim into a chain of sub-claims built on first-order predicates and answers them with generated code. It reports about 8–32% hallucination reduction on MMHal-Bench across LVLMs. — [Pelican, arXiv 2407.02352](https://arxiv.org/html/2407.02352v2)
- DeCC ("Decompose and Compare Consistency") compares a VLM's direct answer with the answer it reaches by decomposing the question into sub-questions. The agreement is used to estimate answer reliability, not to raise accuracy directly. — [DeCC, EMNLP 2024](https://preview.aclanthology.org/setup/2024.emnlp-main.211)
- Existing MLLMs struggle to produce high-quality sub-questions. Selective decomposition (decompose only when needed) gives higher VQA accuracy after fine-tuning. — [Visual Question Decomposition on MLLMs, arXiv 2409.19339](https://arxiv.org/pdf/2409.19339)
- A 3B VLM trained with RL to self-question went from 46.8% to 52.2% on A-OKVQA. This needs training, so it is listed only as evidence that sub-questions help. — [search summary; source listing in VQD results](https://arxiv.org/pdf/2409.19339) *(note: the search engine attributed this result loosely; verify the exact paper before citing)*
- Training-free temporal grounding paper FV-Action argues that VLM failures on timestamps lie "in the task interface, not in perception". Asking "Does this clip show {query}? Answer Yes or No." and reading P(Yes) from first-token logits raised Charades-STA R@0.5 by 28–50 points across 4 backbones versus direct timestamp prompting. — [Your VLM Already Knows When, arXiv 2608.08315](https://arxiv.org/html/2608.08315)
- Test-Time Consistency (WACV 2026) is post-hoc and model-agnostic. It improves answer consistency across semantically equivalent rephrasings and perturbations on MM-R3 (image VQA). — [Chou et al., WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Chou_Test-Time_Consistency_in_Vision_Language_Models_WACV_2026_paper.html)
- MLLMs give confident answers even with no relevant image or with incongruent visual input, driven by language priors. This matches our "door left open ≈0.9 on almost every video" symptom. — [Debiasing MLLMs, arXiv 2403.05262](https://arxiv.org/html/2403.05262v3)

### Inferences (for our setting)
- **Replace the 3-option A/B/C gate with binary atomic probes plus rules.** "Door left open" ≈0.9 everywhere suggests the model answers from the question's prior, not the pixels. Atomic, observable checks are harder to answer from priors. Examples: "Is the door in this region open? Yes/No" (on a crop), "Is a person standing at the desk?", "Are banknotes visible on the desk?", "Is the safe door open?". A violation rule then combines them, e.g. `door_open AND no_person_in_room for > N s`. Each atomic probe can be calibrated separately (section 2) and checked against the 40 labeled videos. Expected impact (inference): the largest single lever on FP rate and type accuracy, because type confusion often comes from options that share visual evidence. **Effort: 6–10 h** to write probes and rules for all types and tune on 40 videos. **T4 cost:** each probe is a short prompt. With prefix/KV caching of visual tokens (section 6), extra probes cost a small fraction of L. Without caching, cost is about L per probe per window.
- **Contrastive pairs:** ask both "Is the door open?" and "Is the door closed?" and use P(yes|open) − P(yes|closed). This cancels part of the generic "yes" bias. It is a cheap consistency check: if both are high, the probe is unreliable for that window, so down-weight it. **Effort: 1–2 h.**
- **Evidence frame:** do not ask the model to name a frame number. Use the argmax window or frame of the calibrated probe score (section 5). That is the same interface FV-Action found works.
- Decomposition and rules should be hand-authored, because the model itself generates poor sub-questions (2409.19339).

### Gaps
- No public numbers on atomic-probe decomposition for *video* procedural compliance or surveillance. Gains for our case must be measured on the 40 labeled videos.
- Did not find a controlled study of 3-option (with "not applicable") versus binary formats for VLM MCQ bias.

---

## 2. Calibration of VLM option probabilities (contextual/content-free, batch calibration, bias subtraction, temperature/Platt on tiny data)

### Takeaway
Content-free calibration (Zhao 2021) gives large gains for text LLMs, but in multimodal form it is inconsistent. Image-free "None" inputs help, while noise or black-image inputs are unstable. Batch Calibration (subtract the mean score over a batch of real inputs) is consistently stronger and needs about 10 unlabeled samples. It maps directly onto "subtract the per-question mean on normal videos". With 40 labels, fit only 1–2 parameters per question: bias plus temperature, or 2-parameter Platt / logistic regression.

### Cited Findings
- Contextual calibration (Zhao et al. 2021): estimate the bias from the prediction on a content-free input ("N/A"), then fit an affine correction so that this prediction becomes uniform. It improves GPT-3/GPT-2 average accuracy by up to 30.0% absolute and reduces variance across prompts. — [Calibrate Before Use, arXiv 2102.09690](https://arxiv.org/abs/2102.09690)
- Batch Calibration (BC), the formula: ŷᵢ = argmax_y [p(y|xᵢ,C) − p̂(y|C)], with p̂(y|C) = mean over M unlabeled test inputs. It is zero-shot, inference-only, with negligible cost. A few-shot variant learns the bias from labeled data. — [Batch Calibration, ICLR 2024, arXiv 2309.17249](https://arxiv.org/html/2309.17249v2)
- BC versus contextual calibration (CC) and domain-context calibration (DC), average accuracy: PaLM 2-S zero-shot BC 64.09 / CC 59.42 / DC 58.88; PaLM 2-S 1-shot 74.41 / 62.39 / 64.98; PaLM 2-L 1-shot 81.09 / 77.83 / 76.89. On CLIP ViT-B/16 zero-shot over 6 image datasets, BC raised accuracy from 36.7% to 41.5%. — [arXiv 2309.17249](https://arxiv.org/html/2309.17249v2)
- BC reaches strong performance with about 10 unlabeled samples. Prototypical calibration needs more than 500. — [arXiv 2309.17249](https://arxiv.org/html/2309.17249v2)
- CC failure case: content-free inputs "can be inappropriate prior estimators", e.g. PaLM 2-S QNLI 1-shot dropped from 81.20% (uncalibrated) to 56.86% with CC. — [arXiv 2309.17249](https://arxiv.org/html/2309.17249v2)
- Multimodal content-free calibration ("Post-Hoc Debias"): replace the image with "None"/"Unk" or meaningless tensors and calibrate to uniform. LLaVA-1.5-7B MME overall went from 891.19 to 952.50 with "None" and 932.62 with "Unk", but fell to 860.83 with "Both". Image-based content-free inputs (noise, zero, one) gave inconsistent gains with higher variance. The reasoning subset dropped (317.86 → 302.50), while gains came from the hallucination subsets (MME-AH 260 → 315). The largest benefit is on low-confidence predictions. — [Debiasing MLLMs, arXiv 2403.05262](https://arxiv.org/html/2403.05262v3)
- Temperature scaling learns one scalar and preserves the ranking, so it changes calibration but not accuracy. Platt scaling (2 parameters) is data-efficient but mostly suited to binary tasks. Matrix scaling's parameters grow as K². Thermometer reportedly outperforms TS when labeled data is below 30 by predicting temperature from unlabeled data. — [Thermometer, arXiv 2403.08819](https://arxiv.org/pdf/2403.08819); [LLMs are Miscalibrated ICL, arXiv 2312.13772](https://arxiv.org/pdf/2312.13772)
- One source recommends pairing ECE with Brier score when evaluating calibration. — [search summary of calibration literature](https://www.kdnuggets.com/a-deep-dive-into-calibration-of-language-models-platt-scaling-isotonic-regression-temperature-scaling) *(secondary source)*

### Inferences (for our setting)
- **Per-question bias subtraction = few-shot BC.** For each probe q, compute the mean (or median) logit of the "yes"/A option over all windows of normal (no-violation) videos. Subtract it, then threshold. This is BC with labels, and it directly attacks "door open ≈0.9 everywhere". Expected impact (inference): large reduction of the 37–63% FP rate for the most biased questions. **Effort: 1–2 h** (offline on cached scores). **T4 cost: zero** at inference.
- **Use log-odds, not raw probabilities.** Work with logit(A) − logit(B), with C ("N/A") as a separate feature. Saturated probabilities near 0.9 compress differences.
- **Within-video temporal BC (key inference):** violations here are mostly *events in time*. Score each window as s_w − median_w(s) within the same video, plus the global per-question bias. This removes per-video scene bias (lighting, camera angle, room layout). It fails for violations that persist through the whole video (e.g. door open all video). So keep both features, absolute calibrated score and relative peak height, and let a tiny logistic regression combine them. **Effort: 2 h.**
- **Content-free video input** (black or blurred frames, or text-only "no video"): cheap to try (1 h, L per question once, cacheable), but the evidence says image-based content-free inputs are unstable (2403.05262), and CC lost to BC in every PaLM table. Priority: low. BC on real normal videos is better grounded.
- **Tiny-data fitting with 40 videos:** per type, fit at most 2 parameters (bias + scale, i.e. Platt / 1-D logistic regression on the calibrated max-window logit). Select thresholds by leave-one-video-out CV. Never fit a per-type K×K matrix. If types are mutually exclusive per video, add a softmax over types with a shared temperature. Report Brier score alongside accuracy. **Effort: 2–3 h.**
- **Max-over-windows bias:** the max of N noisy window scores grows with N, so longer videos (145 s ≈ 7+ windows versus 1 window for 10 s) get systematically more FPs at a fixed threshold. Fix: use a length-aware threshold (calibrate on the max of N windows), top-2 mean, or the within-video-relative peak above. **Effort: 1 h.** Likely part of the FP problem (inference; check whether FPs correlate with duration in the 40 labeled videos).

### Gaps
- No published calibration study for *video* LLM per-question bias on surveillance-type footage.
- Thermometer needs a trained auxiliary model across tasks, so it is not directly usable. No tiny-data (n≈40) calibration benchmark for VLMs found.

---

## 3. Self-consistency / test-time augmentation / ensembling for video

### Takeaway
Label-free test-time methods on video LLMs give about +3–4 points average accuracy on video QA benchmarks. Frame-subset self-consistency is a solid baseline, but it is expensive: K subsets × N samples. For a probability-scoring pipeline, deterministic TTA (frame-offset shifts, small spatial jitter) averaged in logit space is the cheap analogue. Expect modest gains, mostly in variance reduction. No public evidence quantifies multi-VLM ensembles for our kind of task.

### Cited Findings
- TTA-Vid (test-time RL with frame-subset sampling and a bandit for frame importance) gives InternVL3-8B 51.37 → 55.13 (+3.76) and Qwen2.5-VL-7B 48.82 → 51.84 (+3.02), averaged over VideoMMMU, MMVU, SciVideoBench, VideoMME and LongVideoBench. Defaults: K=4 subsets of 4 frames, N=8 rollouts. Adaptation used 4× A100 40GB, so it is not T4-feasible as is. — [TTA-Vid, arXiv 2604.00696](https://arxiv.org/html/2604.00696v1)
- In that paper, plain self-consistency on MMVU got 62.72% versus TTA-Vid's 65.60%. Self-consistency was about 35% slower than TTA-Vid in one appendix setup (InternVL3-2B, 18,617 s). Runtime scales about 2.1× when the number of subsets doubles. — [arXiv 2604.00696](https://arxiv.org/html/2604.00696v1)
- A CVPR 2026 VidLLMs Challenge entry used answer self-consistency (multiple stochastic passes, then vote) plus re-asking the question only when the vote margin is low. The authors report substantial gains over single-pass inference. — [ASC-MQRA, arXiv 2606.04323](https://www.alphaxiv.org/abs/2606.04323v1)
- On a static-image MCQ benchmark, sampling 8 → 16 chains added only 0.15 pp for small VLMs. — [Test-time scaling small VLMs, arXiv 2506.22395](https://web3.arxiv.org/pdf/2506.22395)
- FV-Action ablation: 5 frames per probe improved over 3 frames. More temporal context per probe helps grounding. — [arXiv 2608.08315](https://arxiv.org/html/2608.08315)

### Inferences (for our setting)
- **Margin-triggered re-check (cascade TTA):** run TTA only on windows or videos whose calibrated score is within ±δ of the threshold, as in ASC-MQRA's low-margin arbitration. Use 2–3 views: frame offset +0.25 s (shifted 2 fps sampling grid), a crop or zoom view (section 4), and a horizontal flip only if the procedure is not left/right-sensitive. Average logits. **Effort: 3–4 h. T4 cost:** about 2–3 × L per borderline window, typically only 10–30% of windows. Expected impact (inference): a few points of accuracy and fewer threshold flips. Smaller than the gains from calibration and atomic probes.
- **Different fps as augmentation:** 1 fps over 40 s versus 2 fps over 20 s gives different temporal context. Useful for long-duration violations (cash left unattended); less useful for brief actions.
- **Ensembling a second VLM** (e.g. Qwen2.5-VL-3B/7B or InternVL, 4-bit on T4, both Apache-2.0 / MIT-style licenses; check each license): roughly doubles cost. Only justified as a borderline re-checker. No evidence found on gains for surveillance/procedural tasks. **Effort: 6–10 h** for integration and memory juggling on 16 GB.
- Stochastic sampling self-consistency is the wrong fit: we read probabilities, not sampled text. Deterministic input perturbations are the right TTA.

### Gaps
- No public study isolating per-augmentation gains (crop, flip, offset) for video LLM classification. No multi-VLM ensemble numbers for event detection.

---

## 4. Visual focus for small objects (zoom crops, tiling, attention-guided crops, visual CoT, set-of-mark)

### Takeaway
Training-free cropping and zooming reliably and substantially improves MLLM accuracy on small details. Example: LLaVA-1.5 on V* went 42.4 → 62.3 with attention-guided crops; ZoomEye gave +15–18% on HR-Bench for InternVL2.5-8B. For a fixed overhead camera, the cheapest and probably most effective form is **static ROIs** (door, safe, desk) defined once and fed as zoomed crops with the full frame.

### Cited Findings
- MLLM zero-shot VQA accuracy is very sensitive to the size of the relevant visual subject, declining by up to 45.91% as the subject shrinks. — [MLLMs Know Where to Look (early version), arXiv 2310.16033](https://arxiv.org/html/2310.16033v2)
- ViCrop (ICLR 2025): MLLMs "consistently know where to look" even when they answer wrongly. It crops using the model's own attention (rel-att, grad-att, pure-grad) and concatenates original and crop tokens to keep global context. LLaVA-1.5 V*: 42.41 → 62.30; TextVQA 47.80 → 55.17. — [ViCrop, arXiv 2502.17422](https://arxiv.org/pdf/2502.17422)
- ViCrop needs full attention weights, which makes it incompatible with efficient attention (FlashAttention). — [FOCUS, arXiv 2506.21710](https://arxiv.org/html/2506.21710v2)
- ZoomEye (training-free tree search over zoomed sub-regions): InternVL2.5-8B +15.71% and +17.69% on HR-Bench subsets. Small 3–8B models beat GPT-4o on some detail tasks, but tree traversal is costly. — [ZoomEye, arXiv 2411.16044](https://arxiv.org/html/2411.16044v3); [cost critique in FOCUS](https://arxiv.org/html/2506.21710v2)
- Zoom-Refine (zoom plus self-refinement): +3.60–5.07% on MME-RealWorld reasoning and +2.1–5.0% on HR-Bench. — [alphaXiv summary 2506.01663](https://alphaxiv.org/overview/2506.01663v2) *(secondary summary; verify)*
- VTimeCoT (training-free) overlays a progress bar and highlighting on frames as visual tools for video grounding and reasoning. This is a "visual prompting in time" variant. — [VTimeCoT, arXiv 2510.14672](https://arxiv.org/abs/2510.14672v1)

### Inferences (for our setting)
- **Static ROI crops (top recommendation):** the camera is fixed and overhead, and the frame is 464×848. Define 2–4 ROIs per room layout (door, safe, cash desk, teller position) by hand on one frame. If layouts vary, use a quick detector or one VLM grounding call per video. Upscale each crop to the model's native tile size and ask that ROI's atomic probes on the crop sequence (e.g. door probes on door-crop frames at 1–2 fps). Objects tens of pixels wide become hundreds, which is exactly the size-sensitivity regime documented above. **Effort: 4–6 h** (ROI definition, crop pipeline, prompt re-tune). **T4 cost:** crop clips are small. Visual tokens scale with crop area × upscale, so per-ROI calls are likely ≤ L. Total roughly 1–2 × L per window for all ROIs. Expected impact (inference): large for door, safe and banknote types, likely the main fix for "door open ≈0.9" combined with calibration.
- **Full frame + crop concatenation** (ViCrop's trick) keeps context, e.g. whether a person is near the door. Costs more tokens; use it for the re-check stage only.
- **Attention-guided ViCrop:** on a T4 FlashAttention-2 is not supported anyway (sm75), so eager attention is available. Still, extracting attention maps for a video model is extra engineering (8+ h) for something static ROIs already give us. Low priority.
- **Set-of-mark / box overlays:** draw a colored box and label on the ROI in the full frame ("Is the door inside the red box open?"). Cheap (1–2 h), no token increase. Worth an A/B test against crops. No numbers fetched here (see Gaps).
- Avoid ZoomEye-style tree search: too many calls for a 5 min budget across many windows and types.

### Gaps
- Did not fetch Set-of-Mark (Yang et al. 2023) results, so SoM gains on small-object VQA remain unverified in these notes.
- No published zoom/crop study on *video* LLMs for small-object state (door open/closed) over time. All evidence is image VQA.

---

## 5. Temporal localization (dense scoring, peak picking, boundary refinement, change-points, training-free grounding)

### Takeaway
The strongest 2026 training-free evidence (FV-Action) is coarse binary-probe scanning → NMS top-3 peaks → a fine scan of 8 clips around each peak → geometric mean of coarse and fine scores. Over 4 frozen 7–8B VLMs it lifted Charades-STA R@0.5 from 0–28% to 40–57%, at about 36 VLM calls per video and about 10–14 s on an A100. This fits our ±2 s timecode problem almost exactly. Our grounding is point-like, so the fine-scan resolution, not window width, sets the error.

### Cited Findings
- FV-Action procedure: K_c clip centres uniformly spaced (12 for Charades), one Yes/No call each, P(Yes) from first-token logits. NMS keeps N_p=3 candidates (suppression radius 8 s on Charades). Fine scan of K_f=8 clips around each candidate. Final peak = argmax of the geometric mean of coarse and fine scores (α=0.5). Then a fixed-width window centred on the peak. Calls = K_c + N_p·K_f (36 on Charades). Probes are batched and need no autoregressive decoding. — [arXiv 2608.08315](https://arxiv.org/html/2608.08315)
- Charades-STA R@0.5 / R@0.7, direct regression → yes/no scan: InternVL2-8B 3.76/1.08 → 53.39/26.64; Qwen2.5-VL-7B 28.41/14.57 → 56.77/29.68; LLaVA-NeXT-Video-7B 0.05/0.00 → 47.04/23.06; mPLUG-Owl3-7B 2.31/1.02 → 39.68/20.22. — [arXiv 2608.08315](https://arxiv.org/html/2608.08315)
- Median latency per sample: 14.4 s on a shared A100 MIG slice and about 10.3 s on a dedicated A100 (Charades). 95.6 s for TACoS (120 calls). — [arXiv 2608.08315](https://arxiv.org/html/2608.08315)
- Geometry finding: with a fixed output window W and event length g, IoU ≥ 0.5 requires g ∈ [W/2, 2W]. The peak-ceiling correlates with final R@0.5 at r=0.998. The fixed window, not peak localization, limits mid/long events. Adaptive width gave +3.4 pp on ActivityNet. — [arXiv 2608.08315](https://arxiv.org/html/2608.08315)
- The paper is internally inconsistent on frames per probe (3 vs 5) and on some window widths. No code link found. — [arXiv 2608.08315](https://arxiv.org/html/2608.08315)
- ZEAL (training-free): an LLM splits an action into start and end descriptions, an LVLM scores every frame for each, and segments come from pairing the most confident start and end frames. — [ZEAL, arXiv 2410.14340](https://arxiv.org/html/2410.14340v1)
- TFVTG (ECCV 2024): an LLM decomposes a query into sub-events, each split into dynamic-transition and static-status parts and scored with VLM-based dynamic/static scoring functions. Best zero-shot results at the time on Charades-STA and ActivityNet Captions. — [TFVTG, arXiv 2408.16219](https://arxiv.org/html/2408.16219v1)
- OZ-TAL (online zero-shot TAL): per-frame VLM classification with background-aware k-way classification to penalize low-confidence predictions, plus a class-specific state machine that assembles segments. — [OZ-TAL, arXiv 2605.09976](https://arxiv.org/pdf/2605.09976)
- T3AL adapts a VLM per video at test time and filters proposals with captioner-generated descriptions. — [T3AL, arXiv 2404.05426](https://arxiv.org/html/2404.05426v1)
- LAVAD (CVPR 2024, training-free VAD): per-frame captions → LLM anomaly scores, refined by aggregating scores among frames with semantically similar temporal summaries (cross-modal similarity). Beats unsupervised and one-class methods on UCF-Crime and XD-Violence. — [LAVAD, arXiv 2404.01014](https://arxiv.org/html/2404.01014v1)

### Inferences (for our setting)
- **Use the detection probe itself as the grounding signal**, not a separate free-form grounding query. Pipeline: (1) coarse: the existing 20 s windows, or better 4–6 s clips at 2 fps with stride 2 s for videos ≤145 s (≈70 clips max, batched). (2) Take the top-3 peaks of the *calibrated* probe curve (section 2) with NMS radius ≈5 s. (3) Fine: 8 clips of 2–3 s at 4–8 fps centred at offsets ±0.5 s around each peak. (4) Timecode = argmax of the geometric mean of coarse and fine scores. **Effort: 4–6 h. T4 cost:** fine stage ≈ 24 short-clip calls, each much cheaper than L (fewer frames). Likely 15–40 s per video for the probed type only (estimate; measure). Expected impact (inference): moving from 56% within ±2 s toward the 70–80% range is plausible, since FV-Action-style peaks are much sharper than direct timestamp prompts. This is unverified for our data.
- **Define the timecode event as a *state transition*** (ZEAL/TFVTG idea): ask "Is the door open?" frame-wise and locate the closed→open transition, instead of "when does the violation happen?". Change-point detection on the 1-D probe-probability series (e.g. PELT/binary segmentation via `ruptures`, or simply the argmax of the smoothed derivative) gives frame-accurate boundaries cheaply. **Effort: 2–3 h. T4 cost:** negligible beyond the probe series.
- **Smoothing:** median or Gaussian smoothing of the dense score curve before peak picking (in LAVAD's spirit). Hysteresis or state-machine logic (OZ-TAL) for "duration > N s" violations, e.g. cash left on desk unattended.
- **Embedding change-point (no VLM):** cosine-distance change-points on cheap per-frame embeddings (e.g. a SigLIP/CLIP-class encoder on ROI crops, commercially licensed) can propose candidate times at 4–8 fps for almost no cost. VLM probes then verify them. Useful when the probe curve is flat. **Effort: 3–4 h.**
- The geometry finding matters less for us because we output a point, but it implies fine-scan step size ≤0.5 s is needed for exact-frame scoring.

### Gaps
- No public results for training-free grounding on short, single-camera procedural videos. FV-Action's code is not located. Timing numbers are A100, not T4. A T4 is likely 3–6× slower for 7B fp16 (my estimate, not sourced).

---

## 6. Hierarchical / cascade inference to fit 5 min on T4; feature caching across questions

### Takeaway
The cited methods all point to a coarse-to-fine cascade with batched, decoding-free first-token probability reads. FV-Action does it in ~36 calls per video. The biggest practical saving for our multi-question setup is encoding each window's visual tokens once and reusing the prefix KV cache across all probe questions.

### Cited Findings
- FV-Action probes are batched and need no autoregressive decoding (single forward, first-token logits), with ~36–40 calls per video on short benchmarks. — [arXiv 2608.08315](https://arxiv.org/html/2608.08315)
- Margin-triggered re-arbitration (re-ask only low-margin cases) is an effective compute allocation in video QA. — [ASC-MQRA, arXiv 2606.04323](https://www.alphaxiv.org/abs/2606.04323v1)
- ViCrop-style attention methods conflict with FlashAttention, which is relevant for memory and speed. — [FOCUS, arXiv 2506.21710](https://arxiv.org/html/2506.21710v2)

### Inferences (for our setting)
- **Proposed budget (estimate; validate by timing L on the T4):**
  1. Stage 0 (cheap, ~5–15 s): decode at 2–4 fps, optional embedding change-points per ROI.
  2. Stage 1 (coarse, main cost): per 20 s window, encode the video once, then run all atomic probes (≈10–20 short prompts) on a shared visual prefix, with the KV cache reused or cloned per question. Cost ≈ L + n_probes × small decode-free suffix pass.
  3. Stage 2 (verification, only for types whose calibrated score exceeds the lower threshold): ROI-crop probes plus TTA views on top windows.
  4. Stage 3 (localization, only for the predicted type): fine scan around the top-3 peaks.
- **Prefix/KV caching:** put the video tokens first and the question last in the prompt, so that the shared prefix is reusable. In HF transformers this means running the prefix forward once, then `past_key_values` copies per question (memory: a 4B model with a few thousand visual tokens in fp16 should fit in 16 GB; verify). Alternatively, vLLM automatic prefix caching, if VideoJev-4B is supported and vLLM works on sm75 (T4 lacks bf16 and FA2; use fp16 and SDPA/xformers). **Effort: 3–6 h.** Expected saving (inference): roughly a factor of (#questions) on the language side. The vision encoder runs once per window instead of once per question, which is likely the single largest speedup if the current code re-encodes per question.
- **Cache vision-encoder outputs on disk** during development so calibration, rule and threshold experiments on the 40 videos rerun in minutes without the GPU. **Effort: 1–2 h.** This also makes the section 2 calibration work cheap.
- Cut 3 options to Yes/No first-token reads with no generation: no `generate()`, one forward per probe.

### Gaps
- No T4 latency figures found for VideoJev-4B or similar 4B video LLMs. Whether VideoJev-4B's processor allows reusing visual-token KV caches across prompts depends on its implementation (not verified).

---

## Summary priority table (all impact and effort values are inferences/estimates)

| # | Technique | Evidence strength | Expected impact here | Effort | T4 runtime cost |
|---|---|---|---|---|---|
| 1 | Per-question bias subtraction on normal videos (few-shot BC) + length-aware max threshold | Strong (BC, ICLR'24) | High on FP rate | 2–3 h | ~0 |
| 2 | Static ROI zoom crops for door/safe/cash | Strong (ViCrop, ZoomEye; image VQA) | High on small-object types | 4–6 h | +0.5–1× L per window |
| 3 | Atomic Yes/No probes + rules (replace A/B/C) | Moderate (Pelican, FV-Action interface) | High on type accuracy and FPs | 6–10 h | small with KV cache |
| 4 | Coarse-to-fine yes/no scan + peak/NMS + fine re-scan for timecode | Strong (FV-Action: +28–50 pp R@0.5) | High on ±2 s timecode | 4–6 h | +15–40 s/video (est.) |
| 5 | Change-point on probe curve / embeddings for transition time | Moderate (ZEAL, TFVTG ideas) | Medium-high on exact-frame | 2–4 h | negligible |
| 6 | Visual-prefix KV caching, decode-free logits | Engineering | Enables 1–5 in budget | 3–6 h | saves a lot |
| 7 | Logistic/Platt per type, LOO-CV on 40 videos | Standard | Medium (calibration, thresholds) | 2–3 h | 0 |
| 8 | Margin-triggered TTA (offset, crop, fps) | Moderate (TTA-Vid +3–4, ASC-MQRA) | Low-medium | 3–4 h | +2–3× L on borderline only |
| 9 | Contrastive negation probes | Weak-moderate | Medium on biased questions | 1–2 h | +1 probe each |
| 10 | Content-free (blank video) calibration | Mixed (unstable in MLLMs) | Low | 1 h | one-off |
| 11 | Second VLM ensemble | No direct evidence | Unknown | 6–10 h | ~2× |
