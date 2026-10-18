# Efficient fine-tuning and data strategies for video models with tiny datasets on a single Tesla T4

Project context, as the coordinator gave it: 40 unique staged overhead cash-room videos (19 normal, 21 violation) with one timecode per video and 8 types having 1–5 examples each. Test videos come from a different camera angle. The current model is VideoJev-4B (Qwen3.5-4B backbone). Only its decision head (2×2560→512) was fine-tuned, on frozen hidden states with leave-one-video-out (LOVO). That cut false positives from 63% to 37%, but recall fell from 86% to 76%, and type accuracy is 43%. Hardware: T4, 16 GB, sm75, FP16 only, no FA2. Commercial-use open-source licenses only. Research date: 2026-10-17. Number of tool calls was capped (~16), so some points are marked as gaps.

## 1. LoRA/QLoRA for video-LLMs/VLMs on a T4 (memory, kernels, FP16, tools, few-shot evidence)

### Takeaway
QLoRA on a 2B–4B Qwen-family VLM is technically feasible on a T4: bitsandbytes NF4 supports sm75, Unsloth officially lists the T4, and Qwen video tokens can be budgeted precisely. But the published evidence that LoRA on a VLM improves anomaly-type classification with very small data is weak. One surveillance benchmark found LoRA helped general QA but had "limited impact on anomaly-specific QA". With 40 clips and a new test viewpoint, full LoRA of the backbone carries a high overfitting risk for modest expected gain. Treat it as a stretch option, not the main path.

### Cited Findings
- bitsandbytes needs compute capability 6.0+ for NF4/FP4 quantization and 8-bit optimizers, and 7.5+ (Turing, T4) for LLM.int8(). PyPI Linux x86-64 wheels for CUDA 11.8–13.2 all include sm75 targets — [bitsandbytes installation docs](https://huggingface.co/docs/bitsandbytes/main/en/installation)
- QLoRA's ingredients are NF4, double quantization and paged optimizers (to absorb memory spikes). The authors released CUDA kernels for 4-bit training — [HF blog: 4-bit transformers and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- Unsloth requires minimum CUDA capability 7.0 and names the T4 explicitly. Its listed minimum VRAM for QLoRA (4-bit) is 3.5 GB for 3B, 5 GB for 7B and 6 GB for 8B models. For LoRA (16-bit) it is 8 GB for 3B, 19 GB for 7B and 22 GB for 8B (text-only minimums; the page warns some models need more and suggests batch 1–3 on OOM) — [Unsloth requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements)
- Unsloth's earlier vision release said its vision notebooks fine-tune on a free Tesla T4 (examples: Llama 3.2 Vision 11B, Qwen2-VL 7B) — [Unsloth vision blog](https://unsloth.ai/blog/vision); notebook index lists Qwen2.5-VL and Qwen3-VL entries — [Unsloth notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks)
- Unsloth's Qwen3-VL page says fine-tuning support includes video and object detection. It offers free Colab notebooks (8B SFT and GRPO/GSPO) and claims 1.7x faster training with 60% less VRAM. It gives no absolute VRAM numbers and does not say T4 specifically — [Unsloth Qwen3-VL guide](https://unsloth.ai/docs/fr/modeles/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune.md)
- A third-party write-up fine-tuned a Qwen VLM on a single T4 (Kaggle) with Unsloth, using 4-bit QLoRA, gradient checkpointing and PEFT, treating memory as a hard limit (image-to-text task, not video) — [Towards AI](https://pub.towardsai.net/fine-tuning-qwen-for-image-to-text-extraction-on-a-single-t4-gpu-using-unsloth-and-trl-15918f6899c9)
- Unsloth publishes Qwen3-VL in 16-bit and "Dynamic 4-bit bitsandbytes" formats for fine-tuning — [HF unsloth/qwen3-vl collection](https://huggingface.co/collections/unsloth/qwen3-vl)
- Qwen2.5-VL vision config: patch_size=14, spatial_merge_size=2, temporal_patch_size=2. So one LLM visual token covers 28×28 px across 2 consecutive frames. The processor exposes min_pixels/max_pixels, docs show `attn_implementation="sdpa"` and examples use fps=1 — [HF Transformers Qwen2.5-VL docs](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl)
- Official Qwen VL fine-tuning recipe defaults: video_fps 2, video_min_frames 4, video_max_frames 8, video_max_pixels 1664×28×28, video_min_pixels 256×28×28. Default flags tune the LLM only (`tune_mm_llm True`, vision and MLP False). Optional LoRA defaults are r=8, alpha=16, with separate LRs for projector (1e-5) and vision tower (1e-6). The README says "tune_mm_vision should be False" when mixing image and video. FlashAttention-2 is optional (a config switch), and `--bf16` is described as "Ampere+ GPUs" — [QwenLM qwen-vl-finetune README](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune)
- SurveillanceVQA-589K: LoRA tuning of Qwen2.5-VL-3B raised the average score by about 5%, but "has limited impact on anomaly-specific QA" — [arXiv 2505.12589](https://arxiv.org/pdf/2505.12589)
- A community LoRA adapter for Qwen2.5-VL-7B (driving-scene anomaly classification) was trained on ~4,055 balanced images and reports 93.8% on a 1,020-image test. That is thousands of samples, not tens — [SAVANT anomaly classifier LoRA](https://friendli.ai/models/Brusnicki/SAVANT-anomaly-classifier-lora)
- FP16 overflow (inf/NaN in hidden states or attention accumulators) is a documented failure mode in other transformer models (T5, a llama.cpp case). Transformers ships a debug hook that detects the first module producing inf/NaN — [HF forum T5 fp16](https://discuss.huggingface.co/t/t5-fp16-issue-is-fixed/3139); [llama.cpp PR 17746](https://github.com/ggml-org/llama.cpp/pull/17746); [transformers debugging doc mirror](https://git.dghs.gov.bd/riaz.somc/transformers/src/commit/c63fcabfe92229eac983ce743f4e9fce863c84dd/docs/source/debugging.rst)

### Inferences
- **Token budget.** Qwen2.5/3-VL-style models produce tokens ≈ (H/28)·(W/28)·(T/2). Example: 448×448 px, 32 frames → 16·16·16 = 4,096 visual tokens; 336×336, 16 frames → 12·12·8 = 1,152 tokens. The 2B–4B models in this family have a hidden size of roughly 2–2.5k. Activations for ~1–4k tokens with gradient checkpointing should fit next to a ~2–3 GB 4-bit base model inside 16 GB. The real risk is the vision encoder's attention without FA2: SDPA on Turing falls back to memory-efficient or math kernels. This is an inference from the token formula plus general memory reasoning, not a measured number. Expect to measure it.
- **VRAM estimate** (inference, not measured): QLoRA on a 4B VLM with 16 frames at ~336 px, batch 1, gradient checkpointing, LoRA r=8–16 on LLM attention/MLP → roughly 8–13 GB. 32 frames at 448 px likely OOMs or needs the vision tower in no-grad mode, with features precomputed.
- **Speed estimate** (inference): T4 FP16 throughput is far below A100. 40 clips × LOVO folds is impractical for full LoRA. Use 5-fold CV instead: ~40 clips × 3 epochs × ~5–15 s/step ≈ 10–30 min per fold, so ~1.5–3 h for 5 folds.
- **FP16 on T4.** No BF16 means training uses FP16 autocast with a GradScaler, or fp32 master LoRA weights. Unsloth and PEFT handle this automatically. The Qwen recipe's `--bf16` must be replaced with `--fp16`. Keeping the vision tower frozen and computing its output in fp16 without grad lowers both NaN risk and memory.
- **Tools.** Unsloth (T4 officially supported; QLoRA vision) is the most T4-friendly path. LLaMA-Factory and ms-swift also support Qwen-VL video LoRA, but their T4 specifics were not verified in this session. TRL SFTTrainer + PEFT is the generic fallback.
- **Expected value.** With 21 positive clips across 8 types and a camera-angle shift, LoRA on the LLM would mostly learn this camera and these actors. The SurveillanceVQA result suggests small anomaly-specific gains even with much more data. If tried, restrict to a binary verdict plus coarse type, use strong augmentation, low rank (r=4–8), 1–3 epochs, and evaluate under cross-validation that holds out a whole scenario.
- **Effort:** 4–8 h engineering plus 2–4 h compute for a first credible result. High variance.

### Gaps
- I did not verify LLaMA-Factory/ms-swift issue threads on T4/FP16 NaNs for Qwen2.5/3-VL specifically (search returned nothing specific).
- I found no measured VRAM for Qwen-VL video LoRA on 16 GB. The numbers above are estimates.
- Qwen3.5 VLM architecture and license were not verified in this session. I also did not verify whether VideoJev-4B's license allows commercial use.
- I found no paper reporting LoRA on a video-LLM with only tens of clips.

## 2. Training only the projector, last layers, prompt tuning or visual prompt tuning

### Takeaway
The Qwen recipe itself exposes separate switches and learning rates for the projector, the vision tower and the LLM, and recommends freezing vision when mixing modalities. Evidence from CLIP few-shot work suggests that tuning only the last projection layer is a strong, data-efficient baseline. The team's current head-only training is already in this family. The next cheap step is to widen what is trained slightly (last 1–2 LLM blocks, or a LoRA on the last few layers), not to go to full LoRA.

### Cited Findings
- The Qwen VL fine-tune script exposes `tune_mm_vision`, `tune_mm_mlp` and `tune_mm_llm` independently, with recommended LRs of projector 1e-5 and vision tower 1e-6 — [qwen-vl-finetune README](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune)
- "Fine-tuning CLIP's last visual projector" (ProLIP, arXiv 2410.05270) proposes few-shot adaptation by training only CLIP's final visual projection matrix, positioned as a strong few-shot baseline — [arXiv 2410.05270](https://arxiv.org/html/2410.05270v1)
- Frozen Feature Augmentation (FroFA, CVPR 2024) applies augmentations in frozen-feature space for few-shot linear probes. It covers twenty augmentations and was evaluated on images, not video — [arXiv 2403.10519](https://arxiv.org/pdf/2403.10519)

### Inferences
- The FP↓/recall↓ pattern from head-only training points to a decision-threshold or calibration problem, not only capacity (see section 6). Retraining a larger part of the model will not fix threshold choice.
- Cheap upgrades in rough cost order:
  - (a) Train the head on hidden states pooled from several layers rather than the last only (~1 h, no extra VRAM; features cached once).
  - (b) Use FroFA-style feature-space augmentation (noise, dropout, mixup of cached features) for the head (~1 h).
  - (c) Apply LoRA to the last 2–4 LLM blocks only, with cached inputs to those blocks. This makes backprop cheap and fits easily in 16 GB (~3–5 h).
- Prompt or visual prompt tuning adds few parameters, but each step still needs a full forward pass through the 4B model. On a T4 it is not cheaper than LoRA on the last layers, and no evidence was found that it beats head tuning at n≈40.

### Gaps
- I found no study of projector-only or prompt tuning for video-LLMs at tens of clips.

## 3. Frozen video encoders + linear or attentive probe or a small temporal head

### Takeaway
Frozen self-supervised video encoders with a small attentive probe are the most data-efficient and T4-friendly alternative or complement. V-JEPA 2 (MIT-licensed code; ViT-L 300M) is the strongest documented candidate. Its authors report that feature-prediction models are more label-efficient and that the gap widens as labels decrease. Per-frame image encoders with temporal pooling are a simpler baseline. Licenses for VideoMAE V2 and InternVideo2 weights are unclear or gated and must be checked before use.

### Cited Findings
- V-JEPA 2 repo: "The majority of V-JEPA 2 is licensed under MIT" (a few files Apache-2.0). Models: ViT-L/16 300M, ViT-H 600M, ViT-g 1B; V-JEPA 2.1 adds ViT-B 80M and ViT-G 2B. Input is 256 px (384 px for g/384 and 2.1). Checkpoints are on HF (e.g. `facebook/vjepa2-vitg-fpc64-256`). The repo page does not state a separate weights license — [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- V-JEPA 2 frozen attentive-probe results: ViT-L SSv2 73.7%, Diving48 89.0%; ViT-g/384 SSv2 77.3%, Diving48 90.2% — [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- The V-JEPA 2 probe is 4 transformer blocks, with the last using cross-attention from a learnable query, followed by a linear classifier — [arXiv 2506.09985](https://arxiv.org/pdf/2506.09985)
- V-JEPA (v1) reports that feature-prediction pretraining is more label-efficient than pixel prediction, with the gap widening as labeled examples decrease. Non-linear attentive probes help V-JEPA, VideoMAE, DINOv2 and OpenCLIP compared with linear probes — [arXiv 2404.08471](https://arxiv.org/pdf/2404.08471)
- Masked-video (pixel-reconstruction) pretraining gives weaker linear-probe performance, which the authors attribute to low-level features — [arXiv 2409.06665](https://arxiv.org/pdf/2409.06665)
- Community conversions of V-JEPA 2 probes on HF are tagged MIT. They note licenses differ per artifact and that probe training data has its own terms — [LibreYOLO/LibreVJEPA2l256-cls-ssv2](https://huggingface.co/LibreYOLO/LibreVJEPA2l256-cls-ssv2)
- InternVideo2_CLIP_S on HF carries an Apache-2.0 tag (that checkpoint only) — [HF OpenGVLab/InternVideo2_CLIP_S](https://huggingface.co/OpenGVLab/InternVideo2_CLIP_S). The 6B variant's license was unclear per a repo issue — [InternVideo issues](https://redirect.github.com/OpenGVLab/InternVideo/issues)
- VideoMAE V2 model zoo downloads require a request form asking about organization and intended use — [VideoMAEv2 MODEL_ZOO](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md)
- A 2026 study trains linear probes on spatiotemporally pooled activations of frozen V-JEPA 2 and VideoMAE-v2, so per-layer features are usable as probe inputs — [arXiv 2602.07050](https://arxiv.org/pdf/2602.07050)

### Inferences
- **Recipe:** V-JEPA 2 ViT-L in FP16, 16-frame clips at 256 px, sliding window (stride 1–2 s) over each video, features cached once. Training a 1–2 block attentive probe or a small temporal head over window features takes minutes on a T4. Window-level scores give timecodes directly.
- **T4 cost** (estimate, not measured): ViT-L 300M on 16×256² clips → ~2k tokens per clip. Inference should be about tens of clips per second, so extracting features for 40 videos × ~60 windows takes ~5–20 min. VRAM < 6 GB in FP16.
- **Per-frame baseline:** SigLIP/SigLIP2 image features per frame + temporal mean/max/attention pooling. Cheap, and robust to a viewpoint change thanks to semantic pretraining. Its SigLIP2 license (believed Apache-2.0) was not verified in this session.
- **Ensembling:** Combine the V-JEPA probe score with the VideoJev head score through logistic stacking under cross-validation. This is the most likely cheap way to recover recall without adding false positives.
- **Effort:** 3–5 h total (feature extraction, probe, CV evaluation). VRAM 4–8 GB.

### Gaps
- I found no reported V-JEPA 2, SigLIP2 or VideoMAE V2 accuracy at tens of labeled clips.
- The V-JEPA 2 weights license was not confirmed separately from the code's MIT. Check the HF model card.
- Measured T4 throughput for V-JEPA 2 was not found.

## 4. Data augmentation and synthetic data for viewpoint robustness; how many clips are needed

### Takeaway
The best-documented route to unseen-viewpoint robustness is more viewpoint diversity in training: rendered synthetic humans or genuinely multi-view recordings. Geometric augmentation of a single view helps less. Since the organizer allows self-recorded data, recording staged clips from 2–3 extra camera angles that approximate the test angle is probably the highest-ROI data action. Generative video (Wan 2.1/2.2, Apache-2.0) is legally usable but slow on a T4 and unproven for this purpose.

### Cited Findings
- SURREACT (IJCV 2021) renders synthetic humans from real action sequences at new viewpoints. It reports state-of-the-art gains for unseen-viewpoint recognition on NTU RGB+D and UESTC, plus improvements in a one-shot Kinetics setting — [SURREACT project](https://www.di.ens.fr/willow/research/surreact)
- Synthesizing descriptors from other viewing angles via mocap (BMVC 2014) raised cross-view IXMAS accuracy from 62.1% to 67.4% averaged over view pairs — [BMVC 2014 abstract](https://www.bmva-archive.org.uk/bmvc/2014/files/abstract017.pdf)
- In event-camera multi-view recognition (DHP19, train 3 views / test 1 held-out), the single-view baseline drops sharply under cross-view testing, and multi-view training recovers part of that loss — [arXiv 2403.19316](https://arxiv.org/pdf/2403.19316)
- Contrastive view-invariant representation learning targets surveillance settings where collecting data for each new camera is impractical — [arXiv 2305.01733](https://ar5iv.labs.arxiv.org/html/2305.01733)
- Wan 2.1 and Wan 2.2 are reported as Apache-2.0, permitting commercial use. Wan2.1-T2V-1.3B needs ~8.19 GB VRAM, and a 5 s 480p video takes ~4 min on an RTX 4090 without optimization. These are third-party sources; check the official repo — [immers.cloud Wan2.1 1.3B](https://en.immers.cloud/ai/wan-ai/Wan2.1-T2V-1.3B-Diffusers/); [GMI Cloud](https://www.gmicloud.ai/en/blog/wan2-1-purchase-deployment-a-guide-to-commercial-video-generation). Claims about "Wan 3.0" open weights could not be verified — [atlascloud](https://www.atlascloud.ai/zh/blog/tips/wan-3.0-vram-requirements)

### Inferences
- **Self-recorded data (top priority).** A T4 is irrelevant here; the bottleneck is human time. A team of 2–3 people with phones mounted high (overhead, ~30–60° oblique, plus a different corner) could record ~30–60 clips in 2–4 h.
  - Target: ≥5–10 clips per violation type plus an equal number of hard normals (same props and motions without the violation), each filmed from ≥2 angles at once.
  - This is a rule-of-thumb inference. No source gave a "clips per class" threshold. Few-shot literature typically sees large gains moving from 1–5 to 10–20 examples per class.
- **Geometric augmentation.** hflip, rotation, zoom and perspective are already in use. Add strong random perspective/homography that simulates a camera-height and yaw change, random crops of the table area, color/blur/compression jitter, and temporal jitter (speed 0.8–1.25x, random start).
  - Effort: ~1–2 h. Do this in pixel space before the frozen encoders, generating 10–20 augmented views per clip for cached features.
- **Synthetic video generation.** On a T4, Wan 1.3B at ~4 min per 5 s clip on a 4090 would likely take ~10–20+ min per clip on a T4 (estimate). Fine-grained hand and cash actions are unlikely to be rendered faithfully.
  - Effort: high (4–8 h for a handful of clips). Low expected value. Not recommended within hackathon hours.
- **3D simulation (SURREACT-style).** Too costly for a hackathon (days).
- **Hard negatives** (normal clips that look like violations) are likely the most effective way to cut the 37% false positive rate.

### Gaps
- I found no study quantifying how many clips per class are needed for gains in staged surveillance or few-shot video at a new viewpoint.
- I did not verify Wan's official license file or T4 benchmarks, or other commercial-licensed generators (e.g. LTX-Video) in this session.

## 5. Weak timecode labels → MIL for weakly supervised temporal localization; pseudo-labeling

### Takeaway
The standard weakly supervised video anomaly detection (WSVAD) recipe is MIL. A video is a bag of segments, and top-K segment scores are trained against the video-level label, on frozen CLIP-like features. It maps directly onto this data (video verdict + one timecode). The single timecode can be used as a soft prior: positive bag = segments near the timecode, negatives = all segments of normal videos. VadCLIP shows frozen CLIP features + a light head reach strong results, but on datasets of hundreds to thousands of videos, not 40.

### Cited Findings
- WSVAD methods mainly use MIL with video-level labels. Videos are bags of segments, and top-scoring segments are selected via Top-K hard attention — [AVadCLIP arXiv 2504.04495](https://arxiv.org/html/2504.04495v2)
- VadCLIP (AAAI 2024) uses frozen CLIP without extra pretraining or fine-tuning and a dual branch: coarse binary classification plus fine-grained language-image alignment. It reports 84.51% AP on XD-Violence and 88.02% AUC on UCF-Crime, with code and features released — [VadCLIP AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28423); [arXiv 2308.11681](https://arxiv.org/html/2308.11681v3)
- CLIP-TSA adds temporal self-attention on CLIP ViT features for WSVAD — [UMD CLIP-TSA](https://cs.umd.edu/content/clip-tsa-clip-assisted-temporal-self-attention-weakly-supervised-video-anomaly-detection)
- AVadCLIP keeps CLIP frozen and uses lightweight parametric adaptation — [arXiv 2504.04495](https://arxiv.org/pdf/2504.04495)
- AnomalyCLIP (CLIP for weakly supervised anomaly recognition, i.e. type classification) has a tutorial — [LearnOpenCV AnomalyCLIP](https://learnopencv.com/anomalyclip-video-anomaly-recognition/)

### Inferences
- **Recipe:**
  - Split each video into 1–2 s segments and cache frozen features (V-JEPA 2 / SigLIP / VideoJev hidden states).
  - Train a small temporal head (1-layer transformer or TCN, ~0.5–2M params) with a MIL top-K loss.
  - For violation videos, add a soft supervision term: a Gaussian around the labeled timecode (σ ≈ 3–5 s) on segment scores.
  - All segments of normal videos are negatives.
  - The type head is applied to the top-K segments' pooled features.
- With timecodes, the 21 violation videos yield ~21 × (a few positive segments) and the normal set yields hundreds of negative segments. This rebalances learning and should help localization.
- **Pseudo-labeling:** Iterate. Take the top-scoring segments in violation videos as positives, add confident low-score segments as negatives, then retrain. Use 1–2 rounds only, because the tiny data amplifies errors. Unlabeled test-angle videos, if available, can be used for self-training or domain adaptation of normalization statistics.
- **Effort:** 3–6 h on cached features. Training takes minutes; VRAM < 4 GB.

### Gaps
- I found no MIL or WSVAD results at ~40 training videos. Published benchmarks use hundreds to thousands (UCF-Crime, XD-Violence).
- I did not fetch papers on few-shot or "point-supervised" (single timestamp) temporal action localization, which are directly relevant to one-timecode labels.

## 6. Calibration under recall priority: thresholds, class-balanced loss, conformal prediction

### Takeaway
The trade-off the team observed (FP 63→37%, recall 86→76%) is largely an operating-point issue. Choose the decision threshold on out-of-fold scores to hit a target recall, rather than using 0.5. Conformal risk control gives a formal false-negative-rate bound for a chosen threshold. With ~21 positives, the guarantee is marginal and noisy, and coverage conditional on one small calibration set can fall well below target. Report it as an empirical estimate with its uncertainty.

### Cited Findings
- Split conformal coverage ≥1−α holds for any calibration size n, but small n is less stable — [Angelopoulos & Bates, Gentle Introduction, arXiv 2107.07511](https://arxiv.org/pdf/2107.07511)
- Conformal calibration picks a threshold τ on a calibration set to reach target coverage, with a finite-sample guarantee for a fresh exchangeable test point — [RAPS, arXiv 2009.14193](https://arxiv.org/pdf/2009.14193)
- Critique: coverage conditional on a fixed small calibration set can fall well below the target with high probability, and the classic argument presumes frequent recalibration — [arXiv 2512.14727](https://arxiv.org/html/2512.14727v1)
- A 2025 paper proposes probabilistic (PAC-style) guarantees for a single conformal predictor that stay informative for small calibration sets — [arXiv 2512.04566](https://arxiv.org/html/2512.04566v2)
- Conformal Risk Control (ICLR 2024) bounds the expected value of any monotone loss, with false negative rate as a worked example. It is tight up to O(1/n), and extensions cover distribution shift. Code: github.com/aangelopoulos/conformal-risk — [arXiv 2208.02814](https://arxiv.org/abs/2208.02814v4); [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)

### Inferences
- **Practical recipe (~1 h):**
  - Collect out-of-fold LOVO scores for all 40 videos.
  - Plot the ROC/PR curve and pick τ = the highest threshold achieving ≥90–95% recall on the 21 positives.
  - With n=21, one missed positive is about 4.8 recall points, so targets are coarse.
  - Apply conformal risk control (FNR) on the same out-of-fold scores to state the bound. Use a Clopper-Pearson interval to show the uncertainty: with 21 positives, a 95% CI on recall at an observed 20/21 is roughly 76–100% (computed by inference, not from a source).
- **Class-balanced or recall-weighted loss:** Weighted BCE (pos_weight ≈ 2–3) or focal loss in the head. Weighting can be tuned per CV fold, but threshold moving achieves most of the same effect more transparently.
- **Domain shift:** Conformal guarantees assume exchangeability, which a different test camera angle violates. Hold out a self-recorded different-angle set as the calibration set (ties to section 4). Alternatively, use the distribution-shift extensions in CRC.
- **Two-stage policy for type accuracy (43%):** Report the top-2 types, or a conformal prediction set, for the type in violation cases. This trades specificity for set coverage, which may suit a review UI where a human confirms.

### Gaps
- I found no source on calibration specifically for video anomaly detection with tens of samples.
- I did not research isotonic or Platt calibration at small n (general knowledge says Platt scaling is safer than isotonic at n≈40).
