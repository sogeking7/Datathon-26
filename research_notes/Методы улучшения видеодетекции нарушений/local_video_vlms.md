# Local open-source video VLMs that could replace or complement VideoJev-4B on a single T4 (state as of Oct 2026)

Scope: one Tesla T4 (16 GB, sm75, no BF16 tensor cores, no FlashAttention 2), commercial-use license, 5 min per video end to end, short overhead bank-cash-room clips (10–145 s, 464x848 portrait) with small objects.
Research ran to about 19 tool calls. Many numbers come from model cards and could not be cross-checked. Anything without a URL is marked as inference.

## 1. Which models fit in 16 GB on a T4, and what runtimes/throughput are reported?

### Takeaway
Every 2B–9B candidate fits on a T4 in FP16 (4B is about 8–10 GB of weights plus activations; 8–9B needs 8-bit or 4-bit quantization to leave room for video tokens). The real limits on a T4 are the attention backend (no FA2, so use SDPA/eager in HF or TRITON_ATTN in vLLM), BF16-only checkpoints that overflow in FP16 (Gemma-family), and Gated-DeltaNet kernels in Qwen3.5. I found no reliable T4 throughput numbers for video VLMs. The team should measure them.

### Cited Findings
- T4 is compute capability 7.5. That is above vLLM's minimum of 7.0, but it has no native BF16 or FP8, so `--dtype half` is the practical choice. — [markaicode vLLM T4 guide](https://markaicode.com/benchmarks/cuda-production-benchmark-latency/) (secondary source)
- A user reports Qwen2.5-VL on T4s in vLLM with `--dtype half`. T4 users have forced `VLLM_ATTENTION_BACKEND=XFORMERS` (older vLLM). — [vLLM forum](https://discuss.vllm.ai/t/speeding-up-vllm-inference-for-qwen2-5-vl/615/13)
- One source says some newer vLLM model backends (Qwen3-VL, late 2025) dropped Turing support. A later vLLM release note reports Marlin support for Turing (sm75) and a user running Qwen3-VL-32B-AWQ on 2080Ti (Turing). This comes from search snippets of LinkedIn/vLLM release posts and is unverified. — [vLLM v0.24.0 release summary (LinkedIn)](https://pl.linkedin.com/in/dariusz-kreft-010a2687); [Reddit snapshot](https://reddit.sentinel-team.org/posts/1qim0e9/snapshots/2026-01-21T17%3A11%3A35.0614Z)
- vLLM picks the first backend that validates. On SM8.x/9.x the order is FLASH_ATTN > FLASHINFER > TRITON_ATTN > FLEX_ATTENTION. The TRITON_ATTN source declares support for all compute capabilities, fp16/bf16/fp32, head size ≥32, which makes it the likely backend on Turing. — [vLLM attention backends doc](https://docs.vllm.ai/en/stable/design/attention_backends/)
- A Gemma 3 report says vLLM rejects float16 for Gemma 3 ("does not support float16 — numerical instability"). — cited in search summary of [vLLM forum/HF card](https://huggingface.co/OmarJemli/gopro-AI-gemma3-4b) (weak source)
- The Qwen3-VL-4B-Instruct card recommends `flash_attention_2` for video, and its example uses `torch.bfloat16`. Weights ship in BF16. — [HF Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- Qwen3.5-4B is a hybrid: 8 blocks of (3× Gated DeltaNet + 1× Gated Attention), BF16 weights (~5B params counting vision). SGLang/vLLM need main/nightly builds for Qwen3.5. — [HF Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- MiniCPM-V 4.5 (8B) offers int4, GGUF and AWQ variants and runs in llama.cpp/Ollama/vLLM/SGLang. The 28 GB GPU figure in its Video-MME table is for the full model. — [HF MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5)
- MiniCPM-V 4.5 paper: on VideoMME it uses "46.7% GPU memory cost and 8.7% inference time of Qwen2.5-VL 7B". — [arXiv 2509.18154](https://arxiv.org/pdf/2509.18154)
- SmolVLM2-2.2B needs about 5.2 GB GPU RAM for video inference. — [HF SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- Gemma 4 E2B/E4B are edge models with a ~150M vision encoder. One third-party source estimates E4B at 12–16 GB. — [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4); [datanorth](https://datanorth.ai/news/google-releases-gemma-4-open-models)

### Inferences (not verified with a source in this session)
- FlashAttention 2 needs sm80+, so on a T4 use `attn_implementation="sdpa"` (PyTorch mem-efficient kernel works on sm75) or `"eager"`. SDPA memory grows quickly with video tokens: at 2 fps × 20 s = 40 frames and ~64–256 tokens per frame, a window is 2.5k–10k tokens, which is fine. Avoid long single-pass contexts (>16k tokens) on a T4.
- The BF16→FP16 cast is the main T4 risk. Qwen2/2.5/3-VL and InternVL generally run in FP16 (the team already runs VideoJev/Qwen3.5-4B in FP16). Gemma 3/4 activations are known to overflow in FP16. Expect NaNs or garbage unless you use FP32 for some layers, which is slow.
- Qwen3.5 Gated DeltaNet uses Triton kernels (flash-linear-attention). They run on sm75 because VideoJev works, but they are untuned for Turing. Qwen3-VL (pure attention) may be faster per token on a T4. Benchmark both.
- Quantization on sm75: bitsandbytes 8-bit/4-bit NF4 works on Turing (slow but memory-saving). AutoAWQ GEMM kernels support sm75. GPTQ via ExLlama v2 works on Turing. Classic Marlin kernels were sm80+, and Turing support in vLLM is reported only recently (above). FP8 does not work. llama.cpp CUDA builds work on sm75 and support Qwen2.5-VL/Qwen3-VL/Gemma/MiniCPM-V via `mtmd`. Video must be fed as frames there.
- Rough budget: 145 s at 2 fps is 290 frames. With 20 s windows that is about 8 windows × 8 gate questions = 64 forward passes. With a scoring-only approach (no generation), a 4B FP16 model on a T4 should do one 40-frame prefill in roughly 2–6 s (estimate, not measured). That fits in 5 min if the vision encoding is cached per window and reused across all 8 questions (KV-cache prefix reuse).

### Gaps
- I found no published T4 tokens/s or frames/s numbers for Qwen3-VL, Qwen3.5, InternVL3.5, MiniCPM-V 4.5 or Gemma 4 on video. Measure locally.
- I could not confirm from official vLLM docs whether the current vLLM V1 fully supports sm75 for Qwen3-VL/Qwen3.5 (only forum or release-note snippets).

## 2. Which licenses allow commercial use (per checkpoint)?

### Takeaway
Safe Apache-2.0 choices: Qwen3-VL (2B/4B/8B), Qwen3.5 (0.8B/2B/4B/9B), Qwen2.5-VL-7B, InternVL3.5 (2B/4B/8B), MiniCPM-V 4.5, Gemma 4 (all sizes, Apache-2.0 for the first time) and SmolVLM2. Watch out for Qwen2.5-VL-3B (Qwen Research License, non-commercial), Gemma 3 (Gemma Terms of Use), VideoLLaMA3 (conflicting statements) and TimeLens/TimeLens2 (academic-use notice).

### Cited Findings
- Qwen3-VL-4B-Instruct: license `apache-2.0`. — [HF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct). A secondary source says the whole Qwen3-VL family (2B, 4B, 8B…) is Apache-2.0. — [codersera](https://codersera.com/blog/qwen3-vl-4b-vs-qwen3-vl-8b-benchmarks-vram-guide/)
- Qwen3.5-4B: Apache-2.0. — [HF Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The small Qwen3.5 models (0.8B/2B/4B/9B) were released around 2 March 2026 under Apache-2.0, per secondary coverage. — [VentureBeat](https://venturebeat.com/technology/alibabas-small-open-source-qwen3-5-9b-beats-openais-gpt-oss-120b-and-can-run)
- **Qwen2.5-VL-3B-Instruct: "Qwen RESEARCH LICENSE AGREEMENT", s.2(a) "FOR NON-COMMERCIAL PURPOSES ONLY", s.2(b) commercial use requires requesting a license. Not usable as-is.** — [HF LICENSE file](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE)
- InternVL3.5-4B / 8B: model card metadata `apache-2.0` (taken from a PR revision of the README). — [HF InternVL3_5-4B](https://huggingface.co/OpenGVLab/InternVL3_5-4B), [README rev](https://huggingface.co/OpenGVLab/InternVL3_5-4B/blob/refs%2Fpr%2F4/README.md), [InternVL3_5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B)
- MiniCPM-V 4.5: "model weights and code are open-sourced under the Apache-2.0 license". The questionnaire is optional. — [HF MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5). Note: earlier MiniCPM-V releases (2.6, 4.0) used a custom "MiniCPM Model License" that required registration for commercial use (from memory, not verified here).
- Gemma 4: Apache-2.0 (official model card), released 2 April 2026. Previous Gemma versions used the restrictive Gemma Terms. — [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4); [The Decoder](https://the-decoder.com/googles-gemma-4-is-now-available-with-apache-2-0-licensing-for-the-first-time/)
- SmolVLM2-2.2B: Apache-2.0. — [HF](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- VideoLLaMA3: sources conflict. A fork is tagged Apache-2.0, but a project summary says the service is "for non-commercial use ONLY, subject to the licenses of Qwen, OpenAI, and Gemini". — [SourcePulse](https://www.sourcepulse.org/projects/1844827); [HF fork](https://huggingface.co/cbipok/VideoLLaMA3-2B-fork/blob/main/README.md); official repo [GitHub](https://github.com/DAMO-NLP-SG/VideoLLaMA3) (not checked)
- TimeLens2-4B fine-tune card: license "unresolved", with an academic-use notice and no commercial license. TimeLens-7B/8B are based on Qwen2.5-VL-7B/Qwen3-VL-8B. — [mixpeek card](https://mixpeek.com/model/Alibaba-VELLDEPTH/CTVG-4B); [TimeLens arXiv 2512.14698](https://arxiv.org/html/2512.14698v2); [TimeLens2](https://huggingface.co/papers/2607.17423.md)

### Inferences
- VideoJev-4B inherits Qwen3.5-4B (Apache-2.0), so the base license is fine. Check VideoJev's own card and fine-tune data terms separately.
- From memory (verify before use): Phi-4-multimodal-instruct is MIT, image and audio only (no native video), 5.6B. LLaVA-OneVision-1.5 is Apache-2.0 (Qwen3 LLM). LLaVA-Video-7B (Qwen2-7B) is Apache-2.0, but its training data includes GPT-generated captions with ToS caveats. Apollo (Meta/Stanford) was pulled from HF after release and is effectively unusable. VideoChat-Flash builds on Qwen2/2.5. The 2B variant is on Qwen2.5-1.5B (Apache), but a 3B-based variant would inherit the Qwen Research License. InternVL3 (not 3.5) 1B/2B/8B/9B/14B/38B use Qwen2.5 or InternLM3 backbones. The 2B is Qwen2.5-1.5B (Apache), but check each size.

### Gaps
- I did not fetch the official licenses for Phi-4-multimodal, LLaVA-OneVision-1.5, VideoChat-Flash or Apollo. The VideoLLaMA3 official LICENSE was not checked.

## 3. Reported performance on relevant fine-grained/temporal benchmarks, and evidence on surveillance/SOP tasks

### Takeaway
At 4B, Qwen3.5-4B (the VideoJev base) and InternVL3.5-4B report about MVBench 71 and are the strongest commercially licensed small video models. Qwen3-VL adds explicit text-timestamp alignment for temporal grounding. Zero-shot VLMs on UCF-Crime reach 75–90% AUC depending on prompting, and they have a known "normality bias" (under-calling anomalies). Zero-shot procedural-mistake detection is mixed: one 2026 preprint shows near-supervised results on EgoPER, while EMNLP 2025 reports near-chance in the wild. None of these benchmarks covers bank-SOP compliance.

### Cited Findings (video benchmarks)
| Model | License | MVBench | Video-MME (w/o / w sub) | Other | Source |
|---|---|---|---|---|---|
| Qwen3.5-4B | Apache-2.0 | 71.2 | 76.9 / 83.5 | MLVU 82.8, LVBench 66.4, VideoMMMU 74.1, MMVU 64.9 | [HF](https://huggingface.co/Qwen/Qwen3.5-4B) |
| InternVL3.5-4B | Apache-2.0 | 71.2 | 65.4 / 68.6 | MLVU 70.4, LongVideoBench 60.8 | [arXiv 2508.18265](https://arxiv.org/pdf/2508.18265) |
| InternVL3.5-8B | Apache-2.0 | 72.1 | 66.0 / 68.6 | MLVU 70.2, LongVideoBench 62.1 | [arXiv 2508.18265](https://arxiv.org/pdf/2508.18265) |
| MiniCPM-V 4.5 (8B) | Apache-2.0 | n/a | 73.5 avg | claims high-FPS (up to 10 fps), MotionBench/FavorBench | [HF](https://huggingface.co/openbmb/MiniCPM-V-4_5) |
| Qwen2.5-VL-3B | **non-commercial** | 67.0 | 61.5 / 67.6 | Charades-STA mIoU 38.8 | [HF](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| VideoLLaMA3-2B | unclear | 65.5 | — | Charades-STA 55.5 | [arXiv 2501.13106](https://arxiv.org/pdf/2501.13106) |
| SmolVLM2-2.2B | Apache-2.0 | — | 52.1 | — | [HF](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) |
| Gemma 4 E2B/E4B | Apache-2.0 | not reported | not reported | video as frames, max 60 s at 1 fps | [model card](https://ai.google.dev/gemma/docs/core/model_card_4) |

- Caution: Qwen3.5-4B's Video-MME 76.9 is ~11 points above InternVL3.5-4B and probably reflects a thinking-mode, many-frame setup. The card does not say which. — [HF Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- Qwen3-VL claims "Text–Timestamp Alignment" for "timestamp-grounded event localization". The tech report says it replaces T-RoPE-style time encoding with explicit textual timestamp tokens. — [HF card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct); [review of the tech report](https://www.themoonlight.io/tw/review/qwen3-vl-technical-report); [arXiv 2511.21631](https://arxiv.org/pdf/2511.21631v1)
- Video-MME-v2 (Apr 2026) lists Qwen3-VL-8B-Think slightly above 4B-Think (512/64-frame settings). — [arXiv 2604.05015](https://arxiv.org/pdf/2604.05015)
- A 2026 paper diagnoses "accuracy without grounding" in video LLM benchmarks (models score well without really using visual evidence). This is a warning that MVBench/Video-MME may not predict small-object performance. — [arXiv 2607.13305](https://arxiv.org/pdf/2607.13305)

### Cited Findings (anomaly / surveillance / procedure)
- ASK-HINT: the Qwen2.5-VL-7B zero-shot baseline gets 74.50% AUC on UCF-Crime, rising to 89.83% with fine-grained, action-centric question prompting. A full-prompt baseline gets 67.17. — [arXiv 2510.02155](https://arxiv.org/pdf/2510.02155)
- HiProbe-VAD (tuning-free hidden-state probing): Qwen2.5-VL at 80.60% AUC on UCF-Crime, InternVL2.5 at 81.35%, best 86.72% with InternVL2.5. Probing intermediate hidden states with a light classifier beats text generation. — [arXiv 2507.17394](https://arxiv.org/pdf/2507.17394)
- Unified zero-shot reasoning framework: UCF AUC 84.28% with VideoLLaMA3-7B, ~83.23% with a Qwen2.5-VL-7B variant (metric column unconfirmed). — [arXiv 2511.00962](https://arxiv.org/pdf/2511.00962)
- CoAT (Qwen2.5-VL-7B baseline) notes that LVLMs "exhibit a strong normality bias, often failing to recognize crime". — [arXiv 2512.20417](https://arxiv.org/html/2512.20417v1)
- AnyAnomaly (WACV 2026): zero-shot customizable VAD with user-defined anomaly text. This is close to "unseen violation types defined by text". — [arXiv 2503.04504](https://arxiv.org/pdf/2503.04504v4)
- ZeProM (19 Jun 2026): a single pre-trained VLM, zero-shot, for mistake detection plus temporal action segmentation. On EgoPER it is +4.4 EDA and +2.0 F1@.5 over the strongest supervised methods. The model name is not in the abstract. — [arXiv 2606.21579](https://arxiv.org/abs/2606.21579)
- Storks et al. (EMNLP 2025): current approaches do not get above chance in zero-shot procedural mistake detection in the wild. They propose visual rationales plus NLI-based coherence metrics. — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.706)
- PREGO / TI-PREGO: online one-class mistake detection that predicts the next step with an LLM and flags mismatches. They note that no technique reliably detects open-set procedural mistakes online. — [arXiv 2404.01933](https://www.arxiv.org/pdf/2404.01933); [arXiv 2411.02570](https://arxiv.org/pdf/2411.02570v2)

### Inferences
- The normality bias matches VideoJev's pattern only partly. Recall is 76%, but false positives on normals are 37%, so the issue there is calibration rather than bias alone. Score-threshold calibration per question (on held-out normals) and ASK-HINT-style decomposition of each violation into concrete, observable sub-questions ("is the safe door ajar?", "is any banknote visible on the desk while a second person is present?") are the most evidence-backed levers.
- HiProbe-style probing is a cheap complement: extract mid-layer hidden states of Qwen3.5-4B/Qwen3-VL-4B per window and train a logistic/linear probe on 40 videos. This uses the same forward pass as VideoJev scoring, with no extra model.
- For timecodes, Qwen3-VL-4B/8B (Apache-2.0, textual timestamps) is the most plausible commercially licensed "temporal localizer" to run on a candidate window after VideoJev flags it.

### Gaps
- No official Charades-STA/ActivityNet-Grounding numbers were retrieved for Qwen3-VL-2B/4B/8B or Qwen3.5-4B.
- No TempCompass or VideoMME-short-split numbers were retrieved per model.
- No paper found on VLMs for bank/cash-handling SOP or retail/cashier compliance specifically.

## 4. Practical tricks: small objects, frame sampling, yes/no logit scoring, structured output

### Takeaway
Small objects need more pixels per object, not more frames. Use static high-res crops of fixed ROIs (safe, door, desk) and dynamic-resolution token budgets. Frame rate matters less: 1–2 fps is enough for these state-type violations. Scoring yes/no or option logits on a shared, cached visual prefix is faster and easier to calibrate than generation.

### Cited Findings
- Qwen3.5 example defaults to `fps=2`, `do_sample_frames=True`. The video preprocessor is "conservatively configured" (token cap). You can raise `longest_edge` in `video_preprocessor_config.json`. Custom fps/num_frames is configurable in vLLM via `--media-io-kwargs`. — [HF Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- MiniCPM-V 4.5's 3D-Resampler packs up to 6 consecutive frames into 64 tokens (96× compression). Example uses `fps=5`, `MAX_NUM_FRAMES=180`, `MAX_NUM_PACKING=3`. This lets it see more frames at the same LLM cost. — [HF MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5)
- Gemma 4 visual token budget per image is selectable: 70/140/280/560/1120. Lower is for video, higher for OCR/detail. — [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- Fine-grained, action-centric sub-questions ("ASK-HINT") raise Qwen2.5-VL-7B UCF-Crime AUC from 74.5 to 89.8. — [arXiv 2510.02155](https://arxiv.org/pdf/2510.02155)
- Hidden-state probing beats textual output for anomaly scoring with frozen MLLMs. — [arXiv 2507.17394](https://arxiv.org/pdf/2507.17394)

### Inferences (engineering judgment, not sourced here)
- Small objects: at 464x848, Qwen-family dynamic resolution with 28-px patches (2x2 merged) gives about 16x30 = ~500 tokens per frame at native size. A banknote that is tens of pixels is only 1–2 merged tokens. Options:
  - (a) Send fixed ROI crops (safe, door, desk/cash tray) upscaled 2–3× as extra images alongside a downsampled full frame.
  - (b) Use a cheap detector (YOLO-family, Apache/AGPL caveats, or OWLv2/Grounding-DINO-style open-vocab, check licenses) to propose crops.
  - (c) Run the VLM on ROI crops at 1 fps rather than the whole frame at 2 fps.
  - A different camera angle in the test set breaks fixed ROIs. An open-vocab detector or a one-time VLM "locate the safe/door" grounding call per video (Qwen3-VL has 2D grounding) makes ROIs adaptive.
- Frame sampling: most of the violation types are persistent states (safe open, door open, items on desk, cash unattended) that 0.5–1 fps detects. Event types (bag handover, entry of visitor) need about 2 fps around motion. Use frame differencing or motion energy to pick dense windows and sparse sampling elsewhere. This frees budget for higher resolution.
- Scoring:
  - Compare logits of " Yes"/" No" (or option letters A–H) at the first answer position. Normalize with softmax over just those tokens, and calibrate a per-question threshold on normal videos (to cut the 37% FP).
  - Contrastive prompts (score "violation present" minus "normal procedure") reduce prior bias.
  - Cache the visual prefix KV once per window and append the 8 questions, which gives about 8× fewer vision encodes.
- Structured output: use generation with JSON schema (vLLM guided decoding / xgrammar, or llama.cpp grammars) only for the final explanation/type on flagged windows, not for screening.
- Ensemble: VideoJev-4B (screening) + Qwen3-VL-4B or InternVL3.5-4B (second opinion on flagged windows, plus timestamp) fits sequentially in 16 GB if one is unloaded or both use 8-bit (about 4.5 GB each).

### Gaps
- No published ablation found on ROI-crop vs full-frame for small-object video QA with these specific models.

## 5. Video temporal grounding models that run on a T4 with commercial licenses

### Takeaway
Most dedicated temporal-grounding fine-tunes (TimeLens/TimeLens2, and from memory TRACE, Time-R1, UniTime, VideoChat-R1) are academic releases with unclear or non-commercial terms, or inherit Qwen2.5-VL-3B's research license. The safest commercially licensed option is base Qwen3-VL-2B/4B/8B (Apache-2.0) with its built-in timestamp alignment, or your own fine-tune of it or Qwen3.5.

### Cited Findings
- TimeLens-7B / TimeLens-8B are fine-tunes of Qwen2.5-VL-7B / Qwen3-VL-8B. No official weight license statement was found (the arXiv CC-BY-4.0 covers the paper only). — [arXiv 2512.14698](https://arxiv.org/html/2512.14698v2)
- TimeLens2 (Jul 2026, MCG-NJU, "generalist video temporal grounding"). A derivative card reports an academic-use notice and no commercial license. — [HF paper page](https://huggingface.co/papers/2607.17423.md); [mixpeek](https://mixpeek.com/model/Alibaba-VELLDEPTH/CTVG-4B)
- TaRO (Jun 2026) reports zero-shot Charades-STA for Qwen3-VL-8B-Instruct/Think as baselines, with Qwen2.5-VL-3B as a 3B baseline. This shows the base Qwen3-VL already does grounding zero-shot. — [arXiv 2606.09248](https://arxiv.org/pdf/2606.09248)
- Qwen3-VL "Text–Timestamp Alignment… timestamp-grounded event localization", Apache-2.0. — [HF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- VideoLLaMA3-2B reports Charades-STA 55.5, but its license is unclear. — [arXiv 2501.13106](https://arxiv.org/pdf/2501.13106)

### Inferences
- For timecodes, a grounding model is probably overkill. Per-window scores from VideoJev at 2 fps, with argmax over a sliding-window score curve, give a timecode at ±(window stride) resolution. A shorter stride (5 s) inside flagged regions refines it cheaply. Use the evidence frame as the frame of max ROI-crop score.
- Any 7–8B grounding fine-tune runs on a T4 only in 8-bit/4-bit, and a 145 s video at 2 fps is ~290 frames. That is too many tokens in one pass on a T4, so chunking is needed anyway.

### Gaps
- I did not verify licenses or T4 feasibility for TRACE, Time-R1, UniTime or VideoChat-R1/Flash in this session. Treat them as "check before use", likely research-only or inheriting their base license.
