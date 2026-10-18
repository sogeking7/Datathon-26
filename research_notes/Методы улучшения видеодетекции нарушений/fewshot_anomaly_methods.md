# Few-shot, reference-based and anomaly-detection methods for procedural (SOP) violation detection in cash-room video

Scope: methods that exploit (a) very few labeled violations, (b) many normal videos, (c) the procedural/rule nature of the task, under limits of local open-source commercial-license models, a single T4 16 GB, and 5 min per video. Current system baseline (from the assignment, not a source): VideoJev-4B gate questions + fine-tuned head, 70% verdict, 76% recall, 37% FP on normals, 43% type accuracy (LOVO).

Research note: about 18 tool calls; several full-text tables (EgoPER, ZeProM, Holmes-VAD details) could not be retrieved (403 / size limits). Those items are listed under Gaps instead of being filled from memory.

## 1. VLM/LLM-based video anomaly detection (training-free / weakly supervised): how they work, accuracy, T4 feasibility, transfer to SOP anomalies

### Takeaway
The leading VLM-based VAD methods sit at about 80–89% frame-level AUC on UCF-Crime. In ranked order: Holmes-VAU 88.96, VERA 86.55, VADor 85.90, Holmes-VAD 84.61, AnyAnomaly about 80.7, LAVAD 80.28. The most transferable ideas for a SOP setting are AnomalyRuler (rules induced from normal references), VERA (guiding questions learned from video-level labels on a frozen 8B VLM) and AnyAnomaly (user-defined text anomaly, zero-shot, fast with Qwen2.5-VL-3B). The heavy fine-tuned models (Holmes-VAD/VAU) are trained on crime datasets and are poor fits.

### Cited Findings
- **LAVAD (CVPR 2024)** is training-free. A VLM captions each frame, then an LLM aggregates captions over time and scores anomalies. It beats unsupervised and one-class methods on UCF-Crime and XD-Violence with no training — [arXiv 2404.01014](https://arxiv.org/html/2404.01014v1); [CVPR poster](https://cvpr.thecvf.com/virtual/2024/poster/29246)
- LAVAD scores **80.28 AUC on UCF-Crime** and **85.36 AUC on XD-Violence**. Zero-shot CLIP gets 53.16 / 38.21 and LLaVA-1.5 gets 72.84 / 79.62 — [VERA paper table, arXiv 2412.01095](https://arxiv.org/html/2412.01095v3); the same LAVAD figure appears in [arXiv 2511.17094](https://arxiv.org/pdf/2511.17094)
- **VERA (2024/2025)** keeps the VLM frozen. An optimizer VLM and a learner VLM interact verbally to learn a small set of *guiding questions* (default m = 5; 1 question is worse, more than 5 gives no significant gain). Training uses only coarse video-level labels, with 8 uniformly sampled frames per video, batch 2, up to 10 epochs. The default backbone is InternVL2-8B; InternVL2-40B and Qwen2-VL-7B were also tested — [arXiv 2412.01095v3](https://arxiv.org/html/2412.01095v3)
- VERA scores **86.55 AUC on UCF-Crime** and **88.26 on XD-Violence**, versus VADor 85.90 and Holmes-VAD 84.61 on UCF-Crime. The strongest non-explainable baseline listed is CLIP-TSA at 87.58 — [arXiv 2412.01095v3](https://arxiv.org/html/2412.01095v3)
- Example learned VERA questions for UCF-Crime are about people, vehicles or objects behaving or positioned unusually, visible damage, and unusual sounds. The initial seed question was "Is there any suspicious person or object that looks unusual in this scene?" — [arXiv 2412.01095v3](https://arxiv.org/html/2412.01095v3)
- **Holmes-VAU (CVPR 2025)** pairs an anomaly scorer with a density-aware sampler that picks frames by anomaly score, so the MLLM focuses on anomaly-rich regions. It is trained on HIVAU-70k (more than 70k clip-, event- and video-level annotations). It reports **88.96 AUC on UCF-Crime** and **87.68 AP on XD-Violence** — [arXiv 2412.06171](https://arxiv.org/html/2412.06171v1); [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Holmes-VAU_Towards_Long-term_Video_Anomaly_Understanding_at_Any_Granularity_CVPR_2025_paper.html)
- **AnyAnomaly (WACV 2026)** does "customizable VAD" (C-VAD). The user writes a text description of the abnormal event, and an LVLM answers context-aware VQA without fine-tuning. CLIP ViT-B/32 key-frame selection picks 4 key frames per segment. Position context comes from WinCLIP-based attention, and temporal context from a grid image of frames. Code: github.com/SkiddieAhn/Paper-AnyAnomaly — [arXiv 2503.04504](https://arxiv.org/abs/2503.04504v4); [HTML](https://arxiv.org/html/2503.04504v4)
- AnyAnomaly results:
  - On its C-VAD benchmarks it raises AUC from 78.01 to 85.72 (C-ShT) and from 79.43 to 90.27 (C-Ave).
  - With MiniCPM-V it scores Ave 87.3, ShT 79.7, UB 74.5, UCF 80.7 AUC with context, and 81.4 / 77.2 / 73.1 / 77.8 without context.
  - Speed with **Qwen2.5-VL-3B + vLLM is 34.78 FPS** (AUC maintained) and 24.08 FPS with Qwen2.5-VL-7B, on a single RTX 3090.

  [arXiv 2503.04504v4](https://arxiv.org/html/2503.04504v4)
- A 2026 benchmark of compact VLMs (Qwen2.5-VL, InternVL3, Gemma3 cited) for clip-level surveillance anomaly detection under weak supervision exists, using LoRA / LLaMA-Factory. Its numbers could not be extracted — [arXiv 2603.13306](https://arxiv.org/pdf/2603.13306)

### Inferences
- UCF-Crime and XD-Violence anomalies are visually salient (fighting, explosions, accidents). Our violations are context/rule anomalies: an open safe, cash with nobody present, a CIT bag crossing the desk. AUC on crime benchmarks is weak evidence for transfer. The text-defined designs (AnyAnomaly, AnomalyRuler) are structurally closer to SOP checking than models fine-tuned on crime (Holmes-VAD/VAU).
- **T4 feasibility (inference, not sourced):**
  - AnyAnomaly's Qwen2.5-VL-3B path at about 35 FPS on a 3090 suggests at least several FPS on a T4 in fp16. A 145 s video sampled at 1–2 fps (145–290 frames) should fit well within 5 min.
  - VERA at inference is just a frozen 8B VLM with 5 extra questions in the prompt. On a T4 that needs 4-bit or 8-bit quantization of InternVL2-8B or Qwen2-VL-7B, or swapping to a 2–4B VLM.
  - VERA's *training loop* (optimizer VLM + learner VLM) is offline and could run slowly on the T4 or on any machine, because the output is only text questions.
- **VERA adapts most directly to our data.** It needs only video-level labels, and we have 40 videos with verdicts. The output is a set of verbal questions, which our gate-question system already consumes. Estimated effort: 12–20 h to re-implement the question-optimization loop over our existing VideoJev or another local VLM.
- **AnyAnomaly-style per-type text prompts:** one prompt per violation type, plus a generic "any unsafe cash handling" prompt, with key-frame selection and grid temporal context. Estimated effort: 8–12 h on a Qwen2.5-VL-3B/7B base. It also gives a route to unseen types, because new types are just new text.
- Licenses: Qwen2.5-VL-3B and Qwen2.5-VL-7B are commonly reported as different licenses (7B Apache-2.0; 3B under a Qwen research license). This was **not verified in this research** and must be checked on the model cards before use.

### Gaps
- Holmes-VAD and VADor architecture details and backbones were not retrieved. Only their AUCs (84.61, 85.90) via the VERA table are confirmed.
- No source reports any of these methods on rule-based/SOP anomalies in banking or retail cash handling. Transfer is inferred.
- VERA inference speed and GPU are not reported in the paper text read.

## 2. Rule-induction approaches (AnomalyRuler-style): few normal/abnormal examples → textual rules → verification

### Takeaway
AnomalyRuler induces normal and anomaly rules from only about 10 normal reference frames, then deduces with a 7B LLM. It reaches 85.2 AUC on ShanghaiTech and 71.9 on UBnormal, which is state of the art among the methods compared there. It also raises LLM-baseline recall from 6–15% to 64%. Rule induction suits a procedural task well, and our abnormal examples can additionally be used to sharpen the rules, which the original does not do.

### Cited Findings
- **Induction:** few-shot normal reference frames (default n = 10) are captioned by CogVLM-17B. GPT-4 induces normal rules and anomaly rules, and LLM-based **rule aggregation** votes across multiple induced rule sets — [arXiv 2407.10299v2](https://arxiv.org/html/2407.10299v2)
- **Deduction:** Mistral-7B-Instruct-v0.2 applies the rules to test-frame captions. **Exponential Majority Smoothing** (EMA plus majority vote, padding 5, α = 0.33) smooths perception, and a "robust reasoning" pass rechecks frames against the rules — [arXiv 2407.10299v2](https://arxiv.org/html/2407.10299v2)
- Only normal frames are used. Anomaly rules are derived by contrast with normal rules, not from anomaly examples — [arXiv 2407.10299v2](https://arxiv.org/html/2407.10299v2)
- AUC for AnomalyRuler (full / base):

  | Dataset | Full | Base |
  |---|---|---|
  | Ped2 | 97.9 | 96.5 |
  | Avenue | 89.7 | 82.2 |
  | ShanghaiTech | 85.2 | 84.6 |
  | UBnormal | 71.9 | 69.8 |

  On ShT and UB it is the best in the paper's table. On Ped2 and Avenue some fully trained methods score higher (HF2 99.2 on Ped2) — [arXiv 2407.10299v2](https://arxiv.org/html/2407.10299v2)
- On ShanghaiTech, against direct LLM prompting baselines:
  - Accuracy: **81.8** versus 52.1–58.4.
  - **Recall: 64.3** versus 6.2–15.2.
  - Reasoning correctness: 83% versus 57–74% for GPT-4 and GPT-4V baselines.

  [arXiv 2407.10299v2](https://arxiv.org/html/2407.10299v2)
- Cross-domain test with rules induced on ShT and applied elsewhere: Ped2 97.4, Ave 81.6, UB 65.4. It trails the domain-generalization method zxVAD by 0.6% on average — [arXiv 2407.10299v2](https://arxiv.org/html/2407.10299v2)
- The authors state that results depend on a capable LLM backbone. They used GPT-4 for induction only, because it was too costly for test-time deduction — [arXiv 2407.10299v2](https://arxiv.org/html/2407.10299v2)
- AnyAnomaly notes that AnomalyRuler needs rule generation plus a few normal samples, whereas AnyAnomaly is zero-shot — [arXiv 2503.04504v4](https://arxiv.org/html/2503.04504v4)

### Inferences
- **Why this maps well to our task.** Our 19 normal videos are exactly the "normal references" AnomalyRuler needs. The 8 violation types are effectively rules: "safe door must be closed when unattended", "cash must not be visible when a visitor is present", "counting must use the machine", and so on. An induced rule list can also be extended by hand with bank SOP text, which helps with unseen types.
- **Suggested adaptation (not from a source):**
  1. Caption each frame or clip with the local VLM, using structured attributes: number of people, their zone (behind desk or visitor side), safe state, door state, cash visible or in machine, bag location.
  2. Induce rules offline. This can be done with any strong LLM offline, since only the text rules ship. Whether an offline closed LLM is allowed under the hackathon's "local open-source" rule needs checking. If not, use a local 7–14B instruct LLM.
  3. At test time, check structured captions against the rules with a small local LLM, or with plain Python predicates.
  4. Apply EMA and majority smoothing over time to suppress single-frame false positives.
- **Expected effect on our failure modes.** Temporal smoothing and the "normal rules first" framing directly target the 37% FP rate. Per-rule verdicts give the violation type explicitly, which targets the 43% type accuracy. Rules are the type labels.
- **Camera-angle shift.** Rules over semantic attributes (door open, cash visible) are viewpoint-invariant in a way that embedding classifiers are not. This is an argument for rules over linear probes, given that the test set uses a different camera.
- Estimated effort: 16–24 h for a structured captioner + rule induction + rule checker + smoothing + LOVO evaluation, or 8–10 h if rules are hand-written from the 8 types and SOP text and only the checker and smoothing are built.

### Gaps
- AnomalyRuler runtime per frame and memory footprint were not found. CogVLM-17B as captioner would not fit a T4 unquantized; a smaller captioner would need to be substituted (inference).
- No source tested rule induction with *both* normal and abnormal examples. Using our 21 violation videos for contrastive rule refinement is an untested extension.

## 3. Procedure / mistake detection in instructional and industrial videos (EgoPER, Assembly101, CaptainCook4D, HoloAssist, IndustReal)

### Takeaway
The mistake-detection literature trains mainly on error-free videos: one-class, prototype-based, or next-step anticipation. That matches our "many normals, few violations" regime. A June 2026 paper (ZeProM) reports that a single pre-trained VLM, zero-shot, approaches or beats fully supervised mistake detectors on EgoPER and CaptainCook4D. However, a 2026 thesis documents VLM failures in temporal reasoning.

### Cited Findings
- **EgoPER (CVPR 2024)** introduces an egocentric procedural error dataset and a framework, EgoPED, trained **only on error-free videos**. It uses an action segmentation model plus contrastive step-prototype learning, with hand-object relation features from active object detection and a GCN. At inference, errors are flagged by low similarity to learned step prototypes. It outperforms state-of-the-art VAD methods for error detection. Code: github.com/robert80203/EgoPER_official — [CVPR 2024 open access](https://openaccess.thecvf.com/content/CVPR2024/html/Lee_Error_Detection_in_Egocentric_Procedural_Task_Videos_CVPR_2024_paper); [project page](https://www.khoury.northeastern.edu/home/eelhami/egoper.htm)
- **PREGO (CVPR 2024)** is the first *online one-class* mistake detector for procedural egocentric video, evaluated on Assembly101-O and Epic-tent-O. It compares recognized steps with steps anticipated by an LLM — [TI-PREGO, arXiv 2411.02570](https://arxiv.org/html/2411.02570v1)
- **TI-PREGO** extends PREGO and evaluates several LLMs as step anticipators with **few-shot prompting**. Auto-CoT "significantly" improves the anticipation branch. It adopts Balanced Precision so that mislabeling a correct step is penalized on the same scale as the reward for catching a mistake — [arXiv 2411.02570](https://arxiv.org/html/2411.02570v2)
- **ZeProM (arXiv 2606.21579, June 2026)** defines *zero-shot procedural mistake detection*. A single pre-trained VLM performs both mistake detection and temporal action segmentation. On EgoPER (5 tasks averaged) it improves **EDA by +4.4 points** and a mAP-type metric by +2.0 over the strongest *supervised* methods, and it is evaluated on CaptainCook4D as well. The authors say there are no previous zero-shot attempts at this task — [arXiv 2606.21579](https://arxiv.org/pdf/2606.21579); [abstract mirror](https://deeplearn.org/arxiv/777747/the-unreasonable-effectiveness-of-vlms-for-zero-shot-procedural-mistake-detection)
- A 2026 University of Trento thesis evaluates post-completion mistake detection on CaptainCook4D, EgoPER, EgoOops and Assembly101. It reports limitations of end-to-end VLMs, "particularly in temporal reasoning", and proposes a structure-first baseline — [iris.unitn.it](https://iris.unitn.it/handle/11572/484810)
- **CaptainCook4D** (NeurIPS 2024 D&B) frames zero-shot error recognition as video QA for VLMs — [arXiv 2312.14556](https://arxiv.org/html/2312.14556v4); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/f4a04396c2ed1342a5d8d05e94cb6101-Paper-Datasets_and_Benchmarks_Track.pdf)
- **EgoOops** evaluates mistake action detection with reference to procedural texts. It tested open-source MLLMs zero-shot on its dataset — [arXiv 2410.05343](https://arxiv.org/pdf/2410.05343)

### Inferences
- **Most transferable pattern: "structure first, VLM second".** Segment the video into states or events (person enters, safe opened, cash out, visitor arrives, bag passed), then check the ordering and co-occurrence constraints. This is the EgoPED / thesis direction. Our violations are mostly *state co-occurrence* violations ("cash visible AND visitor present"; "safe open AND nobody behind desk"), not ordering violations. That makes them simpler than assembly mistakes and well suited to a state-timeline + rule checker.
- **EgoPED-style prototype matching** (normal-only training, flag low similarity) maps to our 19 normals. But with one fixed overhead camera in training and a *different* camera at test, embedding prototypes may produce false positives from viewpoint shift alone (inference; see section 5).
- **ZeProM** supports making a zero-shot VLM with explicit procedure text the backbone, rather than fine-tuning a head on 40 videos. The head is the likely source of overfitting in a LOVO split of 40 videos (inference).
- Estimated effort: about 20–30 h for the state-timeline + co-occurrence rule checker (shares components with section 2). Replicating EgoPED is not recommended within a hackathon: it needs action segmentation training and hand-object detection, about 40 h or more.

### Gaps
- EgoPER EDA/AUC tables for EgoPED versus VAD baselines could not be retrieved (PDF 403 / connection reset).
- ZeProM's VLM backbone, size and prompt design were not retrievable (PDF exceeded the fetch size limit), so T4 feasibility is unknown.
- HoloAssist and IndustReal mistake-detection results were not researched within the call budget.

## 4. Few-shot / in-context VLM classification, retrieval of nearest normal references, contrastive/siamese, embedding kNN

### Takeaway
No retrieved source evaluates *in-context example clips* for video anomaly detection with a VLM. The closest evidence is AnomalyRuler, which uses normal references converted to text, and older meta-learning work on few-shot scene adaptation. Reference comparison is best done in text or attribute space, or as a "same scene, empty-room baseline" difference, rather than by putting raw multi-video context into a small VLM on a T4.

### Cited Findings
- **Few-shot scene-adaptive anomaly detection** (ECCV 2020) detects anomalies in a previously unseen scene from only a few frames of that scene, using meta-learning. This is the canonical precedent for "new camera at test time, few normal frames available" — [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2983_ECCV_2020_paper.php)
- A 2025 paper uses a Swin Transformer with MAML in a 5-shot training / 10-shot testing setup over 11 anomaly categories (not VLM-based) — [JACEM 2025](https://www.nepjol.info/index.php/JACEM/article/view/84526)
- AnomalyRuler's normal-reference approach turns references into text rules instead of feeding images in context. Direct LLM prompting without rules had only 6.2–15.2% recall on ShanghaiTech — [arXiv 2407.10299v2](https://arxiv.org/html/2407.10299v2)
- AnyAnomaly shows that adding temporal context (a grid of frames) and position context to a zero-shot LVLM raised UCF-Crime AUC from 77.8 to 80.7 and Avenue from 81.4 to 87.3 — [arXiv 2503.04504v4](https://arxiv.org/html/2503.04504v4)

### Inferences
- **Reference-based comparison for the test camera.** If the test set uses a new angle, several *normal* test-angle frames may not be available. Within each test video, however, the first seconds are often a "baseline" state. A per-video self-reference approach can be cheap and robust: compare each moment's attribute vector or VLM description against the video's own initial or empty-room state. This is inference, not tested in a source.
- **In-context example images on a 3–8B VLM.** Each example frame costs hundreds to thousands of visual tokens, and small VLMs degrade with many images. On a T4 this limits prompts to about 2–4 reference frames. Prefer text descriptions of references (AnomalyRuler-style) or one positive/negative exemplar frame per type for *type disambiguation* only, after detection.
- **Retrieval for type assignment (kNN over the 21 violation clips around their timecodes).** This is cheap (about 4–6 h) but has the camera-shift risk. Prefer retrieval over text captions or structured attributes rather than raw visual embeddings.
- **Siamese/contrastive fine-tuning on 40 videos** is very likely to overfit and has no published support at this scale for this domain. It is not recommended.

### Gaps
- No 2024–2026 paper was found that benchmarks multi-image in-context few-shot prompting of open VLMs for video anomaly or violation classification. This is a clear evidence gap.
- The compact-VLM weak-supervision benchmark (arXiv 2603.13306) may contain relevant few-shot or LoRA numbers but could not be parsed.

## 5. Linear probes / kNN on frozen video encoders (V-JEPA 2, VideoMAE V2, InternVideo2, SigLIP2): data efficiency, licenses, T4 feasibility

### Takeaway
V-JEPA 2 is the strongest frozen-feature encoder documented here: 77.3% SSv2 attentive probe for ViT-g, versus 69.7% for the previous best, InternVideo2-1B. Its code is MIT, with a ViT-L 300M variant that fits a T4 easily. VideoMAE V2 is **CC-BY-NC-4.0 (non-commercial) and must be excluded**. InternVideo2 Hugging Face repos are tagged MIT, but this should be verified. With about 21 positives and a test camera change, probes on frozen visual embeddings are risky as the primary detector. They fit better as an auxiliary or type-assignment signal.

### Cited Findings
- V-JEPA 2: "The majority of V-JEPA 2 is licensed under MIT", with three dataset-utility files under Apache 2.0. The README does not state a separate weights license — [official README](https://cdn.jsdelivr.net/gh/facebookresearch/vjepa2@main/README.md)
- Hugging Face checkpoint tags differ: the ViT-L checkpoint is listed as MIT and the ViT-g as Apache-2.0. A third-party conversion warns that "licences in this family differ per artifact" — [mixpeek ViT-L](https://mixpeek.com/model/facebook/vjepa2-vitl-fpc64-256); [mixpeek ViT-g](https://www.mixpeek.com/model/facebook/vjepa2-vitg-fpc64-256); [LibreYOLO conversion](https://huggingface.co/LibreYOLO/LibreVJEPA2g384-embed). Both licenses are commercial-friendly either way.
- V-JEPA 2 sizes are ViT-L/16 300M, ViT-H/16 600M and ViT-g/16 1B, at 256 or 384 px. Attentive probes on frozen features:

  | Model | SSv2 | Diving48 | EK100 R@5 |
  |---|---|---|---|
  | ViT-L/256 | 73.7 | 89.0 | 32.7 |
  | ViT-g/384 | 77.3 | 90.2 | 39.7 |

  The previous SSv2 probe best was 69.7, from InternVideo2-1B — [official README](https://cdn.jsdelivr.net/gh/facebookresearch/vjepa2@main/README.md)
- VideoMAE V2 model cards are tagged **cc-by-nc-4.0**. The VideoMAE codebase states that most of it is CC-BY-NC 4.0 — [HF VideoMAEv2-Base](https://huggingface.co/OpenGVLab/VideoMAEv2-Base/discussions); [ModelScope VideoMAEv2-giant card](https://www.modelscope.cn/models/OpenGVLab/VideoMAEv2-giant/resolve/master/README.md); [VideoMAE GitHub](https://github.com/mcg-nju/videomae)
- InternVideo2 HF repos are tagged **License: mit**: InternVideo2-Chat-8B, InternVideo2_Chat_8B_InternLM2_5 and InternVideo2-Stage2_6B. An aggregator notes that the README does not state the license explicitly — [HF Chat-8B](https://huggingface.co/OpenGVLab/InternVideo2-Chat-8B/discussions/8); [HF Stage2_6B](https://www.huggingface.co/OpenGVLab/InternVideo2-Stage2_6B/tree/main); [SourcePulse](https://www.sourcepulse.org/projects/1845322)
- In VAD, embedding-only zero-shot CLIP is very weak: 53.16 AUC on UCF-Crime and 38.21 on XD-Violence — [VERA table](https://arxiv.org/html/2412.01095v3). Weakly supervised CLIP-feature methods such as CLIP-TSA reach 87.58 with full UCF-Crime training labels — [same](https://arxiv.org/html/2412.01095v3)

### Inferences
- **T4 feasibility (inference):** V-JEPA 2 ViT-L (300M) at 256 px on 16-frame clips runs comfortably in fp16 on a T4. Feature extraction for a 145 s video in sliding windows (for example 1 window per 2 s, about 70 windows) should take well under a minute. ViT-g (1B) is also feasible in fp16 (about 2 GB weights), but slower.
- **Data efficiency for our case.** SSv2-style probe results use tens of thousands of labeled clips. Nothing in the sources shows reliable probes from about 20 positive clips. More promising is a **one-class / kNN-to-normal** score: the distance of each window to the nearest windows from the 19 normal videos. This uses the normals directly, but it will flag *any* novelty, including camera angle, lighting and different actors. On a new camera it would likely *raise* false positives unless normal references from the test angle exist (inference; this is the main risk).
- **Recommended roles for embeddings:**
  1. A temporal *localization* prior, meaning where something changes, used to choose frames for the VLM, as in Holmes-VAU's density-aware sampling.
  2. Type disambiguation among candidate types once a violation is detected.

  Do not use them as the final verdict.
- SigLIP2 frame embeddings (commonly reported as Apache-2.0, **not verified here**) could play the same role at lower cost and support text queries such as "an open safe", which gives some zero-shot capability. CLIP-style zero-shot alone is weak for VAD, as shown above.
- Estimated effort: 6–10 h for V-JEPA 2 ViT-L feature extraction + kNN-to-normal scoring + LOVO evaluation; another 4 h for a logistic probe on the type label.

### Gaps
- An official weights license file for V-JEPA 2 was not confirmed (README covers code; HF tags conflict MIT vs Apache-2.0, both permissive).
- The InternVideo2 LICENSE file text was not read; MIT is from HF metadata only.
- SigLIP2 license and few-shot linear-probe data-efficiency numbers were not researched within the budget.
- No published few-shot (fewer than 50 clips) probe results for V-JEPA 2 on anomaly-like tasks were found.

## 6. Which approaches address our biggest failure modes (37% FP on normals, 43% type accuracy) — synthesis

### Takeaway
On the evidence above, the best-supported path is: structured VLM perception of a few cash-room state attributes, AnomalyRuler/AnyAnomaly-style per-rule checks (one rule per violation type plus a generic catch-all), and temporal smoothing. Use the 19 normal videos to calibrate thresholds, and optionally VERA-style question optimization on the 40 labels. This attacks FP (smoothing, normal-calibrated thresholds, rules instead of an overfit head) and type accuracy (rules are types), and it degrades more gracefully under the test-camera change than embedding probes.

### Cited Findings
- Temporal smoothing plus rule-based deduction raised LLM-based VAD accuracy on ShanghaiTech from 52–58% to 81.8% and recall from 6–15% to 64.3% — [arXiv 2407.10299v2](https://arxiv.org/html/2407.10299v2)
- Learned guiding questions on a frozen VLM improve over the 72.84 AUC of LLaVA-1.5 and the 80.28 of LAVAD to 86.55 on UCF-Crime, using only video-level labels — [arXiv 2412.01095v3](https://arxiv.org/html/2412.01095v3)
- Text-defined anomaly with key-frame and temporal context lifts zero-shot AUC by 3–6 points and is fast with a 3B VLM (34.78 FPS on an RTX 3090) — [arXiv 2503.04504v4](https://arxiv.org/html/2503.04504v4)
- A zero-shot single-VLM procedural mistake detector can match or beat supervised ones on EgoPER (+4.4 EDA) — [arXiv 2606.21579](https://arxiv.org/pdf/2606.21579)
- Mistake detection evaluation favors Balanced Precision, so that false alarms on correct steps are penalized comparably to missed mistakes — [arXiv 2411.02570](https://arxiv.org/html/2411.02570v2)

### Inferences
Prioritized plan with estimated effort for a 2–4 person team:

| # | Method | Addresses | Effort (h) | Risk |
|---|---|---|---|---|
| 1 | Per-video temporal smoothing (EMA plus majority over windows, with a minimum event duration of k consecutive positive windows) on existing gate scores | FP | 2–4 | Low |
| 2 | Threshold calibration on the 19 normals (pick the threshold for at most X% FP on normal LOVO folds, report recall) | FP / recall trade-off | 2–3 | Low |
| 3 | Structured state captioning per window (people count and zones, safe/door open, cash visible and where, machine in use, visitor present, bag location) with the local VLM, constrained MC/JSON output | Type, FP, camera shift | 8–12 | Medium (VLM perception errors) |
| 4 | Rule checker per violation type over the state timeline (Python predicates or a small LLM), plus a generic "SOP breach" rule set induced from normals (AnomalyRuler) | Type accuracy, unseen types | 8–16 | Medium |
| 5 | VERA-style guiding-question optimization using the 40 video labels (offline) | Verdict, recall | 12–20 | Medium (overfit to 40 videos; validate LOVO) |
| 6 | V-JEPA 2 ViT-L kNN-to-normal as a localization prior and a type-retrieval tie-breaker | Timecode, evidence frame | 6–10 | High under camera shift |

- **Dropping the fine-tuned decision head**, or regularizing it heavily, is likely to reduce FP. A learned head on 40 videos in LOVO tends to memorize scene and actor specifics (inference).
- **The "missed violations unacceptable" requirement.** Keep a high-recall generic rule ("cash or valuables visible/accessible without an authorized employee actively attending", "non-employee in staff zone", "safe/door open without activity"). Use smoothing to keep FP down, not a stricter threshold on specific types.
- **Evidence frame and timecode.** Take them from the window where the rule first fires after smoothing. AnyAnomaly's key-frame selection (CLIP-based representative frame) is a cheap way to pick the evidence frame.

### Gaps
- No published method has been evaluated on bank cash-room SOP violations. All transfer claims are inferred.
- Effort estimates are the researcher's judgment, not sourced.
- The licenses for Qwen2.5-VL-3B versus 7B, MiniCPM-V, InternVL2, Mistral-7B (Apache-2.0 commonly reported) and CogVLM were not verified here and must be checked against the commercial-use requirement.
