# Industry and competition practice for detecting SOP-compliance violations in surveillance video (banking, cash handling, retail checkout, kitchen, manufacturing)

Research date: 2026-10-17. Scope is the ForteBank cashier-room hackathon: overhead camera, 8+ violation types, local commercially licensed models, one T4, at most 5 minutes per video, no face recognition, no audio, and test footage from a new camera angle.

Convention used below: **Cited Findings** are facts or claims taken from a source. Vendor numbers are marked "(vendor claim, unverified)". **Inferences** are my own reasoning and are not sourced facts.

---

## Q1. How do commercial video-analytics vendors build their pipelines? What signals do they use, and how do they handle new camera angles?

### Takeaway
Vendors share no deep architecture details. What they do publish has the same shape everywhere: existing CCTV (RTSP/ONVIF/NVR) feeds edge or on-prem inference, which detects people and objects; a rules layer applies zones, dwell time and counting; and alerts go out with a short clip or snapshot. The strongest systems fuse video with a second signal: POS transactions for Everseen, safe-alarm or access-control events for bank vaults. Setup for a new camera is mostly manual zone drawing plus perspective calibration. I found no vendor that publicly documents automatic zone setup.

### Cited Findings
**Bank branch and cash-handling vendors**
- Agrex.ai (AIVIS) lists these banking use cases: vault access compliance, cash-counting-area monitoring, "uncounted cash on counters", cash handling without dual-control verification, CIT (cash-in-transit) handoffs, cashier or staff not present, customer left unattended, mobile-phone usage, camera obstruction, and people avoiding camera coverage. It refers to "behavioral models trained on banking-specific scenarios" and to named zones (vault area, ATM zones). It does not name the model types (pose, detector, etc.). — [Agrex.ai Banking](https://www.agrexai.com/banking-video-analytics)
- Agrex deployment: works with existing IP or analog cameras over RTSP/ONVIF or via NVR/DVR; processes at the edge, in the cloud or on-prem; sends only metadata and event clips to the cloud; integrates with Milestone and Genetec VMS and Honeywell access control; alerts via Slack, Teams, WhatsApp and SMS; includes camera/NVR health monitoring "for compliance". — [Agrex.ai Banking](https://www.agrexai.com/banking-video-analytics)
- Agrex numbers (vendor claim, unverified): 99% detection of unauthorized access, 40–55% fewer cash discrepancies, "zero false alerts on normal customer traffic" in an anonymous case study, alerts within 30 seconds. — [Agrex.ai Banking](https://www.agrexai.com/banking-video-analytics); [Agrex case study](https://www.agrexai.com/banking-video-analytics-for-atm-and-branch/)
- Eocortex describes using a neural-network people-counting module in a bank vault to enforce dual control: at least two people (customer and employee) must be present. — [Eocortex](https://eocortex.com/news/how-eocortex-analytics-improve-banks-performance)
- Convergint suggests tying video analytics to the safe's alarm system so footage is captured each time the safe opens, and reviewing workroom snapshots for dual-control compliance. — [Convergint](https://old.convergint.com/?p=9607)
- Verkada's banking brochure: door alerts (forced entry, tailgating, wire tampering) linked to video. — [Verkada banking brochure](https://docs.verkada.com/docs/banking-finance-brochure.pdf)
- Bankers Online: analytics can run on cameras, NVRs, servers or third-party software, either in real time (alerts) or as post-processing of recorded footage. — [BankersOnline](https://bankersonline.com/qa/smart-surveillance-systems)

**Retail checkout loss prevention (Everseen)**
- Everseen's patent application, "System and method for detecting scan and non-scan events in a self check out process", describes a camera over the scan area, proximity sensors that trigger capture, and a neural network that gives each frame a probability of scan vs. non-scan. These probabilities are **correlated with POS transaction data**, and an event is decided from that correlation. — [Justia patent 20210216785](https://patents.justia.com/patent/20210216785) (summary from search snippet; the full text returned HTTP 403)
- Everseen runs on existing security cameras, sends staff an alert with an image of the item within about 3 seconds, does not use facial recognition, and in newer deployments (Morrisons) asks the shopper on screen to rescan. Reportedly it tells store merchandise apart from personal items (phones, handbags) in the scan area. — [Retail TouchPoints](https://www.retailtouchpoints.com/features/solution-spotlight/everseen-uses-ai-to-detect-non-scanning-items-at-checkout); [Grocery Gazette](https://www.grocerygazette.co.uk/?p=118219)
- Everseen claims about 90% detection of non-scans (vendor claim, unverified). — [diginomica](https://diginomica.com/how-everseen-applies-ai-and-deep-learning-to-point-of-sale-with-a-checkout-free-future-racing-towards-us/)

**Kitchen / food-safety SOP (Wobot.ai)**
- Wobot advertises glove, hairnet and handwash detection (a 20-second handwash check in a "Personal Hygiene Checklist") with real-time alerts. It connects to existing CCTV, detects "people, objects and activities", and sends supervisors a short clip. No architecture is published. — [Wobot blog](https://wobot.ai/blog/improving-food-safety-in-restaurants-with-video-analytics); [Wobot webinar recap](https://wobot.ai/blog/all-about-our-webinar-computer-vision-brains-behind-workplace-safety-and-employee-productivity); [IndianWeb2](https://www.indianweb2.com/2020/08/video-analytics-platform-wobot-raises.html)

**Manufacturing SOP (Drishti)**
- Drishti analyzes worker actions on assembly lines and warns when steps are missed. Reported customers include Ford, Flex, Nissan and DENSO. — [IoT ONE](https://v1.iotone.com/supplier/drishti/v2726)

**VLM-based agent platforms (NVIDIA)**
- The NVIDIA AI Blueprint for Video Search and Summarization (VSS) combines a customizable CV pipeline (object detection and tracking, real-time embeddings) with VLMs and LLMs (VILA, Llama Nemotron) and RAG. It lists "SOP validation" as a use case and includes agent workflows for "real-time alert verification". — [NVIDIA blog](https://blogs.nvidia.com/?p=80495); [NVIDIA dev blog](https://developer.nvidia.com/blog/?p=98690); [NVIDIA Perspectives FAQ](https://perspectives.nvidia.com/nvidiametropolis/task/faq/vision-language-models-video-understanding-1/) (the FAQ pages look auto-generated; treat with caution)

**Camera setup and perspective**
- Axis Loitering Guard uses user-drawn areas, dwell time, "False Alarm Filtering", and **perspective configuration**, which relates an object's image size to its distance from the camera. — [Axis Loitering Guard overview](https://www.networkwebcams.co.uk/axis-loitering-guard-application-licence/)
- ACIC StillnessDetection raises an alarm when an object of a chosen class stays still for, **or disappears after**, a set time. Matching "object stays" or "object vanishes" against a dwell timer is a standard primitive. — [SoftwareOne listing](https://platform.softwareone.com/product/acic-stillnessdetection/PCP-2074-9346)

### Inferences
- The recurring pattern is a **detector plus tracker, then zones and a state machine, then alerts with clip and snapshot**. This maps directly onto our task. Each violation becomes a predicate over tracked objects and zones held for N seconds: "safe door open AND no cashier within its zone for over T seconds", "cash visible on desk AND a non-cashier person present", and so on.
- Vendors report their strongest detections when a **second modality confirms the video**: POS data at Everseen, safe-alarm or door-contact events at Convergint and Verkada. We have no second modality (no audio, no metadata). The closest substitute is **multi-signal confirmation inside the video**: object state, person or hand proximity, and zone occupancy must all agree.
- Agrex's use-case list (uncounted cash on counter, dual control, CIT handoff, staff absent, phone usage) overlaps heavily with ForteBank's list. That suggests these violation types are standard in the industry, and a demo can describe them in the same terms.
- No vendor reviewed documents automatic adaptation to a new viewpoint. In practice an installer draws zones per camera. In a hackathon, auto-proposing zones on a new angle with open-vocabulary detection plus SAM 2, with optional one-click correction, would set the product apart.

### Gaps
- No vendor disclosed model architectures (pose vs. detection vs. action models), training-data sizes or false-positive rates measured independently.
- Everseen's full patent text could not be fetched (HTTP 403), so details of the per-region network are not confirmed.
- I found no public technical material from bank-specific vendors on detecting "counting without a machine" or "cash exposed when a visitor enters".

---

## Q2. Which academic datasets and benchmarks are close to our task, and which approaches win?

### Takeaway
No public dataset covers bank cashier rooms. The closest analogues are AI City Challenge Track 4 (retail checkout, 2022–2023), Track 3 (naturalistic driving action localization: multi-camera, untrimmed, small staged dataset), PoseLift (pose-only shoplifting anomaly detection), and assembly-SOP datasets. Winners rely on ensembles of fine-tuned clip classifiers plus **heavy temporal post-processing** of boundaries, and on detector ensembles with WBF/TTA for rule-violation detection.

### Cited Findings
- **AI City 2024 Track 3** (SynDD2: 16 distracted-driving actions, 3 synchronized cameras in a stationary car, staged recordings). Ranking: TeleAI 1st (0.8282) with an Augmented Self-Mask Attention transformer for bidirectional context, a model ensemble, and **weighted boundary fusion**; SKKU-AutoLab 2nd (0.7798) with fine-tuned large models, ensembled clip-level classification and **multi-step boundary post-processing**; MCPRL 3rd (0.6080) with **multi-view fusion plus adaptive thresholding** to separate similar actions from background, followed by "post-connection" and candidate merging. One team used graph-based change-point detection to propose segments and a **Video-LLM to recognize them**. — [8th AI City Challenge summary, arXiv 2404.09432](https://arxiv.org/html/2404.09432v1); [Zhang et al. CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/AICity/html/Zhang_Augmented_Self-Mask_Attention_Transformer_for_Naturalistic_Driving_Action_Recognition_CVPRW_2024_paper.html)
- Nguyen et al. (CVPRW 2024): fine-tuned four action-recognition models with **K-fold cross-validation**, then built a multi-view ensemble that picks the **most informative camera view per action class**. — [Nguyen et al. CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/AICity/html/Nguyen_Multi-View_Spatial-Temporal_Learning_for_Understanding_Unusual_Behaviors_in_Untrimmed_Naturalistic_CVPRW_2024_paper.html)
- A later paper used a self-supervised recognizer plus **conditional post-processing** for temporal boundaries and placed 6th on the public leaderboard (test A2). — [arXiv 2411.12525](https://arxiv.org/html/2411.12525v1)
- **AI City 2024 Track 5** (helmet rule violations): the winners used Co-DETR plus a minority-class recall optimizer, **WBF** ensembling and calibrated confidence. 2nd place used Co-DETR+DETA, WBF+NMS, **TTA** and multi-scale random-crop augmentation. 3rd place (VNPT) used separate detectors for vehicle/person and head, plus an **association module** that pairs heads with riders and confidence correction for class imbalance. — [arXiv 2404.09432](https://arxiv.org/html/2404.09432v1)
- **AI City Track 4, retail checkout (2022–2023)**: a region-based YOLOv5+DeepSORT approach placed 4th in 2022 (F1 0.44). A 2023 paper reports F1 0.9792 on test A. Retail Counter (2023) used **video inpainting to remove hands**, YOLOv8 detection and DeepSORT tracking. — [7th AI City Challenge](https://ar5iv.labs.arxiv.org/html/2304.07500); [Region-based approach](https://astro.paperswithcode.com/paper/a-region-based-deep-learning-approach-to); [RetailCounter GitHub](https://github.com/davidanastasiu/RetailCounter); [SKKU paper](https://pure.skku.edu/en/publications/improving-deep-learning-based-automatic-checkout-system-using-ima/)
- **PoseLift** (WACV 2025 workshop): real store CCTV converted to anonymized pose sequences with boxes and track IDs, no raw pixels released. Shoplifting is treated as **anomaly detection over pose**. The authors report that pose-based anomaly detectors reach "high accuracy" while avoiding privacy and bias issues. — [arXiv 2501.06591](https://arxiv.org/html/2501.06591v1); [CVF open access](https://openaccess.thecvf.com/content/WACV2025W/ASTAD/html/Rashvand_Exploring_Pose-Based_Anomaly_Detection_for_Retail_Security_A_Real-World_Shoplifting_WACVW_2025_paper.html); repo github.com/TeCSAR-UNCC/PoseLift
- **Assembly SOP compliance**: an electronic-device lens-assembly dataset has 90 videos (72 compliant, 18 non-compliant) and 13 steps per SOP. That is a small-data regime similar to ours. — [IEEE DataPort EDA](https://ieee-dataport.org/documents/computer-vision-human-action-recognition-electronic-device-assembly-eda)
- **Cross-view action recognition** is an established problem. Its motivating case is surveillance where "it is not practical… to collect large amounts of training data when adding a new camera". Contrastive view-invariant representations across RGB and skeleton have been proposed. — [arXiv 2305.01733](https://ar5iv.labs.arxiv.org/html/2305.01733); [Wang et al. CVPR 2014](https://openaccess.thecvf.com/content_cvpr_2014/html/Wang_Cross-view_Action_Modeling_2014_CVPR_paper.html)

### Inferences
- Track 3 is the closest competition analogue in structure (staged actors, few videos, untrimmed footage, temporal localization scored). Its lesson is that **boundary post-processing and ensembling contributed as much as the backbone**: merging fragments, adaptive per-class thresholds, weighted boundary fusion. Our localization score (15 points) would gain from the same step: merge detections separated by gaps under a few seconds, apply minimum durations per class, and refine start and end times on per-frame signals.
- Track 5's association module (head with rider with motorbike) mirrors our need to **associate cash with a desk and a person, or a safe door with a cashier**. Detecting the parts separately and then associating them by rules is the proven pattern.
- PoseLift supports a privacy-compliant, **identity-free pose representation**. That fits the "no face recognition" constraint and travels across camera angles better than raw pixels.
- Retail Counter's hand inpainting and the Everseen patent both point to hands and objects in a defined region as the core signal for handling events, which bears on "counting without a machine" and "CIT bag handover".

### Gaps
- I did not find the 2025 (9th) AI City Challenge winners or a 2025 retail-checkout or SOP track. — [arXiv 2508.13564](https://arxiv.org/pdf/2508.13564v1) exists but was not examined.
- I found no public dataset of bank teller or cash-room violations, and no Kaggle competition on cash-handling SOP.
- The specific backbones used by the Track 3 winners (VideoMAE, X3D, etc.) are not stated in the summary paper.

---

## Q3. What worked in hackathons and competitions with small data and strict runtime?

### Takeaway
The repeated recipe: strong pretrained backbones fine-tuned lightly with K-fold CV; ensembles fused by WBF (detection) or boundary fusion (temporal); multi-view or most-visible-view selection; TTA and multi-scale augmentation; rare-class recall boosting; and heavy rule-based post-processing. DINOv2-backbone detectors such as RF-DETR are documented to adapt well to small datasets and run in real time on a T4.

### Cited Findings
- Ensembles plus WBF/NMS plus TTA were nearly universal among Track 5 leaders. Class imbalance was the main problem and was addressed with rare-class recall algorithms and confidence correction. — [arXiv 2404.09432](https://arxiv.org/html/2404.09432v1)
- K-fold fine-tuning of several action models, then **per-class view selection**. — [Nguyen et al. CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/AICity/html/Nguyen_Multi-View_Spatial-Temporal_Learning_for_Understanding_Unusual_Behaviors_in_Untrimmed_Naturalistic_CVPRW_2024_paper.html)
- Change-point proposals plus Video-LLM recognition were used in Track 3 2024. — [arXiv 2404.09432](https://arxiv.org/html/2404.09432v1)
- RF-DETR uses a DINOv2 backbone that Roboflow says "adapts well to small datasets and complex, occluded scenes". RF-DETR-L reaches 56.5 AP on COCO at **6.8 ms on an NVIDIA T4** (TensorRT FP16). Nano through Large models are Apache-2.0; XL and 2XL are under PML 1.0. Accepted at ICLR 2026. — [RF-DETR docs](https://rfdetr.roboflow.com/1.8.0/); [Roboflow blog](https://blog.roboflow.com/rf-detr/); [PyPI rfdetr](https://pypi.org/project/rfdetr/)
- RTMO (one-stage multi-person pose): RTMO-s 0.677 AP at 8.9 ms, RTMO-m 0.709 at 12.4 ms, RTMO-l 0.724 at 19.1 ms (COCO, V100, ONNXRuntime). The authors claim about 9x the speed of comparable one-stage methods. — [mmpose RTMO project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo)

### Inferences
- With 40 staged videos, **training a heavy end-to-end video classifier for 8+ classes is risky**. Top teams fine-tuned strong pretrained models and leaned on post-processing. The more robust path is to fine-tune a small detector (RF-DETR-S/M) on ~10 object and state classes (safe door open/closed, cash stack, cash bag, counting machine, personal phone/bag, door open), then derive violations with rules. Detector classes transfer across angles better than whole-clip action labels.
- Runtime budget (my estimate, unverified): RF-DETR-S/M at about 3–5 ms and RTMO-s at about 9 ms per frame on a T4 with TensorRT. At 2–5 fps sampling, a 5–10-minute video takes well under a minute, which leaves room for a VLM or open-vocabulary verification pass on a few candidate frames.
- Multi-view augmentation for the new-angle test: random perspective/homography warps, rotation, flips, crops and scale jitter during detector fine-tuning. This copies the multi-scale/crop augmentation and TTA the winners used.
- K-fold CV on 40 videos (split by video, not by frame) gives an honest estimate. Hold out one camera angle, if the data has more than one, to simulate the new-angle test.

### Gaps
- I found no Kaggle writeup that matches small-data SOP violation detection with runtime limits.
- I did not verify published T4 numbers for RTMO or for open-vocabulary detectors.

---

## Q4. Which open-source, commercially licensed building blocks are commonly used? (licenses explicit)

### Takeaway
Safe permissive choices: RF-DETR N–L (Apache-2.0), SAM 2 (Apache-2.0 checkpoints and code), OWLv2 (Apache-2.0), Grounding DINO (Apache-2.0 per a secondary source), the 100DOH hand-object detector (MIT code; weights unlicensed), RTMPose/RTMO via MMPose (Apache-2.0 repo, but training datasets have their own terms). Avoid or isolate: YOLO-World (GPL-3.0; Roboflow even lists AGPL-3.0) and Ultralytics YOLO (AGPL-3.0).

### Cited Findings
| Component | Role | License (as found) | Source |
|---|---|---|---|
| RF-DETR Nano–Large | Fine-tunable real-time detector | Apache-2.0 code and weights; XL/2XL under PML 1.0 (Roboflow account) | [RF-DETR docs](https://rfdetr.roboflow.com/1.8.0/), [Roboflow blog](https://blog.roboflow.com/rf-detr/) |
| SAM 2 | Promptable segmentation/tracking for zone auto-setup and object masks | "SAM 2 model checkpoints, demo code, training code are licensed under Apache 2.0". Repo also shows BSD-3-Clause files | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| OWLv2 (google/owlv2-base-patch16-ensemble) | Open-vocabulary detection | Apache-2.0 on HF. Model card calls it "a research output for research communities", but that is not a license restriction | [HF model card](https://huggingface.co/google/owlv2-base-patch16-ensemble) |
| Grounding DINO / Grounded-SAM | Open-vocabulary detection plus segmentation | Apache-2.0 (secondary listing; verify in the IDEA-Research repo) | [Roboflow Grounded SAM](https://playground.roboflow.com/models/idea-research/grounded-sam) |
| YOLO-World | Open-vocabulary real-time detector | GPL-3.0 upstream. Roboflow's own table says AGPL-3.0, which conflicts. Commercial use requires meeting GPL terms or a commercial license | [Roboflow YOLO-World license](https://roboflow.com/model-licenses/yolo-world), [Luxonis](https://models.luxonis.com/luxonis/yolo-world-l/6684e96f-11fc-4d92-8657-12a5fd8e532a) |
| Ultralytics YOLO (v8/11) | Detector/pose | AGPL-3.0 unless a commercial license is bought (Roboflow is an authorized licensor) | [Roboflow x Ultralytics](https://roboflow.com/ultralytics) |
| ddshan/hand_object_detector (100DOH, CVPR 2020) | Hand boxes with **contact state** (none / self / other-person / portable-object / stationary-object), L/R side, offset to the touched object, object box | Code: MIT. Weights: no license stated. Ego-trained model also carries Epic-Kitchens/EGTEA/CharadesEgo terms | [GitHub](https://github.com/ddshan/hand_object_detector), [hand_detector.d2](https://github.com/ddshan/hand_detector.d2) |
| RTMO / RTMPose (MMPose) | Multi-person pose, identity-free | MMPose repo (license not shown on the fetched page; see Inferences). Released weights are trained on COCO or "body7" (AI Challenger, COCO, CrowdPose, Halpe, MPII, PoseTrack18, sub-JHMDB), so each dataset's terms apply | [mmpose RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo) |
| NVIDIA VSS blueprint | Reference VLM-plus-CV agent architecture | NVIDIA terms (not open-source Apache); hosted NIM SAM 2 falls under the NVIDIA Community Model License | [NVIDIA blog](https://blogs.nvidia.com/?p=80495), [NVIDIA NIM SAM2](https://docs.api.nvidia.com/nim/reference/meta-sam2) |

### Inferences
- From prior knowledge, **not verified in this session**: the MMPose repo is Apache-2.0, ByteTrack is MIT, and the supervision library (Roboflow) is MIT. Check the LICENSE files before the final architecture slide.
- The 100DOH hand-object detector's contact-state output (portable vs. stationary object contact) is a ready-made signal for "cashier is handling cash", "hands on the counting machine" and "handover between two people" (other-person contact). Its Faster R-CNN ResNet-101 is heavy, and the weights have no explicit license. A safer route is a **hand class added to our own RF-DETR fine-tune, plus wrist keypoints from RTMO** to compute hand-to-object proximity.
- For **zone auto-setup on a new angle**: run OWLv2 or Grounding DINO with prompts ("safe", "metal cabinet door", "door", "desk", "cash counting machine") on the first frames, refine the boxes into masks with SAM 2, and store them as zones. Offer a UI to confirm or adjust (human in the loop). Everything here is Apache-2.0.
- Do not use YOLO-World or Ultralytics in the final pipeline if "commercial license" is a hard requirement. Judges may check this, and the architecture criterion (10 points) rewards license hygiene.

### Gaps
- No authoritative per-weight license for the 100DOH model weights.
- Licenses of the datasets behind RTMO's body7 weights (AI Challenger, Halpe, etc.) were not checked. The COCO-only RTMO weights carry less risk.
- I did not confirm Grounding DINO's license directly in the IDEA-Research GitHub repo.

---

## Q5. How do systems separate a "violation" from "normal" while keeping false positives low?

### Takeaway
Industry depends on (1) **dwell/persistence timers** (the most cited false-alarm lever), (2) **multi-signal confirmation** (video plus POS or sensors; association of detected parts), (3) **alert ranking or verification**, now often by a VLM agent, and (4) **human-in-the-loop review** with evidence clips. True events are rare compared with alarms, so precision engineering matters more than raw recall.

### Cited Findings
- For abandoned-object analytics, IPVM practitioners name **dwell/delay time** as the key parameter for cutting false alarms and say it can be set very long, or "you will be responding to nuisance alerts all day" (2018 discussion). — [IPVM discussion](https://ipvm.com/discussions/analytics-abandoned-object-in-crowded-scene)
- A US patent (assignee not checked) on attribute-based alert ranking says up to 50% of abandoned-bag false positives come from people's visual attributes matching alarm profiles, and true drops are only 1–2% of all alarms. The answer it proposes is **ranking and adjudicating alerts by attributes**. — [USPTO 9619715](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9619715)
- Axis Loitering Guard combines a time-in-area threshold with False Alarm Filtering and perspective-based size filtering. — [Axis Loitering Guard](https://www.networkwebcams.co.uk/axis-loitering-guard-application-licence/)
- Everseen decides a non-scan by **correlating** per-frame network probabilities with POS events, not from video alone. Alerts go to staff with an image, and the shopper is asked to rescan, so a person verifies before escalation. — [Justia patent](https://patents.justia.com/patent/20210216785); [Grocery Gazette](https://www.grocerygazette.co.uk/?p=118219)
- MCPRL (AI City 3rd) used **adaptive thresholding** to separate similar actions from background. TeleAI used weighted boundary fusion to keep high-confidence segments only. — [arXiv 2404.09432](https://arxiv.org/html/2404.09432v1)
- NVIDIA VSS includes agent workflows for "real-time alert verification", where a VLM checks CV-triggered alerts. — [NVIDIA Perspectives FAQ](https://perspectives.nvidia.com/nvidiametropolis/task/faq/vision-language-models-video-understanding-1/) (auto-generated-looking source); [NVIDIA dev blog](https://developer.nvidia.com/blog/?p=98690)
- Eocortex's dual-control rule is a count-based predicate ("at least 2 people in the vault"), checked continuously. — [Eocortex](https://eocortex.com/news/how-eocortex-analytics-improve-banks-performance)
- Vendors deliver alerts with short clips and snapshots for supervisor review (Wobot clips, Agrex event clips, Everseen item image). — [IndianWeb2/Wobot](https://www.indianweb2.com/2020/08/video-analytics-platform-wobot-raises.html); [Agrex](https://www.agrexai.com/banking-video-analytics); [Retail TouchPoints](https://www.retailtouchpoints.com/features/solution-spotlight/everseen-uses-ai-to-detect-non-scanning-items-at-checkout)

### Inferences
- Proposed decision layer for our task, combining these practices:
  1. **Per-frame evidence** from detector, pose and tracker: object states (safe door open, cash visible, bag present), person roles (cashier = the track that stays in the cashier zone longest; visitor = a track entering via the door zone), and hand-to-object proximity.
  2. **Temporal persistence** with hysteresis per rule, for example "safe open AND no person within R px for at least 10 s", "cash on desk AND cashier outside the desk zone for at least 5 s", "door open for at least 15 s". Smooth with an EMA or k-of-n frames and merge gaps under about 2–3 s. This is the dwell-time lever, and it also improves localization.
  3. **Multi-signal confirmation**: require at least two independent cues (detector state plus pose/zone, or detector plus SAM 2 mask change) before raising an alert.
  4. **Verifier pass**: a small local VLM (commercial-license model, run only on 1–3 keyframes per candidate) answers a yes/no question tied to the rule, and its answer is used to re-rank or drop candidates. This mirrors VSS alert verification and the Video-LLM recognizer from Track 3.
  5. **Evidence pack for human review**: timecode range, best frame with boxes drawn, confidence, and the NBRK rule text. Review mode doubles as product value (10 points), echoing vendor dashboards.
- Unseen violation types can be partly covered by a VLM or open-vocabulary "anomaly" pass flagged at lower confidence ("needs review"). This is safer than discarding it, and the human-in-the-loop framing excuses lower precision on that tier.
- Thresholds (T seconds, radius R) should be set in **normalized units** (fraction of the person's height or the desk width) rather than pixels, so they carry over to a new camera angle. This parallels Axis's perspective configuration.

### Gaps
- No published false-positive rates for bank or cash-room analytics from independent evaluation.
- No source quantified how much dwell timers or VLM verification cut false positives in comparable settings.
- Vendor claims ("zero false alerts", "~100% SOP compliance") are unverified marketing.
