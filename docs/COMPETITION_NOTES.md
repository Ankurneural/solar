# Solar Filament Segmentation 2026: what matters and why

Sources: competition Overview/Data/Rules, the organizers' Self-Evaluation notebook,
discussion threads and public notebooks (read on 2026-10-15). Thread numbers refer to
`kaggle.com/competitions/filament-segmentation-2026/discussion/<id>`.

## 1. The task

- **Input:** 2048×2048 8-bit grayscale JPEGs, GONG H-alpha full-disk frames from 6 sites:
  Bh Big Bear, Ch Cerro Tololo, Lh Learmonth, Mh Mauna Loa, Th El Teide, Uh Udaipur.
  Names are `YYYYMMDDHHMMSSII`.
- **Train:** 707 frames (1,154 annotation sets: 411 frames with 1 annotator, 145 with 2, 151 with 3), 8,199 polygons.
  **Test:** 180 frames, about 90 on the public LB. Site and year mix match train.
- **Labels:** one closed polygon per filament (no holes, never multi-part), a manual spine
  polyline, and a chirality class (Left / Right / Unidentifiable). The class is irrelevant for scoring.
- **Output:** one row per predicted filament: `<obs_id>_<k>, <COCO RLE counts>`.
  The RLE must be Fortran order; a C-order RLE decodes transposed and silently scores 0.
  Masks must be disjoint (overlaps are rejected) and non-empty. Do not submit empty masks.
- **Judging:** 70% quantitative (LB PQ, mean Dice, Dice/IoU distributions, 1:n / n:1 counts)
  and 30% qualitative (report, visual morphology, code quality).
  - Deliverables: a 4-page report, a public repo with a pinned `requirements.txt`, and a notebook reproducing the predictions.
  - Deadline: Nov 15 2026, 06:00 UTC (Kaggle API; midnight at the end of Nov 14, US Central). Up to 5 submissions/day; 1 final selection.

## 2. The metric is instance-level, not pixel-level

Calculation (notebook): for each annotation set, take the IoU of every GT filament with every
predicted filament of that frame. A pair is a hit if IoU > 0.5.
PQ = ΣIoU(hits) / (TP + ½FP + ½FN), pooled over **all annotation sets**.
`src/metrics.py` reproduces this exactly (`tests/test_metric.py`).

Consequences, and how the pipeline handles each:

| fact | consequence | handled by |
|---|---|---|
| each annotator scored separately | aim at the consensus, not one annotator | soft annotator-mean targets (`targets.py`) |
| every spurious blob costs ½, whatever its size | small, unsure blobs are expensive | `min_area`, `min_score`, later a learned scorer |
| one GT split in 2 pieces | at best 1 TP + 1 FP, often 0 TP + 2 FP + 1 FN | closing / `merge_px` / spine grouping |
| 2 GTs merged into one blob | IoU < 0.5 with both, so 2 FN + 1 FP | `boundary` head, spine watershed |
| a hit needs IoU > 0.5 with loose, hand-drawn polygons | the mask's *extent* matters as much as detection | train on the polygons themselves; `grow_px` |
| frames with no GT are never scored | no penalty for predicting on them | — |

**Keep rule.** Adding a candidate with match probability *p* and expected IoU *q* adds *p·q*
to the numerator and always ½ to the denominator (FN→TP: +1−½; or a new FP: +½).
It raises PQ only if **p·q > PQ/2**, which is about 0.2 at realistic scores. This is the core
selection rule. Top public pipelines learn *p·q* per candidate (LightGBM) and gained about +0.06 PQ
over a fixed threshold.

**Ceiling.** Expert vs expert PQ is about **0.34** (SQ 0.63, only 54% of filaments matched;
under 200 px PQ is 0.14). Models beat that because they predict the *consensus*.
Honest scores are 0.35–0.50: the best documented honest pipeline has OOF 0.46 / LB 0.40.
**Anything above about 0.45 on the LB comes from the leak** (the public MAGFiLO release has test labels)
Simulated on our train set (10-28): submitting one published annotation set per frame scores **PQ 0.757**, because 411 of 707
frames have a single annotator, so a copy is a perfect match. With 139/180 test frames leaked: 0.77 × 0.757 + 0.23 × ~0.40 ≈ 0.68–0.70,
which is the current top of the LB (0.70). One expert scored against the other experts gets PQ **0.336** (SQ 0.634, RQ 0.530),
so honest pipelines at LB 0.40 already beat a human annotator.

**Annotators (train labels, 10-28).** 37 annotator ids. The first 4 digits look like a group: 010401/02/03 always label the same
frames together. Styles differ a lot: 4.3 to 10.5 filaments per set, and median filament area 930 to 2006 px. Assignment is spread
over all sites and years, so the annotator cannot be inferred from the image or its metadata. 411/707 frames (58%) are scored against a
single annotator. These come mostly from 030101 (10.5 fil/set), 060403, 040401, 030302, 040102 and 020101.
PQ pools TP/FP/FN over annotation sets, so a 3-annotator frame weighs 3x in the metric; our scorer currently weighs candidates per frame.
The leaked public MAGFiLO labels are not used for any purpose, including analysis. The same annotators and protocol are in train.
or from a copied leaked CSV. Organizers: "any PQ > 0.35 is of great value", and LB rank is
only a preliminary filter. Every audited pipeline drops about 0.05 from OOF to LB.
The public-half noise is ±0.03, so do not chase LB differences smaller than that.

## 3. What transfers from the crack/scratch projects

| technique (crack/scratch repos) | here | why |
|---|---|---|
| smp + timm encoders, YAML config, auto-resume, bf16 AMP | reused | same tooling |
| tiling + Hann blending (`src/tiling.py`) | reused for `mode: crop` | 2048 frames, thin barbs |
| positive-biased crop sampling, hard negatives | reused (`pos_frac`) | sunspots, plage and cloud edges are the FPs |
| Tversky (β>α) recall bias | **not used** | PQ punishes FPs as hard as FNs, and calibration matters for selection |
| pixel Dice / tolerant F1 for model selection | **replaced by PQ** | a better pixel Dice can mean more fragmented instances |
| threshold sweep | generalized to `tune_postprocess.py` | instance knobs matter more than the threshold |
| CC + small-component removal | kept as `grouping: cc` | baseline grouping |
| 3-channel replicated gray + ImageNet norm | 1-channel input (smp adapts conv1) | organizers ask to treat frames as grayscale |
| device-level split | **time-grouped folds** (`group_days: 14`) | filaments live for days; several sites see the same Sun on the same day |

New problems that cracks did not have: multiple annotators (label noise is the dominant error),
instance identity (fragment vs merge), a fixed circular field of view with limb darkening, and
6 instruments with different seeing and weather.

## 4. Data handling decisions

- **Disk.** GONG frames are pre-centred, at (1023.1, 1024.1) ±0.4 px with r = 899.7 ±0.7 px.
  Otsu detection fails on about 16% of frames, so `detect_disk` falls back to this prior.
  A limb fit would be better still (public best uses a ray-cast limb fit). That is a TODO.
- **Limb darkening.** Divide by the median radial profile, so the disk is about 1.0 and filaments about 0.6–0.9.
  Off-disk pixels are set to 0, and predictions there are clipped (`disk_margin`).
- **Resolution.** Median filament width is 11 px and median bbox short side 53 px at 2048.
  At 1024 that becomes about 5 / 26 px, which is fine for detection but loses barbs. Avoid 512.
  `baseline.yaml` uses the whole disk at 1024 (global context, fast).
  `crop2048.yaml` uses native 512 crops. Expect the crop model plus refinement to win on SQ.
- **Targets.** Soft consensus beat union targets in public reports (0.421 vs 0.386); the medoid annotator also works.
  Spine annotations are explicitly allowed for training (thread 736543).
- **Folds.** 5 folds, grouped by 14-day windows and stratified by site. Folds are fixed in
  `folds.csv`, so OOF predictions from different experiments can be compared and stacked.

## 5. Rules that constrain the design

- Inference may use **only the supplied test image**. No neighbouring frames, magnetograms or sequences.
- Training may use unlabeled external frames (other GONG frames, SDO/AIA), self-supervised pretraining,
  and our own labels copied onto frames a few minutes away (exclude ±1 h around test timestamps and disclose it).
- **No extra ground truth:** no public MAGFiLO labels (they overlap test) and no SWEFil.
- Winning code needs an OSI license. The data is CC BY-NC.
- AI assistance is allowed, but you must be able to explain and justify every component.
- Compute efficiency counts (how it is measured is still unanswered, thread 749526). Keep inference lean
  and report the time per image.

## 6. State of the public field (honest pipelines)

| pipeline | idea | OOF / LB |
|---|---|---|
| tushhaaarrrr | flat-field; dense U-Net ensemble (R34, ConvNeXt-T, UNet++) on soft targets + spine head, 512 crops, EMA, D4; 15-level threshold tree; LightGBM P(match)·E[IoU] scorer + keep rule τ≈0.26; rotated-crop refiner | 0.46 / 0.40 |
| juanmirod | YOLO11m@1280 → crop U-Net refiner@512 → paint by confidence | 0.44 / 0.37 |
| pandadev0069 | YOLO26s-seg at native 2048 | 0.42 / 0.36 |
| ektarr | 5-fold YOLO11s-seg@1024, medoid targets, seed-conditioned refiner, ≥2-fold agreement | 0.41 / 0.36 |
| timgluhkikh | YOLO11s detect → crop U-Net R34 [img, CLAHE, box seed] | 0.41 / 0.37 |
| anthonytherrien | tiled efficientvit U-Net, D4 TTA | — / 0.36 |

Things that did **not** help publicly: clDice / skeleton / Lovász losses (they cost calibration),
morphology-heavy post-processing (−0.01 to −0.10), neighbour-frame tricks, noisy-student on external frames,
and magnetogram heads.

### Update 2026-10-28 (research via competitors' public GitHub mirrors; Kaggle pages are JS-only)
Organizer details quoted by several teams (not read from the Kaggle pages themselves):
- The 70% quantitative part is PQ (primary) plus mean Dice (`torchmetrics` DiceScore), the IoU distribution, and
  fragmentation/over-merge counts (1:n, n:1). `evaluate.py` already reports 1:n / n:1, so put all of them in the report.
- Public LB ≈ 50% of the test set and private = the rest; max 5 submissions/day (these teams quote 2 final selections; the Kaggle UI allowed 1).
- Report: ACM acmart sigconf template (Overleaf `overleaf.com/read/vjztpvrnbrdh`), keep the fixed subtitle/intro/acknowledgment
  text, 4 pages excluding references, must include a pipeline diagram and segmentation examples. A public Git repo is required.
- Kaggle rejects a submission if two instances of one image share a pixel. Our selection paints with a `taken` mask, so this is satisfied.
- The public MAGFiLO release holds labels for 139 of the 180 test images (the leak). The organizers rescored all submissions in August
  (true PQ), so older public scores are void. No statement found on how inference efficiency is measured.

New honest pipelines (OOF / LB):
| pipeline | idea | OOF / LB |
|---|---|---|
| biconcavelens (workshop paper) | Mask R-CNN + YOLO11-seg + RT-DETR, per-detector isotonic calibration, R34 crop refiner with spine head | 0.461 / 0.40 |
| theainerd | EffNet-B3 U-Net + scSE, BCE+Tversky **+ BiCC loss (0.3)**, GBM rescorer τ 0.25 | 0.452 / 0.40 |
| BSergey2000 | R34 + EffNet-B3 (crop 768), 8-view TTA, LightGBM scorer, τ = PQ/2 | 0.432 / 0.36 |
| ameyypawar | limb-edge fit + radial-distance input channel; YOLO groups instances, U-Net bounds them | 0.421 / 0.37 |

Lessons reported:
- **BiCC** (bidirectional connected-component loss, arXiv 2609.30223): one term per GT component and per FP blob.
  It raised theainerd's folds by +0.01–0.02. It is now `loss.heads.<head>.bicc` (gate `c2bicc_f0`).
- Under-trained confidence heads cost +0.01 LB for biconcavelens (fixed by reweighting the classification loss).
- Several teams independently: the SQ ceiling is about the annotators' consensus (human matched IoU ≈ 0.64; refiners plateau at
  SQ 0.67–0.68), and recall (RQ) is the real lever. Their OOF→LB gaps of 0.06–0.08 come from fold-selection bias plus per-annotator
  GT-style variance (the same predictions score 0.26–0.38 PQ depending on the annotator), so our gap of ~0.05 is structural.
- Did not help them either: cross-architecture detector ensembles with greedy pixel assignment, pseudo-labelling, SAM/SAM2 as a detector,
  post-hoc rescue of missed candidates, and joint 2-axis grid search (it overfits a ~116-image validation set).
- **Annotation protocol (MAGFiLO paper, Sci Data 11:1031, 2024; PMC11437148).** Annotated on GONG's own sharpened JPEGs in V7 Darwin.
  Filaments are ignored if too small/faint, if they are small close-by groups, or if they lie beyond ~±70° from the central meridian
  (train GT: max |lon| 74°, 0.05% beyond 70°). A mask must be one piece, without holes, tight to the body, and include every visible barb
  (Unidentifiable-chirality filaments are drawn with less detail). Whether pieces form one filament was judged from the filament's
  evolution in time and chirality; the paper admits annotators disagree on it. Chirality agreement κ 0.47 (3-class).
  No numeric minimum size is given.
- Pretrained solar weights: NASA-IMPACT Surya (366M, SDO AIA/HMI, Apache-2.0) is HF-only (blocked here), the domain gap to Hα is large,
  and it is unproven. Skipped.

## 7. Plan

1. **Baseline on real data:** prepare, EDA (check the inter-annotator PQ ≈ 0.34), 5-fold
   `baseline.yaml`, tune, then submit once to calibrate OOF against LB (expect about −0.05).
2. **Native resolution:** `crop2048.yaml`. Compare OOF PQ and SQ; ensemble both in probability space.
3. **Candidate generation + learned selection (biggest expected gain):**
   - Build candidates from several thresholds (a threshold tree).
   - Compute features per candidate: mean/max prob, area, elongation, spine-head support, distance to limb,
     agreement across folds and TTA.
   - Train a GBM on OOF to predict IoU with the consensus.
   - Keep non-overlapping candidates with p·q > PQ/2.
4. **Instance refiner:** crop around each kept candidate at 2048, U-Net on [img, seed mask],
   to sharpen extent (SQ). Rotating the crop to the filament axis helps.
5. **Encoders and ensembling:** ConvNeXt-T / EfficientNet / UNet++; fold + D4 ensemble.
6. **Report and notebook:** ablation table (each step's OOF PQ delta), figures of morphology,
   runtime per image. This is 30% of the grade, so write it as we go.
