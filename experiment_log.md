# Early Lung Nodule Detection with YOLOv8 — Experiment Log

Living document tracking the development of the lung nodule detection
model (undergraduate thesis / TCC). Updated as new tests are run.

## Goal

Early detection of the lung nodule itself (localization), not malignancy
classification. The model should point to **where** the nodule is in a CT
slice, not just whether one exists.

## Datasets evaluated

### 1. DeepLesion (NIH) — discarded as the main dataset

- Subset used: `kmader/nih-deeplesion-subset` (Kaggle)
- 1,327 usable annotated images (out of ~33k downloaded — most rows in the
  annotation CSV didn't correspond to images present in this subset)
- **Limitation identified:** DeepLesion annotates lesions in any organ
  (liver, kidney, lymph node, lung, etc.), it is not specialized for lung
  nodules. Not the ideal dataset for this specific goal.
- Training result (YOLOv8s, 100 epochs): `mAP50 = 0.329`,
  `precision = 0.557`, `recall = 0.304`

### 2. LIDC-IDRI-slices — chosen dataset

- Source: `zhangweiled/lidcidri` (Kaggle)
- Structure: `patient/nodule/{images, mask-0, mask-1, mask-2, mask-3}`
- Each nodule has 4 annotation masks (one per radiologist)
- **Methodological decision:** the 4 masks were combined (logical union — a
  pixel counts as nodule if at least one of the 4 radiologists marked it),
  to make the bounding box more robust than relying on a single annotation
- 875 patients, 15,548 total slices; 15,486 with a nodule (62 with no mark
  in any of the 4 masks were discarded)
- Bounding box computed from the combined mask (minimal rectangle
  containing the whole white area)

## Training runs

| Model | Dataset | Epochs | imgsz | Optimizer | precision | recall | mAP50 |
|---|---|---|---|---|---|---|---|
| YOLOv8n | DeepLesion | 30 | 512 | default | 0.317 | 0.246 | 0.208 |
| YOLOv8s | DeepLesion | 100 | 512 | default | 0.557 | 0.304 | 0.329 |
| YOLOv8s | LIDC-IDRI | 50 | 224 | AdamW | **0.809** | **0.712** | **0.754** |

Switching to LIDC-IDRI (a dataset specialized in lung nodules rather than
generic lesions) produced a large gain across all metrics compared to
DeepLesion.

## Bugs found and fixes

1. **Broken "Muon" optimizer:** the Ultralytics version used had a bug in
   its default optimizer (`RuntimeError: view size is not compatible...`
   in `optim/muon.py`). Fix: explicitly force `optimizer="AdamW"` in the
   `model.train()` call.
2. **Model "saved" that didn't actually persist:** in one training session,
   the copy command to Google Drive (`shutil.copy2` to
   `/content/drive/MyDrive/...`) ran with no error, but Google Drive had
   never actually been mounted (`drive.mount(...)` was missing from that
   notebook) — the path existed only locally on the temporary Colab
   machine. When the session ended, that training run's model was lost.
   **Lesson learned:** always confirm `drive.mount()` completed
   successfully (authorization popup accepted) *before* saving, and verify
   the file directly on the Google Drive website — don't rely solely on
   the notebook's printed message.

## Inference tests on external images (outside the training dataset)

Model tested: `best.pt` from the YOLOv8s/LIDC-IDRI run (mAP50 0.754),
`conf=0.15`.

### Test 01

| Original | Model detection |
|---|---|
| ![original](assets/teste_01_original.jpg) | ![detection](assets/teste_01_deteccao.jpg) |

- **Result: 0 detections.**
- The image contains multiple small nodules scattered across both lungs
  (miliary pattern).
- **Hypothesis (to confirm):** this pattern of multiple small, diffuse
  nodules may differ from the predominant single/isolated nodule profile
  seen in LIDC-IDRI training, or the nodules may be too small for what the
  model learned to recognize in this first version.

### Test 02

| Original | Model detection |
|---|---|
| ![original](assets/teste_02_original.jpg) | ![detection](assets/teste_02_deteccao.jpg) |

- **Result: 2 detections.**
  - Box 1: confidence 0.70
  - Box 2: confidence 0.36
- **To confirm:** validate visually/clinically whether the marked regions
  actually correspond to nodules or are false positives (e.g., blood
  vessels, artifacts).

### Test 03 — Chest X-ray (different modality than training)

| Original | Model detection |
|---|---|
| ![original](assets/teste_03_original.png) | ![detection](assets/teste_03_deteccao.jpg) |

- **Result: 0 detections.**
- **Important methodological note:** this image is a chest **X-ray**, not
  a CT scan. The model was trained exclusively on CT slices, which look
  very different from an X-ray (a 2D projection of the whole overlapping
  chest, vs. a transverse cross-sectional slice). Zero detections here was
  expected and **does not represent a model failure** — it's simply an
  image modality outside the scope the model was trained for. Worth
  stating explicitly in the thesis: the model should not be used with
  X-rays, only with CT.

### Test 04 — New CT slice

| Original | Model detection |
|---|---|
| ![original](assets/teste_04_original.png) | ![detection](assets/teste_04_deteccao.jpg) |

- **Result: 1 detection**, confidence 0.29 (much lower than the 0.70 in
  Test 02).
- **To confirm:** validate whether the marked region actually corresponds
  to a nodule; the low confidence suggests model uncertainty — could be a
  subtler case or a false positive.

### Planned next tests

- [ ] More varied external images (single nodule, multiple nodules,
  different image quality)
- [ ] Compare confidence/precision on the model's own validation set vs.
  fully external images
- [ ] Investigate whether adjusting `conf` (confidence threshold) changes
  the Test 01 result
- [x] Test with an X-ray to confirm the model is CT-specific (Test 03 —
  confirmed: 0 detections, as expected)
- [ ] Investigate why Test 04's confidence (0.29) was much lower than
  Test 02's (0.70)

## Notes for the thesis / advisor meeting

- Clear technical comparison between two datasets, with a documented
  rationale for discarding DeepLesion
- Methodology for combining multiple annotations (4 radiologists) into a
  more robust bounding box
- Strong quantitative result (mAP50 0.754) alongside honestly identified
  limitations from external tests — valid discussion points for advising
