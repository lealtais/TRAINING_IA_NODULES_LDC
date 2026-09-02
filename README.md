# Early Lung Nodule Detection — TCC (Undergraduate Thesis)

Object detection project for **early lung nodule detection** in CT scans,
using YOLOv8. Developed as part of an undergraduate thesis in early-stage
lung cancer screening support.

**Goal:** localize the nodule itself in a CT slice (bounding box), not
classify malignancy.

## Repository structure

```
.
├── README.md
├── experiment_log.md        # main experiment: dataset choice, training runs, external tests
├── deeplesion_tests.md       # side experiment: DeepLesion model tested on other organs
└── assets/                   # images used in the docs above (original + model detections)
```

## Summary

Two datasets were evaluated:

- **DeepLesion** (NIH) — generic multi-organ lesion dataset. Used first,
  later discarded as the main dataset for not being lung-specific.
- **LIDC-IDRI-slices** — lung-nodule-specific dataset (875 patients,
  ~15.5k annotated slices), with bounding boxes derived from combining
  four radiologists' annotation masks per nodule. Chosen as the final
  dataset.

Best result so far (YOLOv8s, LIDC-IDRI, 50 epochs, `optimizer=AdamW`):

| precision | recall | mAP50 |
|---|---|---|
| 0.809 | 0.712 | 0.754 |

Full details, training comparisons, bugs found along the way, and external
inference tests are documented in [`experiment_log.md`](experiment_log.md).

A side investigation testing the discarded DeepLesion model on non-lung
organs (pancreas, kidney) is documented in
[`deeplesion_tests.md`](deeplesion_tests.md).

## Status

Active thesis project — not production-ready. Current known limitations
(see `experiment_log.md` for details):
- Weak detection on images with multiple small (miliary) nodules
- Not tested/validated for clinical use
- Only supports CT slices, not X-rays

## Next steps

- Expand training data variety (multi-nodule cases, more image sources)
- Convert the trained model to TensorFlow Lite for mobile integration
- Build the mobile/desktop app that will use this model as a
  physician-support tool (flagging suspicious areas, not replacing
  diagnosis)
