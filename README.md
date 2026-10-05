# Explainable Diabetic Retinopathy Screening

A screening-aid prototype for diabetic retinopathy (DR) that grades severity and backs its prediction with visible, named lesion evidence rather than a black-box score.

Portfolio project inspired by **SIH 2026 Problem Statement 26038** (MathWorks, MedTech/BioTech/HealthTech theme). Built independently in Python/PyTorch — **not an official SIH submission** (the official track expects a MATLAB-based pipeline).

> **Screening aid only — not a diagnosis. A clinician must confirm every result.**

---

## Why this exists

Diabetic retinopathy is a leading cause of preventable blindness, and rural India has too few ophthalmologists to screen everyone who needs it. Two things make automated screening hard to trust in practice:

- Portable fundus cameras in the field often produce images that are too blurry, poorly lit, or incompletely framed to grade safely.
- Most DR classifiers are black boxes — they output a severity label with no way for a clinician to check *why*.

This project tackles both: a **quality gate** that rejects ungradeable images before they reach the model, and an **evidence-based explanation** — the segmentation model's lesion detections, not a vague heatmap — attached to every grading.

---

## Pipeline

```
Fundus image
     │
     ▼
Stage 1 — Quality gate (blur / illumination / field-of-view checks)
     │                              │
  ungradeable                  gradeable
     │                              │
"Recapture requested"        CLAHE enhancement (display only)
                                     │
                                     ▼
                    ┌────────────────┴────────────────┐
                    ▼                                  ▼
         Stage 3 — Severity grading          Stage 2 — Lesion segmentation
          (EfficientNet-B0, APTOS)             (U-Net/ResNet34, IDRiD)
                    │                                  │
                    └────────────────┬─────────────────┘
                                     ▼
                  Stage 7 — Consistency check
          (do the grade and the lesion evidence agree?)
                                     │
                                     ▼
               Evidence report + suggested next step
          (routine recheck / refer / urgent referral /
           refer for clinician review if uncertain)
```

The consistency check is the project's core idea: the segmentation model acts as an independent check on the classifier's grade, rather than just a decorative overlay. If the classifier says "No DR" but the segmentation model finds several lesions — or the reverse — the report flags it for manual review instead of presenting a confident-looking but contradicted answer.

---

## Results

### Lesion segmentation (IDRiD, validation set, 11 images)

| Lesion / structure | Dice |
|---|---|
| Optic disc | 0.97 |
| Hard exudates | 0.61 |
| Hemorrhages | 0.52 |
| Soft exudates | 0.22 — unreliable, excluded from the report |
| Microaneurysms | 0.00 — failed at this resolution, excluded from the report |

Microaneurysms are only a few pixels wide in a full-resolution fundus photo; shrinking the image to fit the model erased them. The report marks both lesion types as **"not assessed"** rather than silently reporting zero, since a false "no lesions found" would be worse than admitting the limitation. A v2 training approach (high-resolution, lesion-focused crops) is in progress to address this — see [Known limitations](#known-limitations).

*(Held-out test-set Dice on the 27 official IDRiD test images: TODO — run `evaluate_test()` and fill in before final submission.)*

### Severity classification (APTOS, EfficientNet-B0)

*(Per-class recall/sensitivity, especially for Severe and Proliferative, and overall validation accuracy: TODO — paste from the classification report and fill in here. This matters more than the segmentation numbers for clinical relevance, since missing a severe case is the costliest mistake.)*

### Qualitative check

On manual inspection against the original image, the hard-exudate and hemorrhage overlays line up with the actual bright/red lesions in the source photo (see `docs/result_example.png`). Soft exudate predictions were found to mostly duplicate the hard-exudate detections rather than finding real cotton-wool spots, which is why that channel is excluded from the report.

---

## Tech stack

| Component | Technology | Why |
|---|---|---|
| Preprocessing | OpenCV | Blur/illumination detection, CLAHE, denoising |
| Augmentation | Albumentations | Transforms image + segmentation masks together, consistently |
| Segmentation | PyTorch + `segmentation_models_pytorch` (U-Net, ResNet34 encoder) | Standard architecture for medical image segmentation |
| Classification | PyTorch + `timm` (EfficientNet-B0, pretrained) | Strong accuracy-to-compute ratio; transfer learning avoids overfitting a small medical dataset |
| Evaluation | scikit-learn, Dice/IoU | Per-class sensitivity over raw accuracy, since DR severity classes are imbalanced |
| Demo | Streamlit | Interactive upload-and-predict web app |
| Training environment | Google Colab (T4 GPU) | Free GPU access |

## Datasets

- **[APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection)** — ~3,662 images, 5-class DR severity labels. Used to train the classifier.
- **[IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)** — 81 images with pixel-level lesion masks (microaneurysms, hemorrhages, hard/soft exudates, optic disc) + 516 images with severity/DME grading. Used to train and evaluate the segmentation model.

Datasets are not included in this repo. Download links and setup steps are in [`docs/setup.md`](docs/setup.md).

---

## Project structure

```
.
├── app.py                          # Streamlit demo
├── requirements.txt
├── stage1_quality_assessment.py    # Quality gate + CLAHE enhancement
├── stage5_segmentation_train.py    # U-Net lesion segmentation (v1)
├── stage5_v2_segmentation.py       # High-res, lesion-focused crop training (v2, in progress)
├── stage6_classification_train.py  # EfficientNet-B0 severity classifier
├── stage7_pipeline.py              # End-to-end inference + evidence report
├── models/                         # Trained checkpoints (not committed — see below)
└── docs/
    └── result_example.png          # Example original vs. overlay comparison
```

## Running it locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows; use `source venv/bin/activate` on Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

Needs `models/best_model.pth` (classifier) and `models/seg_model.pth` (segmentation) in place first — see [`docs/setup.md`](docs/setup.md) for how to get or retrain them. Model checkpoints aren't committed directly (the segmentation model is ~90–100MB); they're distributed via [Git LFS / a linked download — fill in whichever you choose].

---

## Known limitations

- **Microaneurysms and soft exudates are not reliably detected** at the current training resolution. The report marks them "not assessed" rather than reporting a false negative. A fix (training on high-resolution, lesion-centred crops) is scoped in `stage5_v2_segmentation.py`.
- **Segmentation was trained on only 43 images** (IDRiD's full pixel-annotated set is 81 images total). Per-lesion reliability varies sharply as a result — this is a known gap in the public dataset itself, not just this implementation.
- **No left/right eye detection**, so lesion locations are reported as image-relative (e.g. "upper-left") rather than clinical terms like nasal/temporal.
- **Not clinically validated.** Trained and tested only on public datasets (APTOS, IDRiD), with no review by a practicing ophthalmologist and no deployment testing on real rural-camera images. This is a research/portfolio prototype, not a validated clinical tool.
- **Not evaluated for calibration or generalization** across different fundus camera models — a real deployment would need testing across camera types, lighting conditions, and patient populations beyond what APTOS/IDRiD cover.

---

## Future work

- Fix microaneurysm/soft exudate detection via high-resolution lesion-focused training (in progress, see `stage5_v2_segmentation.py`)
- Fovea and vessel segmentation (DRIVE dataset)
- Report lesion area coverage instead of blob counts (current counts can over-count a single lesion cluster as many small patches)
- On-device/offline deployment (quantization via ONNX/TFLite) for actual rural field use with poor connectivity
- Quantify how often the classifier and segmentation model agree vs. disagree on the test set, as a measurable result for the consistency-check idea

---

## Disclaimer

This is a student portfolio project. It is not a medical device, has not been clinically validated, and must not be used for real patient screening or diagnosis without proper clinical oversight and regulatory approval.
