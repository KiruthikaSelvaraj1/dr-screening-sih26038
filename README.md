# Explainable Diabetic Retinopathy Screening

A screening-aid system for diabetic retinopathy (DR) that grades severity and backs its prediction with visible, named lesion evidence rather than a black-box score.

Inspired by **SIH 2026 Problem Statement 26038** (MathWorks, MedTech/BioTech/HealthTech theme). Built independently in Python/PyTorch — **not an official SIH submission** (the official track expects a MATLAB-based pipeline).

> **Screening aid only — not a diagnosis. A clinician must confirm every result.**

---

## Why this exists

Diabetic retinopathy is a leading cause of preventable blindness, and rural India has too few ophthalmologists to screen everyone who needs it. Two things make automated screening hard to trust in practice:

- Portable fundus cameras in the field often produce images that are too blurry, poorly lit, or incompletely framed to grade safely.
- Most DR classifiers are black boxes — they output a severity label with no way for a clinician to check *why*.

This system tackles both: a **quality gate** that rejects ungradeable images before they reach the model, and an **evidence-based explanation** — the segmentation model's lesion detections, not a vague heatmap alone — attached to every grading.

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

The consistency check is the system's core idea: the segmentation model acts as an independent check on the classifier's grade, rather than just a decorative overlay. If the classifier says "No DR" but the segmentation model finds several lesions — or the reverse — the report flags it for manual review instead of presenting a confident-looking but contradicted answer.

---

## How this maps to SIH 2026 Problem Statement 26038

The official PS asks for more than this build currently covers — the table below states honestly what's done, partial, or intentionally out of scope.

| PS requirement | Status |
|---|---|
| Image quality assessment & enhancement | ✅ Done |
| Retinal structures — microaneurysms, hemorrhages, exudates, optic disc | ✅ Done, all reportable on held-out test data |
| Fovea, vessels, neovascularization | ❌ Not built — IDRiD's 81 pixel-annotated images limit how many lesion types can be reliably trained from scratch; scoped as future work |
| DR severity grading, sensitivity >90% / specificity >85% | ✅ **Met** (91.3% / 93.3%) |
| Explainability — lesion-level evidence + confidence | ✅ Done, via lesion coverage reporting instead of a heatmap alone (see [Results](#results)) |
| Grad-CAM specifically | ✅ Done — shown alongside the lesion-evidence report, not replacing it (see below) |
| Calibrated confidence | ✅ Done — temperature scaling, fit on the validation set (see [Results](#results)) |
| Doctor review workflow, <30 sec target | ⚠️ Report format supports fast review; no confirm/override UI, no timing measured |
| Simulink district-level workflow simulation | ❌ Not built — out of scope for this build |
| External benchmark validation | ❌ Not done — tested only on APTOS/IDRiD's own held-out splits |

**Why lesion evidence instead of Grad-CAM:** a heatmap shows *where* a model focused, not *what* it found. This system instead reports named, segmented lesions with coverage and approximate location — closer to how a clinician would actually describe findings. This was a deliberate design choice to satisfy the PS's underlying goal (interpretable clinical reasoning) even where it departs from the literal technique named.

---

## Results

### Severity classification (APTOS, EfficientNet-B0) — held-out test set

The SIH problem statement sets an explicit clinical bar for referable DR (Moderate/Severe/Proliferative, grouped): **sensitivity >90%, specificity >85%**. Measured result:

| Metric | Result | Target |
|---|---|---|
| Sensitivity (referable DR) | **91.3%** | >90% ✅ |
| Specificity (referable DR) | **93.3%** | >85% ✅ |

### Lesion segmentation (IDRiD, held-out test set, 27 images)

Two models were trained. v1 trains on whole images resized to 512×768; v2 trains on 512×512 crops centred on lesions, specifically to fix v1's failure on tiny lesions. Each had strengths, so the final system **combines both**: v1 for hard exudates, v2 for everything else.

| Lesion / structure | v1 Dice | v2 Dice | Used in final system |
|---|---|---|---|
| Optic disc | 0.97 (val) | **0.905** | v2 |
| Hard exudates | **0.61** | 0.306 (regressed) | v1 |
| Hemorrhages | 0.52 (val) | **0.541** | v2 |
| Soft exudates | 0.22 (val, unreliable) | **0.506** | v2 |
| Microaneurysms | 0.00 (failed) | **0.503** | v2 |

All five lesion types are now reportable (Dice ≥ 0.4 on held-out test data). The microaneurysm fix was the main goal of v2 — it went from completely failing (0.00) to a genuinely usable 0.503, by training on lesion-focused high-resolution crops instead of a downsized whole image, since microaneurysms are only a few pixels wide and were erased by the resize v1 used. The one tradeoff: v2's hard-exudate score dropped, likely because its crop-sampling weighted microaneurysms and soft exudates more heavily during training; using v1's model for that one lesion type avoided retraining while keeping the best score for each.

### Explainability output

The report shows **lesion coverage percentage** (e.g. "hard exudates: covering 5.08% of the retina"), not a raw blob count — a single lesion cluster can fragment into dozens of small connected regions, so a count alone overstates severity. A rough spot count and approximate location are still shown alongside, for context.

**Consistency check:** the segmentation findings are cross-checked against the classifier's grade. If they disagree (e.g. "No DR" predicted but lesions are found) or the classifier's confidence is below 60%, the system responds with "refer for clinician review" instead of a confident-looking but unreliable answer. Example from testing: one image graded "Proliferative" at only 58% confidence correctly triggered this referral message rather than presenting the grade as settled.

### Qualitative check

Manual comparison against source images confirms the overlay lines up with real lesions — hard exudate and hemorrhage masks track the actual bright/dark lesion regions in the original photo (see `docs/result_example.png`). This also caught a real problem during development: v1's soft-exudate channel was found to mostly duplicate hard-exudate detections rather than finding genuine cotton-wool spots, which is why v2 (trained independently on lesion-focused crops) replaced it.

On a separate test case, the Grad-CAM heatmap's highest-attention regions did not fully align with where the lesion-evidence model found actual hemorrhages and exudates — a concrete illustration of why lesion-level evidence is treated as the primary explanation here, with Grad-CAM included as a secondary, PS-required view rather than the main source of truth.

### Confidence calibration

Raw softmax confidence from deep classifiers is typically overconfident. Temperature scaling (Guo et al., 2017) was fit on the validation set — a single scalar `T` divides the logits before softmax, correcting the confidence without changing any predictions. Expected Calibration Error (ECE) before and after is logged by `calibration.py` and saved alongside the fitted temperature in `models/temperature.json`; see that file for the exact before/after numbers from the training run.

---

## Tech stack

| Component | Technology | Why |
|---|---|---|
| Preprocessing | OpenCV | Blur/illumination detection, CLAHE, denoising |
| Augmentation | Albumentations | Transforms image + segmentation masks together, consistently |
| Segmentation | PyTorch + `segmentation_models_pytorch` (U-Net, ResNet34 encoder) | Standard architecture for medical image segmentation |
| Classification | PyTorch + `timm` (EfficientNet-B0, pretrained) | Strong accuracy-to-compute ratio; transfer learning avoids overfitting a small medical dataset |
| Evaluation | scikit-learn, Dice/IoU | Per-class sensitivity over raw accuracy, since DR severity classes are imbalanced |
| Demo UI | Streamlit | Interactive upload-and-predict web app |
| Inference API | FastAPI | Separates the model from the UI — the production pattern, not just a notebook-to-UI script |
| Containerization | Docker | Runs identically in any environment |
| CI | GitHub Actions | Builds and health-checks the Docker image on every push |
| Experiment tracking | Weights & Biases | Live per-epoch loss and per-lesion Dice charts during training |
| Explainability | Grad-CAM + lesion-evidence segmentation | Grad-CAM for the PS requirement; lesion evidence as the primary, more clinically specific explanation |
| Calibration | Temperature scaling | Corrects overconfident softmax output without changing predictions |
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
├── api.py                          # FastAPI inference service
├── Dockerfile
├── requirements.txt
├── .github/workflows/
│   └── docker-build.yml            # CI: builds + health-checks the Docker image
├── stage1_quality_assessment.py    # Quality gate + CLAHE enhancement
├── stage5_segmentation_train.py    # U-Net lesion segmentation (v1 — used for hard exudates)
├── stage5_v2_segmentation.py       # High-res, lesion-focused crop training (v2 — used for the rest)
├── stage6_classification_train.py  # EfficientNet-B0 severity classifier
├── stage7_pipeline.py              # End-to-end inference: combines v1+v2, evidence report
├── gradcam.py                      # Grad-CAM for the classifier
├── calibration.py                  # Temperature scaling (confidence calibration)
├── models/                         # Trained checkpoints (not committed — see below)
│   ├── best_model.pth
│   ├── seg_model.pth
│   ├── seg_model_v2.pth
│   ├── seg_model_v2_thresholds.json
│   └── temperature.json
└── docs/
    └── result_example.png          # Example original vs. overlay comparison
```

## Running it locally

**Streamlit demo:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows; use `source venv/bin/activate` on Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

**FastAPI service** (separate from the demo UI — this is how a real client would call the model):
```bash
uvicorn api:app --reload --port 8000
```
Interactive docs at `http://localhost:8000/docs`.

**Docker:**
```bash
docker build -t dr-screening-api .
docker run -p 8000:8000 dr-screening-api
```

Needs all four files in `models/` first (see the structure above) — see [`docs/setup.md`](docs/setup.md) for how to get or retrain them. Model checkpoints aren't committed as plain files (the segmentation models are ~90–100MB each, close to GitHub's single-file limit) — they're tracked with **Git LFS** instead.

---

## Known limitations

- **Segmentation was trained on only 43–54 images per model** (IDRiD's full pixel-annotated set is 81 images total). This is a known gap in the public dataset itself, not just this implementation, and is the likely cause of v2's hard-exudate regression.
- **No left/right eye detection**, so lesion locations are reported as image-relative (e.g. "upper-left") rather than clinical terms like nasal/temporal.
- **Fovea, vessel, and neovascularization detection are not implemented.**
- **No doctor-facing confirm/override UI**, and the PS's <30-second review target has not been measured.
- **Not clinically validated.** Trained and tested only on public datasets (APTOS, IDRiD), with no review by a practicing ophthalmologist and no deployment testing on real rural-camera images. This is a research prototype, not a validated clinical tool.
- **Not evaluated for generalization** across different fundus camera models or populations beyond what APTOS/IDRiD cover — no external benchmark (e.g. Messidor-2) validation has been run.

---

## Future work

- Fovea and vessel segmentation (DRIVE dataset); investigate whether "Refined IDRiD" (2026) has enough additional lesion annotations to help retrain a more robust single segmentation model
- Rebalance v2's crop-sampling weights and retrain, to recover hard-exudate performance without needing to keep two separate checkpoints
- External validation on a second dataset (Messidor-2) to check generalization beyond APTOS/IDRiD
- Quantify how often the classifier and segmentation model agree vs. disagree across the full test set, as a measurable result for the consistency-check idea
- On-device/offline deployment (quantization via ONNX/TFLite) for actual rural field use with poor connectivity

---

## Disclaimer

This is a student-built research system. It is not a medical device, has not been clinically validated, and must not be used for real patient screening or diagnosis without proper clinical oversight and regulatory approval.