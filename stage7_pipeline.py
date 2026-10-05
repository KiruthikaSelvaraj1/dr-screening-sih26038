"""
Step 7: End-to-end pipeline + evidence report
quality gate -> classification + segmentation -> lesion counts -> overlay + text report

Needs stage1_quality_assessment.py, stage5_segmentation_train.py and
stage6_classification_train.py in the same folder (p1).
"""

import cv2
import numpy as np
import torch

import stage1_quality_assessment as s1
import stage5_segmentation_train as s5
import stage6_classification_train as s6
import gradcam

SEVERITY = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
LESION_NAMES = {"MA": "microaneurysms", "HE": "hemorrhages",
                "EX": "hard exudates", "SE": "soft exudates"}
# Held-out TEST Dice with the combined v1+v2 models (v1 for EX, v2 for the rest):
#   MA=0.50  HE=0.54  EX=0.61(v1)  SE=0.51  OD=0.91
# All five clear the 0.40 reliability bar, so nothing is excluded currently.
# Update this set if you retrain either model and a lesion type drops below your bar.
NOT_ASSESSED = set()
LOW_RELIABILITY = set()

# Generic screening guidance - check against your local clinical guidelines before real use.
RECOMMENDATION = [
    "Routine rescreening as per the local screening schedule.",
    "Rescreen at a closer interval; a clinician should set the timing.",
    "Refer to an eye specialist for evaluation.",
    "Urgent referral to an eye specialist.",
    "Urgent referral to an eye specialist.",
]
COLORS = {"MA": (255, 0, 0), "HE": (0, 0, 255), "EX": (255, 255, 0),
          "SE": (255, 0, 255), "OD": (0, 255, 0)}  # RGB


def load_models(cls_path="/content/drive/MyDrive/p1/models/best_model.pth",
                seg_v1_path="/content/drive/MyDrive/p1/models/seg_model.pth",
                seg_v2_path="/content/drive/MyDrive/p1/models/seg_model_v2.pth"):
    """
    v2 (lesion-focused crop training) is better for MA, HE, SE, OD, but regressed
    on hard exudates (test Dice 0.61 for v1 vs 0.31 for v2). So: use v1 for EX,
    v2 for everything else. Held-out test Dice used for this call:
      v1: EX=0.61 | v2: MA=0.50 HE=0.54 SE=0.51 OD=0.91
    """
    import stage5_v2_segmentation as s5v2

    cls_model = s6.build_model()
    cls_model.load_state_dict(torch.load(cls_path, map_location=s6.DEVICE))
    cls_model.eval()

    seg_v1 = s5.build_model()
    seg_v1.load_state_dict(torch.load(seg_v1_path, map_location=s5.DEVICE))
    seg_v1.eval()

    seg_v2 = s5v2.s5.build_model() if hasattr(s5v2, "s5") else s5.build_model()
    seg_v2.load_state_dict(torch.load(seg_v2_path, map_location=s5.DEVICE))
    seg_v2.eval()

    return cls_model, {"v1": seg_v1, "v2": seg_v2, "v2_thresholds": s5v2.load_thresholds(seg_v2_path)}


@torch.no_grad()
def classify(cls_model, image_rgb):
    _, val_tf = s6.get_transforms()
    x = val_tf(image=image_rgb)["image"].unsqueeze(0).to(s6.DEVICE)
    probs = torch.softmax(cls_model(x), dim=1)[0].cpu().numpy()
    return int(probs.argmax()), probs


def count_lesions(mask, min_area=3):
    """Connected blobs in a binary mask -> (count, list of centroids).
    Blob count is approximate - one real lesion can fragment into several small
    blobs, especially for tiny lesions like microaneurysms. Use coverage_pct
    (below) as the more trustworthy number; keep count for rough location only."""
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return len(keep), [tuple(cents[i]) for i in keep]


def coverage_pct(mask):
    """Percent of the retinal image area covered by this lesion mask.
    More reliable than a blob count, since it isn't affected by one lesion
    fragmenting into multiple small connected components."""
    return round(100.0 * mask.sum() / mask.size, 2)


def region_label(cx, cy, w, h, ref=None):
    """Image-relative region (upper/lower, left/right) around the optic disc if found,
    else the image centre. Left/right eye is unknown, so no temporal/nasal terms."""
    rx, ry = ref if ref else (w / 2, h / 2)
    return ("upper" if cy < ry else "lower") + "-" + ("left" if cx < rx else "right")


def make_overlay(image_rgb, masks, alpha=0.45):
    overlay = image_rgb.copy()
    for c, m in masks.items():
        if c in NOT_ASSESSED or m.sum() == 0:
            continue
        overlay[m > 0] = (np.array(COLORS[c]) * alpha + overlay[m > 0] * (1 - alpha)).astype(np.uint8)
    return overlay


def run_pipeline(image_rgb, cls_model, seg_models):
    """image_rgb: H x W x 3 uint8 (RGB). Returns a result dict.
    seg_models: the dict returned by load_models() - {"v1", "v2", "v2_thresholds"}.
    """
    import stage5_v2_segmentation as s5v2
    h, w = image_rgb.shape[:2]

    # Stage 1: quality gate (checked at 224x224, the scale the thresholds were tuned on)
    small_bgr = cv2.cvtColor(cv2.resize(image_rgb, (224, 224)), cv2.COLOR_RGB2BGR)
    quality = s1.assess_quality(small_bgr)
    if not quality["gradeable"]:
        failed = [k for k in ("blur", "illumination", "field_of_view") if not quality[k]["pass"]]
        return {"status": "recapture", "quality": quality,
                "report": "Image could not be graded (" + ", ".join(failed) +
                          "). Please recapture the image."}

    enhanced = cv2.cvtColor(s1.enhance_image(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)),
                            cv2.COLOR_BGR2RGB)  # display only

    # Stage 3: severity + Grad-CAM (PS asks for Grad-CAM explicitly; lesion evidence
    # below remains the primary explanation - see README for why)
    sev_idx, probs = classify(cls_model, image_rgb)
    _, val_tf = s6.get_transforms()
    gradcam_overlay, _, _ = gradcam.compute_gradcam_overlay(
        cls_model, image_rgb, val_tf, s6.DEVICE, class_idx=sev_idx)

    # Stage 2: lesion masks - v1 for hard exudates (better on held-out test: 0.61 vs 0.31),
    # v2 for everything else (fixed MA/SE, same-or-better HE/OD)
    masks_v1 = s5.predict_masks(seg_models["v1"], image_rgb)
    masks_v2 = s5v2.predict_masks_v2(seg_models["v2"], image_rgb, seg_models["v2_thresholds"])
    masks = dict(masks_v2)
    masks["EX"] = masks_v1["EX"]

    # Evidence: counts + rough locations
    od_pts = np.argwhere(masks["OD"] > 0)
    ref = (od_pts[:, 1].mean(), od_pts[:, 0].mean()) if len(od_pts) else None
    min_area = max(3, int(1e-5 * h * w))

    evidence, total = {}, 0
    for c in ("MA", "HE", "EX", "SE"):
        n, cents = count_lesions(masks[c], min_area)
        regions = sorted({region_label(cx, cy, w, h, ref) for cx, cy in cents})
        evidence[c] = {"count": n, "regions": regions, "coverage_pct": coverage_pct(masks[c])}
        if c not in NOT_ASSESSED:
            total += n

    # Consistency check between the two models
    notes = []
    if sev_idx == 0 and total >= 5:
        notes.append("Classifier says No DR but segmentation found several lesions - review manually.")
    if sev_idx >= 2 and total == 0:
        notes.append("Classifier suggests moderate or worse DR but no lesions were segmented "
                     "- the evidence is weak, review manually.")

    lines = [f"Predicted severity: {SEVERITY[sev_idx]} ({probs[sev_idx]*100:.0f}% confidence)"]
    for c, e in evidence.items():
        if c in NOT_ASSESSED:
            lines.append(f"- {LESION_NAMES[c]}: not assessed (model is not reliable for this lesion type)")
            continue
        tag = " [low reliability]" if c in LOW_RELIABILITY else ""
        if e["coverage_pct"] > 0:
            lines.append(f"- {LESION_NAMES[c]}: present, covering {e['coverage_pct']}% of the retina "
                         f"(~{e['count']} spots, {', '.join(e['regions'])}){tag}")
        else:
            lines.append(f"- no {LESION_NAMES[c]} detected{tag}")
    rec = RECOMMENDATION[sev_idx]
    if probs[sev_idx] < 0.6 or notes:
        rec = "Refer for clinician review - the result is uncertain or the findings are inconsistent."

    lines += [f"Note: {n}" for n in notes]
    lines.append(f"Suggested next step: {rec}")
    lines.append("Screening aid only - not a diagnosis. A clinician must confirm.")

    return {"status": "graded", "quality": quality, "severity": SEVERITY[sev_idx],
            "confidence": float(probs[sev_idx]), "probs": probs.tolist(),
            "recommendation": rec,
            "evidence": evidence, "overlay": make_overlay(image_rgb, masks),
            "gradcam_overlay": gradcam_overlay,
            "enhanced": enhanced, "report": "\n".join(lines)}
