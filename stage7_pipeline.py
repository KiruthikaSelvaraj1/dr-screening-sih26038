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

SEVERITY = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
LESION_NAMES = {"MA": "microaneurysms", "HE": "hemorrhages",
                "EX": "hard exudates", "SE": "soft exudates"}
# From validation Dice: MA = 0.00 (failed), SE = 0.22 (weak; mostly overlaps hard exudates). Update if you retrain.
NOT_ASSESSED = {"MA", "SE"}
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
                seg_path="/content/drive/MyDrive/p1/models/seg_model.pth"):
    cls_model = s6.build_model()
    cls_model.load_state_dict(torch.load(cls_path, map_location=s6.DEVICE))
    cls_model.eval()
    seg_model = s5.build_model()
    seg_model.load_state_dict(torch.load(seg_path, map_location=s5.DEVICE))
    seg_model.eval()
    return cls_model, seg_model


@torch.no_grad()
def classify(cls_model, image_rgb):
    _, val_tf = s6.get_transforms()
    x = val_tf(image=image_rgb)["image"].unsqueeze(0).to(s6.DEVICE)
    probs = torch.softmax(cls_model(x), dim=1)[0].cpu().numpy()
    return int(probs.argmax()), probs


def count_lesions(mask, min_area=3):
    """Connected blobs in a binary mask -> (count, list of centroids)."""
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return len(keep), [tuple(cents[i]) for i in keep]


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


def run_pipeline(image_rgb, cls_model, seg_model):
    """image_rgb: H x W x 3 uint8 (RGB). Returns a result dict."""
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

    # Stage 3: severity;  Stage 2: lesion masks (both on the raw image)
    sev_idx, probs = classify(cls_model, image_rgb)
    masks = s5.predict_masks(seg_model, image_rgb)

    # Evidence: counts + rough locations
    od_pts = np.argwhere(masks["OD"] > 0)
    ref = (od_pts[:, 1].mean(), od_pts[:, 0].mean()) if len(od_pts) else None
    min_area = max(3, int(1e-5 * h * w))

    evidence, total = {}, 0
    for c in ("MA", "HE", "EX", "SE"):
        n, cents = count_lesions(masks[c], min_area)
        regions = sorted({region_label(cx, cy, w, h, ref) for cx, cy in cents})
        evidence[c] = {"count": n, "regions": regions}
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
        if e["count"]:
            lines.append(f"- {e['count']} {LESION_NAMES[c]} detected ({', '.join(e['regions'])}){tag}")
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
            "enhanced": enhanced, "report": "\n".join(lines)}
