"""
Step 5: Lesion segmentation on IDRiD ("A. Segmentation")
U-Net (ResNet34 encoder) -> 5 mask channels: MA, HE, EX, SE, OD
  MA = microaneurysms, HE = hemorrhages, EX = hard exudates,
  SE = soft exudates,  OD = optic disc

Paths are discovered automatically (by folder keywords and the _MA/_HE/_EX/_SE/_OD
filename suffixes), and a summary is printed before training so mistakes show up early.
"""

import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

CLASSES = ["MA", "HE", "EX", "SE", "OD"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, W = 512, 768  # both divisible by 32; lesions are tiny, so much larger than 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------- Discover files ----------

def build_index(root):
    """Returns (originals, masks) for each split ('train' / 'test').
    originals[split][img_id] = path
    masks[split][img_id][class] = path
    """
    originals = {"train": {}, "test": {}}
    masks = {"train": {}, "test": {}}
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

    for f in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
        low = f.lower()
        ext = os.path.splitext(low)[1]
        if ext not in exts:
            continue
        split = "train" if "training" in low else "test" if "testing" in low else None
        if split is None:
            continue
        stem = os.path.splitext(os.path.basename(f))[0]

        if "groundtruth" in low:
            for c in CLASSES:
                if stem.endswith("_" + c):
                    img_id = stem[: -(len(c) + 1)]
                    masks[split].setdefault(img_id, {})[c] = f
        elif "original" in low and ext in (".jpg", ".jpeg", ".png"):
            originals[split][stem] = f

    return originals, masks


def print_index_summary(originals, masks):
    for split in ("train", "test"):
        print(f"[{split}] images found: {len(originals[split])}")
        for c in CLASSES:
            n = sum(1 for i in originals[split] if c in masks[split].get(i, {}))
            print(f"    {c}: masks for {n} images")


# ---------- Dataset ----------

class IDRiDSeg(Dataset):
    def __init__(self, ids, originals, masks, transform):
        self.ids, self.originals, self.masks, self.transform = ids, originals, masks, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        img = cv2.cvtColor(cv2.imread(self.originals[img_id]), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        chans = []
        for c in CLASSES:
            p = self.masks.get(img_id, {}).get(c)
            m = cv2.imread(p, cv2.IMREAD_UNCHANGED) if p else None
            if m is None:  # no mask file for this lesion type -> treat as empty
                chans.append(np.zeros((h, w), np.uint8))
                continue
            if m.ndim == 3:
                m = m.max(axis=2)
            m = (m > 0).astype(np.uint8)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            chans.append(m)

        mask = np.stack(chans, axis=-1)  # H x W x 5
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].permute(2, 0, 1).float()


def get_transforms():
    train_tf = A.Compose([
        A.Resize(H, W),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(H, W),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
    return train_tf, val_tf


# ---------- Model / metrics ----------

def build_model():
    return smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3,
                    classes=len(CLASSES)).to(DEVICE)


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    """Dice and IoU per lesion type, aggregated over the whole loader."""
    model.eval()
    inter = torch.zeros(len(CLASSES))
    pred_sum = torch.zeros(len(CLASSES))
    true_sum = torch.zeros(len(CLASSES))
    for x, y in loader:
        p = (torch.sigmoid(model(x.to(DEVICE))) > threshold).float().cpu()
        inter += (p * y).sum(dim=(0, 2, 3))
        pred_sum += p.sum(dim=(0, 2, 3))
        true_sum += y.sum(dim=(0, 2, 3))
    eps = 1e-6
    dice = (2 * inter + eps) / (pred_sum + true_sum + eps)
    iou = (inter + eps) / (pred_sum + true_sum - inter + eps)
    return {c: (round(dice[i].item(), 4), round(iou[i].item(), 4)) for i, c in enumerate(CLASSES)}


# ---------- Training ----------

def train_seg(root="/content/drive/MyDrive/p1/A. Segmentation", epochs=60, batch_size=4,
              lr=3e-4, save_path="/content/drive/MyDrive/p1/models/seg_model.pth"):
    originals, masks = build_index(root)
    print_index_summary(originals, masks)

    ids = sorted(originals["train"])
    assert ids, "No training images found - check the folder path / structure."

    tr_ids, va_ids = train_test_split(ids, test_size=0.2, random_state=42)
    train_tf, val_tf = get_transforms()
    tr_loader = DataLoader(IDRiDSeg(tr_ids, originals["train"], masks["train"], train_tf),
                           batch_size=batch_size, shuffle=True, num_workers=2)
    va_loader = DataLoader(IDRiDSeg(va_ids, originals["train"], masks["train"], val_tf),
                           batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"train images: {len(tr_ids)} | val images: {len(va_ids)}")

    model = build_model()
    dice_loss = smp.losses.DiceLoss(mode="multilabel", from_logits=True)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best = -1.0

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                out = model(x)
                loss = dice_loss(out, y) + bce(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * x.size(0)
        sched.step()

        scores = evaluate(model, va_loader)
        mean_dice = float(np.mean([d for d, _ in scores.values()]))
        print(f"Epoch {epoch+1}/{epochs} | loss={total/len(tr_ids):.4f} | "
              f"val mean Dice={mean_dice:.4f} | " +
              " ".join(f"{c}:{d:.2f}" for c, (d, _) in scores.items()))

        if mean_dice > best:
            best = mean_dice
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved best model (mean Dice={best:.4f})")

    print(f"Done. Best val mean Dice={best:.4f}. Saved to {save_path}")
    return model


def evaluate_test(root="/content/drive/MyDrive/p1/A. Segmentation",
                  model_path="/content/drive/MyDrive/p1/models/seg_model.pth"):
    """Final check on the official held-out IDRiD test images (run once, after training)."""
    originals, masks = build_index(root)
    ids = sorted(originals["test"])
    _, val_tf = get_transforms()
    loader = DataLoader(IDRiDSeg(ids, originals["test"], masks["test"], val_tf),
                        batch_size=2, shuffle=False)
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    scores = evaluate(model, loader)
    for c, (d, i) in scores.items():
        print(f"{c}: Dice={d:.4f}  IoU={i:.4f}")
    return scores


@torch.no_grad()
def predict_masks(model, image_rgb, threshold=0.5):
    """image_rgb: H x W x 3 uint8 array. Returns {class: binary mask at original size}.
    Used later by the evidence-report step."""
    model.eval()
    h, w = image_rgb.shape[:2]
    x = A.Compose([A.Resize(H, W), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])(image=image_rgb)["image"]
    p = torch.sigmoid(model(x.unsqueeze(0).to(DEVICE)))[0].cpu().numpy()
    return {c: cv2.resize((p[i] > threshold).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            for i, c in enumerate(CLASSES)}
