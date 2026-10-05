"""
Step 5 v2: high-resolution, lesion-focused segmentation training
Targets the v1 weakness on microaneurysms (MA) and soft exudates (SE):
  - trains on 512x512 crops from near-full-resolution images (tiny lesions stay visible)
  - samples crops centred on lesions most of the time (random crops rarely contain MA)
  - Dice + Focal loss (better for tiny positive regions)
  - tunes one probability threshold per lesion type on the validation images
  - evaluates on WHOLE images, and reports which lesion types pass a stated Dice bar

Reuses build_index / CLASSES / build_model from stage5_segmentation_train.py,
so the model architecture is identical to v1 (same U-Net, same checkpoint format).
"""

import os
import json
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

import stage5_segmentation_train as s5

CLASSES, DEVICE, MEAN, STD = s5.CLASSES, s5.DEVICE, s5.MEAN, s5.STD
LONG_SIDE = 2048                       # long side after resize (never upscaled)
CROP = 512
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6)
CLASS_SAMPLING = {0: 0.40, 3: 0.30, 1: 0.15, 2: 0.15}   # MA, SE, HE, EX


# ---------- Data ----------

def load_pair(img_path, mask_paths, long_side=LONG_SIDE):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = min(1.0, long_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    nh, nw = img.shape[:2]

    chans = []
    for c in CLASSES:
        p = mask_paths.get(c)
        m = cv2.imread(p, cv2.IMREAD_UNCHANGED) if p else None
        if m is None:
            chans.append(np.zeros((nh, nw), np.uint8))
            continue
        if m.ndim == 3:
            m = m.max(axis=2)
        m = (m > 0).astype(np.uint8)
        if m.shape != (nh, nw):
            # area-average then threshold, so tiny lesions are not lost by nearest-neighbour
            m = (cv2.resize(m * 255, (nw, nh), interpolation=cv2.INTER_AREA) > 60).astype(np.uint8)
        chans.append(m)
    return img, np.stack(chans, axis=-1)


def load_split(ids, originals, masks):
    return [load_pair(originals[i], masks.get(i, {})) for i in ids]


class CropDataset(Dataset):
    def __init__(self, data, n_samples, p_lesion=0.7):
        self.data, self.n, self.p_lesion = data, n_samples, p_lesion
        self.tf = A.Compose([
            A.PadIfNeeded(min_height=CROP, min_width=CROP, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
        self.pos = []
        for _, m in data:
            d = {}
            for k in range(len(CLASSES)):
                pts = np.argwhere(m[..., k] > 0)
                if len(pts) > 5000:
                    pts = pts[np.random.choice(len(pts), 5000, replace=False)]
                d[k] = pts
            self.pos.append(d)

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        i = random.randrange(len(self.data))
        img, m = self.data[i]
        h, w = img.shape[:2]

        cy = cx = None
        if random.random() < self.p_lesion:
            avail = [k for k in CLASS_SAMPLING if len(self.pos[i][k])]
            if avail:
                k = random.choices(avail, [CLASS_SAMPLING[k] for k in avail])[0]
                pts = self.pos[i][k]
                cy, cx = pts[random.randrange(len(pts))]
        if cy is None:
            cy, cx = random.randrange(h), random.randrange(w)

        cy += random.randint(-CROP // 4, CROP // 4)
        cx += random.randint(-CROP // 4, CROP // 4)
        y0 = int(np.clip(cy - CROP // 2, 0, max(h - CROP, 0)))
        x0 = int(np.clip(cx - CROP // 2, 0, max(w - CROP, 0)))

        out = self.tf(image=img[y0:y0 + CROP, x0:x0 + CROP], mask=m[y0:y0 + CROP, x0:x0 + CROP])
        return out["image"], out["mask"].permute(2, 0, 1).float()


# ---------- Whole-image inference / evaluation ----------

@torch.no_grad()
def predict_probs(model, img):
    """img: H x W x 3 uint8 (already resized to <= LONG_SIDE). Returns C x H x W probabilities."""
    h, w = img.shape[:2]
    x = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])(image=img)["image"]
    x = x.unsqueeze(0).to(DEVICE)
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
        p = torch.sigmoid(model(x))
    return p[0, :, :h, :w].float().cpu()


@torch.no_grad()
def evaluate_full(model, data, thresholds=THRESHOLDS):
    """Dice matrix [len(thresholds) x len(CLASSES)], aggregated over all images."""
    model.eval()
    T, C = len(thresholds), len(CLASSES)
    inter, ps, ts = torch.zeros(T, C), torch.zeros(T, C), torch.zeros(T, C)
    for img, m in data:
        p = predict_probs(model, img)
        y = torch.from_numpy(m).permute(2, 0, 1).float()
        for ti, t in enumerate(thresholds):
            b = (p > t).float()
            inter[ti] += (b * y).sum(dim=(1, 2))
            ps[ti] += b.sum(dim=(1, 2))
            ts[ti] += y.sum(dim=(1, 2))
    eps = 1e-6
    return (2 * inter + eps) / (ps + ts + eps)


# ---------- Training ----------

def train_v2(root="/content/drive/MyDrive/p1/A. Segmentation", epochs=60, batch_size=4, lr=3e-4,
             crops_per_epoch=200, eval_every=3,
             save_path="/content/drive/MyDrive/p1/models/seg_model_v2.pth",
             use_wandb=True, wandb_project="dr-screening"):
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, name="seg-v2-lesion-crops", config={
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "crops_per_epoch": crops_per_epoch, "crop_size": CROP, "long_side": LONG_SIDE,
        })

    originals, masks = s5.build_index(root)
    s5.print_index_summary(originals, masks)
    ids = sorted(originals["train"])
    assert ids, "No training images found - check the folder path."

    tr_ids, va_ids = train_test_split(ids, test_size=0.2, random_state=42)  # same split as v1
    print("Loading images into memory (takes a minute)...")
    tr_data = load_split(tr_ids, originals["train"], masks["train"])
    va_data = load_split(va_ids, originals["train"], masks["train"])
    print("Image size after resize (first train image):", tr_data[0][0].shape)
    print(f"train images: {len(tr_ids)} | val images: {len(va_ids)}")

    loader = DataLoader(CropDataset(tr_data, crops_per_epoch), batch_size=batch_size,
                        shuffle=False, num_workers=0)

    model = s5.build_model()
    dice_loss = smp.losses.DiceLoss(mode="multilabel", from_logits=True)
    focal = smp.losses.FocalLoss(mode="multilabel")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    thr_path = save_path.replace(".pth", "_thresholds.json")
    best_score = -1.0

    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                out = model(x)
            out = out.float()
            loss = dice_loss(out, y) + focal(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()

        msg = f"Epoch {epoch+1}/{epochs} | loss={total/n:.4f}"
        log = {"epoch": epoch + 1, "train_loss": total / n, "lr": sched.get_last_lr()[0]}

        if (epoch + 1) % eval_every == 0 or epoch + 1 == epochs:
            dice = evaluate_full(model, va_data)
            best, idx = dice.max(dim=0)
            score = best.mean().item()
            msg += f" | val Dice(best thr)={score:.4f} | " + " ".join(
                f"{c}:{best[i]:.2f}@{THRESHOLDS[idx[i]]}" for i, c in enumerate(CLASSES))
            log["val_mean_dice"] = score
            for i, c in enumerate(CLASSES):
                log[f"val_dice_{c}"] = best[i].item()
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), save_path)
                with open(thr_path, "w") as f:
                    json.dump({c: THRESHOLDS[idx[i]] for i, c in enumerate(CLASSES)}, f)
                msg += "  -> saved"
                log["saved_new_best"] = True
        print(msg)
        if use_wandb:
            wandb.log(log)

    if use_wandb:
        wandb.summary["best_val_mean_dice"] = best_score
        wandb.finish()
    print(f"Done. Best val Dice={best_score:.4f}. Model: {save_path} | thresholds: {thr_path}")
    return model


# ---------- Final test + inference ----------

def load_thresholds(model_path="/content/drive/MyDrive/p1/models/seg_model_v2.pth"):
    with open(model_path.replace(".pth", "_thresholds.json")) as f:
        return json.load(f)


def evaluate_test_v2(root="/content/drive/MyDrive/p1/A. Segmentation",
                     model_path="/content/drive/MyDrive/p1/models/seg_model_v2.pth",
                     min_dice=0.40):
    """Score on the official held-out IDRiD test images (run once, after training).
    A lesion type is marked reportable only if its test Dice reaches min_dice
    (a bar you choose and state in the README)."""
    originals, masks = s5.build_index(root)
    ids = sorted(originals["test"])
    data = load_split(ids, originals["test"], masks["test"])

    model = s5.build_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    thr = load_thresholds(model_path)
    dice = evaluate_full(model, data)

    results = {}
    for ci, c in enumerate(CLASSES):
        ti = int(np.argmin([abs(t - thr[c]) for t in THRESHOLDS]))
        d = dice[ti, ci].item()
        results[c] = d
        status = "reportable" if d >= min_dice else "NOT reportable"
        print(f"{c}: test Dice={d:.3f} (threshold {thr[c]}) -> {status}")
    return results


@torch.no_grad()
def predict_masks_v2(model, image_rgb, thresholds):
    """Drop-in replacement for stage5's predict_masks: {class: binary mask at original size}."""
    model.eval()
    h, w = image_rgb.shape[:2]
    scale = min(1.0, LONG_SIDE / max(h, w))
    img = image_rgb
    if scale < 1.0:
        img = cv2.resize(image_rgb, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    p = predict_probs(model, img).numpy()
    out = {}
    for i, c in enumerate(CLASSES):
        m = (p[i] > thresholds[c]).astype(np.uint8)
        out[c] = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) if m.shape != (h, w) else m
    return out
