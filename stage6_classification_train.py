"""
Stage 6: DR Severity Classification
- EfficientNet-B0 (pretrained), fine-tuned on APTOS
- Handles class imbalance with weighted loss
- Saves best checkpoint to Drive
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image


LABEL_MAP = {
    "no_retinopathy": 0,
    "mild_retinopathy": 1,
    "moderate_retinopathy": 2,
    "severe_retinopathy": 3,
    "proliferative_retinopathy": 4,
}
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Dataset ----------

class APTOSDataset(Dataset):
    def __init__(self, hf_dataset, indices, transform=None):
        self.ds = hf_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.ds[real_idx]
        image = np.array(item["image"].convert("RGB"))
        label = LABEL_MAP[item["label"]] if isinstance(item["label"], str) else item["label"]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


def get_transforms():
    train_tf = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_tf, val_tf


# ---------- Model ----------

def build_model():
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
    return model.to(DEVICE)


# ---------- Training ----------

def train_model(ds, epochs=10, batch_size=32, lr=1e-4, save_path="/content/drive/MyDrive/p1/models/best_model.pth"):
    all_indices = list(range(len(ds["train"])))
    all_labels = [LABEL_MAP[l] if isinstance(l, str) else l for l in ds["train"]["label"]]

    train_idx, val_idx = train_test_split(
        all_indices, test_size=0.2, random_state=42, stratify=all_labels
    )

    train_tf, val_tf = get_transforms()
    train_dataset = APTOSDataset(ds["train"], train_idx, transform=train_tf)
    val_dataset = APTOSDataset(ds["train"], val_idx, transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Handle class imbalance
    train_labels = [all_labels[i] for i in train_idx]
    class_weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_loss /= len(val_dataset)
        val_acc = correct / len(val_dataset)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved new best model (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val_acc={best_val_acc:.4f}. Model saved to {save_path}")
    return model


if __name__ == "__main__":
    from datasets import load_from_disk
    ds = load_from_disk("/content/drive/MyDrive/p1/data/aptos_hf")
    train_model(ds, epochs=10, batch_size=32, lr=1e-4)
