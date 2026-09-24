"""
training/train_classifier.py — Fine-tune ResNet-50 on EuroSAT (RGB, 10 classes).

Usage (from the repo root):
    python -m training.train_classifier --epochs 10 --batch-size 64

Data: torchvision downloads EuroSAT RGB into data/eurosat/ automatically. If
that mirror is down, download EuroSAT_RGB.zip manually and pass
--image-folder path/to/2750 (one sub-folder per class).

Outputs:
    models/checkpoints/resnet50_eurosat.pth   (best val accuracy, state_dict)
    models/checkpoints/resnet50_eurosat.metrics.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import EUROSAT_CLASSES  # noqa: E402
from models.classifier import SceneClassifier  # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_datasets(args):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def make(tf):
        if args.image_folder:
            return datasets.ImageFolder(args.image_folder, transform=tf)
        return datasets.EuroSAT(args.data_dir, transform=tf, download=True)

    full_train, full_eval = make(train_tf), make(eval_tf)
    classes = full_train.classes
    if classes != EUROSAT_CLASSES:
        raise SystemExit(f"class order mismatch: {classes} != {EUROSAT_CLASSES}")

    # Stratification-free 80/10/10 split with a fixed seed (reproducible).
    idx = np.random.default_rng(args.seed).permutation(len(full_train))
    n_val = n_test = len(idx) // 10
    test_idx, val_idx, train_idx = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    return Subset(full_train, train_idx), Subset(full_eval, val_idx), Subset(full_eval, test_idx)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    per_class = {c: float((preds[labels == i] == i).mean()) for i, c in enumerate(EUROSAT_CLASSES) if (labels == i).any()}
    return float((preds == labels).mean()), per_class


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/eurosat")
    p.add_argument("--image-folder", default=None, help="Use an extracted EuroSAT RGB folder instead of downloading")
    p.add_argument("--out", default="models/checkpoints/resnet50_eurosat.pth")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true", help="Skip ImageNet init (smoke tests only)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds = build_datasets(args)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} device={device}")
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,  # noqa: E731
                                            num_workers=args.workers, pin_memory=device.type == "cuda")
    train_dl, val_dl, test_dl = loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)

    model = SceneClassifier(num_classes=len(EUROSAT_CLASSES), pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_dl))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_acc, history = 0.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total_loss = time.time(), 0.0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * len(x)
        val_acc, _ = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_ds), "val_acc": val_acc,
                        "seconds": round(time.time() - start, 1)})
        print(history[-1])
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.out)

    model.load_state_dict(torch.load(args.out, map_location=device))
    test_acc, per_class = evaluate(model, test_dl, device)
    metrics = {"dataset": "EuroSAT RGB", "best_val_acc": best_acc, "test_acc": test_acc,
               "per_class_test_acc": per_class, "history": history, "args": vars(args)}
    with open(args.out.replace(".pth", ".metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"test accuracy {test_acc:.4f} — saved {args.out}")


if __name__ == "__main__":
    main()
