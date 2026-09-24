"""
training/train_change_detector.py — Fine-tune the Siamese U-Net on LEVIR-CD.

Usage (from the repo root):
    python -m training.train_change_detector --epochs 50 --batch-size 8

Data: download LEVIR-CD (https://justchenhao.github.io/LEVIR/) and extract so
that the layout is

    data/levir_cd/{train,val,test}/{A,B,label}/*.png

Each 1024×1024 pair is cut into sixteen 256×256 patches (637 pairs → 10,192
patches across the three splits), matching the setup used by STANet.

Outputs:
    models/checkpoints/siamese_unet_levircd.pth           (best val F1, state_dict)
    models/checkpoints/siamese_unet_levircd.metrics.json  (P / R / F1 / IoU / OA on test)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.change_detector import SiameseChangeDetector  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LevirCDPatches(Dataset):
    def __init__(self, root: str, split: str, patch: int = 256, augment: bool = False):
        self.dir = os.path.join(root, split)
        names = sorted(os.listdir(os.path.join(self.dir, "A")))
        if not names:
            raise SystemExit(f"no images in {self.dir}/A — see the docstring for the expected layout")
        self.patch, self.augment = patch, augment
        per_side = 1024 // patch
        self.items = [(n, r, c) for n in names for r in range(per_side) for c in range(per_side)]
        self._cache = {}

    def __len__(self):
        return len(self.items)

    def _load(self, name):
        if name not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            read = lambda sub: np.array(Image.open(os.path.join(self.dir, sub, name)))  # noqa: E731
            self._cache[name] = (read("A")[..., :3], read("B")[..., :3], read("label"))
        return self._cache[name]

    def __getitem__(self, i):
        name, r, c = self.items[i]
        a, b, lbl = self._load(name)
        sl = np.s_[r * self.patch:(r + 1) * self.patch, c * self.patch:(c + 1) * self.patch]
        a, b = a[sl].astype(np.float32) / 255.0, b[sl].astype(np.float32) / 255.0
        y = (lbl[sl] > 127).astype(np.float32) if lbl.ndim == 2 else (lbl[sl][..., 0] > 127).astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                a, b, y = a[:, ::-1], b[:, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5:
                a, b, y = a[::-1], b[::-1], y[::-1]
            k = np.random.randint(4)
            a, b, y = np.rot90(a, k), np.rot90(b, k), np.rot90(y, k)
            if np.random.rand() < 0.5:          # temporal swap: change is symmetric
                a, b = b, a
        to_t = lambda x: torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy())  # noqa: E731
        return to_t(a), to_t(b), torch.from_numpy(y.copy())


def dice_loss(logits, target, eps=1.0):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum((1, 2))
    return 1 - ((2 * inter + eps) / (prob.sum((1, 2)) + target.sum((1, 2)) + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = tn = 0
    for a, b, y in loader:
        pred = torch.sigmoid(model(a.to(device), b.to(device))) > 0.5
        y = y.to(device).bool()
        tp += (pred & y).sum().item()
        fp += (pred & ~y).sum().item()
        fn += (~pred & y).sum().item()
        tn += (~pred & ~y).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "iou": tp / max(tp + fp + fn, 1), "overall_accuracy": (tp + tn) / max(tp + fp + fn + tn, 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/levir_cd")
    p.add_argument("--out", default="models/checkpoints/siamese_unet_levircd.pth")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true", help="Skip ImageNet init (smoke tests only)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = LevirCDPatches(args.data_dir, "train", augment=True)
    val_ds, test_ds = LevirCDPatches(args.data_dir, "val"), LevirCDPatches(args.data_dir, "test")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} patches, device={device}")
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,  # noqa: E731
                                            num_workers=args.workers, pin_memory=device.type == "cuda")
    train_dl, val_dl, test_dl = loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)

    model = SiameseChangeDetector(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Change pixels are ~5% of LEVIR-CD, so weight positives and add Dice.
    pos_weight = torch.tensor(5.0, device=device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_f1, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total = time.time(), 0.0
        for a, b, y in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}"):
            a, b, y = a.to(device), b.to(device), y.to(device)
            logits = model(a, b)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight) + dice_loss(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        scheduler.step()
        val = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "train_loss": total / len(train_ds), **{f"val_{k}": v for k, v in val.items()},
                        "seconds": round(time.time() - start, 1)})
        print(history[-1])
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save(model.state_dict(), args.out)

    model.load_state_dict(torch.load(args.out, map_location=device))
    test = evaluate(model, test_dl, device)
    with open(args.out.replace(".pth", ".metrics.json"), "w") as f:
        json.dump({"dataset": "LEVIR-CD (256px patches)", "best_val_f1": best_f1, "test": test,
                   "history": history, "args": vars(args)}, f, indent=2)
    print(f"test: {test} — saved {args.out}")


if __name__ == "__main__":
    main()
