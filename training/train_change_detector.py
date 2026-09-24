"""
training/train_change_detector.py — Fine-tune the Siamese U-Net on LEVIR-CD.

    python -m training.train_change_detector --epochs 50 --batch-size 8          # LEVIR-CD (GPU)
    python -m training.train_change_detector --max-samples 400 --epochs 5        # quick subset
    python -m training.train_change_detector --dataset synthetic --epochs 3      # no download

Data: import LEVIR-CD once with
    python -m training.download_data levir --from /path/to/LEVIR-CD.zip
which produces data/levir_cd/{train,val,test}/{A,B,label}. Each 1024×1024 pair
is cut into sixteen 256×256 patches (637 pairs → 10,192 patches), as in STANet.

Outputs (next to --out):
    siamese_unet_levircd.pth            best-validation-F1 weights (state_dict)
    siamese_unet_levircd.metrics.json   P / R / F1 / IoU / OA on test + history
    analytics/                          dataset statistics, architecture, threshold sweep,
                                        prediction panels, training curves
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
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.change_detector import SiameseChangeDetector  # noqa: E402
from training import analytics  # noqa: E402
from training.common import Progress, build_with_pretrained_fallback, read_batches  # noqa: E402
from training.datasets import LEVIR_DIR, describe_change, levir_ready  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LevirCDPatches(Dataset):
    def __init__(self, root: str, split: str, patch: int = 256, augment: bool = False):
        self.dir = os.path.join(root, split)
        names = sorted(os.listdir(os.path.join(self.dir, "A")))
        if not names:
            raise SystemExit(f"no images in {self.dir}/A — run: python -m training.download_data levir --from …")
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

    def raw(self, i):
        name, r, c = self.items[i]
        a, b, lbl = self._load(name)
        sl = np.s_[r * self.patch:(r + 1) * self.patch, c * self.patch:(c + 1) * self.patch]
        y = lbl[sl] if lbl.ndim == 2 else lbl[sl][..., 0]
        return a[sl], b[sl], y > 127

    def __getitem__(self, i):
        a, b, y = self.raw(i)
        a, b, y = a.astype(np.float32) / 255.0, b.astype(np.float32) / 255.0, y.astype(np.float32)
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


POS_WEIGHT = 5.0   # change pixels are ~5% of LEVIR-CD


def loss_fn(logits, y):
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(POS_WEIGHT, device=y.device)) + dice_loss(logits, y)


@torch.no_grad()
def evaluate(model, loader, device, keep_probs=False):
    model.eval()
    tp = fp = fn = tn = 0
    total_loss, n = 0.0, 0
    probs, labels = [], []
    for a, b, y in loader:
        a, b, y = a.to(device), b.to(device), y.to(device)
        logits = model(a, b)
        total_loss += loss_fn(logits, y).item() * len(y)
        n += len(y)
        prob = torch.sigmoid(logits)
        pred, yb = prob > 0.5, y.bool()
        tp += (pred & yb).sum().item()
        fp += (pred & ~yb).sum().item()
        fn += (~pred & yb).sum().item()
        tn += (~pred & ~yb).sum().item()
        if keep_probs:
            probs.append(prob[:, ::4, ::4].cpu().numpy().ravel())    # subsample pixels for the sweep
            labels.append(yb[:, ::4, ::4].cpu().numpy().ravel())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    out = {"precision": precision, "recall": recall,
           "f1": 2 * precision * recall / max(precision + recall, 1e-9),
           "iou": tp / max(tp + fp + fn, 1), "overall_accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
           "loss": total_loss / max(n, 1), "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}
    if keep_probs:
        out["probs"], out["labels"] = np.concatenate(probs), np.concatenate(labels)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["levir", "synthetic"], default="levir")
    p.add_argument("--data-dir", default=LEVIR_DIR)
    p.add_argument("--out", default="models/checkpoints/siamese_unet_levircd.pth")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-samples", type=int, default=0, help="Use at most this many training patches (0 = all)")
    p.add_argument("--synthetic-pairs", type=int, default=400, help="Synthetic training pairs")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true", help="Random initialisation instead of ImageNet")
    p.add_argument("--progress-file", default=None, help="JSON-lines progress for the training studio")
    args = p.parse_args()
    progress = Progress(args.progress_file)
    try:
        run(args, progress)
    except Exception as exc:
        progress.emit("error", message=f"{type(exc).__name__}: {exc}")
        raise


def run(args, progress):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    an_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "analytics")
    os.makedirs(an_dir, exist_ok=True)

    # 1. Data
    if args.dataset == "synthetic":
        from training.synthetic_data import SyntheticChangePairs
        n = args.synthetic_pairs
        train_ds = SyntheticChangePairs(n, seed=args.seed, augment=True)
        val_ds = SyntheticChangePairs(max(n // 8, 8), seed=args.seed + 1)
        test_ds = SyntheticChangePairs(max(n // 8, 8), seed=args.seed + 2)
        source = "synthetic change pairs"
    else:
        if not levir_ready(args.data_dir):
            raise SystemExit("LEVIR-CD not found — run: python -m training.download_data levir --from /path/to/LEVIR-CD.zip")
        train_ds = LevirCDPatches(args.data_dir, "train", augment=True)
        val_ds, test_ds = LevirCDPatches(args.data_dir, "val"), LevirCDPatches(args.data_dir, "test")
        source = f"LEVIR-CD ({args.data_dir})"
    base_train = train_ds
    if args.max_samples and len(train_ds) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        train_ds = Subset(train_ds, rng.permutation(len(train_ds))[:args.max_samples].tolist())
        val_ds = Subset(val_ds, rng.permutation(len(val_ds))[:max(args.max_samples // 8, 8)].tolist())
        test_ds = Subset(test_ds, rng.permutation(len(test_ds))[:max(args.max_samples // 8, 8)].tolist())
    raw = base_train.raw
    stats = describe_change(raw, len(base_train), {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
                            source, an_dir)
    analytics.write_json(an_dir, "dataset.json", stats)
    progress.emit("info", message=f"Dataset: {source}; {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test pairs")

    loader = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,  # noqa: E731
                                            num_workers=args.workers, pin_memory=device.type == "cuda")
    train_dl, val_dl, test_dl = loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)

    # 2. Model
    model, pretrained = build_with_pretrained_fallback(
        lambda pre: SiameseChangeDetector(pretrained=pre), not args.no_pretrained, progress)
    model = model.to(device)
    enc, dec = model.unet.encoder, model.unet.decoder
    modules = [(f"encoder.{n}", getattr(enc, n)) for n in ("conv1", "layer1", "layer2", "layer3", "layer4")]
    modules += [(f"decoder.block{i}", blk) for i, blk in enumerate(dec.blocks)]
    modules += [("segmentation_head", model.unet.segmentation_head)]
    x = torch.zeros(1, 3, 256, 256, device=device)
    arch = analytics.architecture_summary(model, (x, x), modules, "Siamese U-Net change detector")
    arch["notes"] = {"encoder": "shared-weight ResNet-34 applied to both dates (Siamese)",
                     "fusion": "absolute difference |f(T1) − f(T2)| at every encoder scale",
                     "decoder": "U-Net decoder with skip connections from the difference features",
                     "head": "1-channel logit per pixel → sigmoid → change probability",
                     "initialisation": "ImageNet" if pretrained else "random"}
    analytics.write_json(an_dir, "architecture.json", arch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    hyper = {"optimizer": "AdamW", "weight_decay": 1e-4, "lr": args.lr, "schedule": "cosine annealing",
             "loss": f"weighted BCE (pos_weight {POS_WEIGHT}) + Dice", "batch_size": args.batch_size,
             "epochs": args.epochs, "patch": "256×256", "augmentation": "flips, 90° rotations, temporal swap"}
    progress.emit("start", model="change_detector", dataset=args.dataset, epochs=args.epochs,
                  steps_per_epoch=len(train_dl), train_size=len(train_ds), val_size=len(val_ds),
                  test_size=len(test_ds), device=str(device), pretrained=pretrained,
                  params=arch["total_params"], hyperparameters=hyper)

    # 3. Train
    best_f1, history, step = -1.0, [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total = time.time(), 0.0
        for i, (a, b, y) in enumerate(tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}"), 1):
            a, b, y = a.to(device), b.to(device), y.to(device)
            logits = model(a, b)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))
            optimizer.step()
            step += 1
            total += loss.item() * len(y)
            progress.batch(epoch=epoch, step=i, global_step=step, loss=loss.item(),
                           lr=optimizer.param_groups[0]["lr"], grad_norm=gnorm)
        scheduler.step()
        val = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "train_loss": total / len(train_ds), "val_loss": val["loss"],
                        **{f"val_{k}": v for k, v in val.items() if k not in ("loss", "confusion")},
                        "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - start, 1)})
        progress.emit("epoch", **history[-1])
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save(model.state_dict(), args.out)

    # 4. Evaluate + explain
    progress.emit("info", message="Evaluating the best checkpoint on the test set and generating analytics")
    model.load_state_dict(torch.load(args.out, map_location=device))
    test = evaluate(model, test_dl, device, keep_probs=True)
    sweep = analytics.threshold_sweep(test.pop("probs"), test.pop("labels"))
    evaluation = {**{k: v for k, v in test.items()}, "threshold_sweep": sweep,
                  "best_threshold": max(sweep, key=lambda r: r["f1"])}
    analytics.write_json(an_dir, "evaluation.json", evaluation)
    analytics.filters_png(enc.conv1.weight, os.path.join(an_dir, "filters.png"),
                          "The 64 learned 7×7 kernels of the shared encoder's first convolution")

    base_test = test_ds.dataset if isinstance(test_ds, Subset) else test_ds
    samples = []
    model.eval()
    with torch.no_grad():
        for i in range(min(4, len(base_test))):
            a_raw, b_raw, y_raw = base_test.raw(i)
            ta, tb, _ = base_test[i]
            prob = torch.sigmoid(model(ta[None].to(device), tb[None].to(device)))[0].cpu().numpy()
            samples.append((a_raw, b_raw, y_raw.astype(float), prob))
        captured = {}
        h = enc.layer1.register_forward_hook(lambda m, i, o: captured.__setitem__("a", o[0]))
        ta, tb, _ = base_test[0]
        model(ta[None].to(device), tb[None].to(device))
        h.remove()
    analytics.change_predictions_png(samples, os.path.join(an_dir, "predictions.png"))
    analytics.feature_maps_png(samples[0][1], captured["a"], os.path.join(an_dir, "feature_maps.png"), "encoder.layer1")
    analytics.curves_png(history, read_batches(args.progress_file), os.path.join(an_dir, "training_curves.png"),
                         "val_f1", "validation F1")

    metrics = {"dataset": "LEVIR-CD (256px patches)" if args.dataset == "levir" else "synthetic change pairs",
               "best_val_f1": best_f1, "test": {k: v for k, v in test.items() if k != "confusion"},
               "pretrained": pretrained, "params": arch["total_params"], "hyperparameters": hyper,
               "history": history, "args": vars(args)}
    with open(args.out.replace(".pth", ".metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    progress.emit("done", checkpoint=args.out, test=metrics["test"], best_val_f1=best_f1)


if __name__ == "__main__":
    main()
