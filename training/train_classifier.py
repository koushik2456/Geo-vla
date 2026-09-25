"""
training/train_classifier.py â€” Fine-tune ResNet-50 on EuroSAT (RGB, 10 land-cover classes).

    python -m training.train_classifier                               # full EuroSAT, 10 epochs (GPU)
    python -m training.train_classifier --max-samples 2000 --img-size 64 --epochs 5   # quick, CPU-friendly
    python -m training.train_classifier --dataset synthetic --epochs 3                # no download

EuroSAT (~90 MB) downloads automatically on first use (training/datasets.py).

Outputs (next to --out):
    resnet50_eurosat.pth            best-validation weights (state_dict)
    resnet50_eurosat.metrics.json   headline metrics + per-epoch history
    analytics/                      dataset statistics, architecture, confusion matrix,
                                    per-class metrics, filters, feature maps, embeddings,
                                    prediction gallery, training curves (see training/analytics.py)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import EUROSAT_CLASSES  # noqa: E402
from models.classifier import SceneClassifier  # noqa: E402
from training import analytics  # noqa: E402
from training.common import Progress, build_with_pretrained_fallback, read_batches  # noqa: E402
from training.datasets import describe_classification, download_eurosat, split_manifest, stratified_split  # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class ImageList(Dataset):
    """Images addressed by index through a loader function, with a torchvision transform."""

    def __init__(self, load, labels, indices, transform):
        self.load, self.labels, self.indices, self.transform = load, labels, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        return self.transform(Image.fromarray(self.load(i))), int(self.labels[i])


def load_source(args, progress):
    """Returns (load(i) -> HxWx3 uint8, labels array, source description, stable file ids)."""
    if args.dataset == "synthetic":
        from training.synthetic_data import SyntheticEuroSAT
        ds = SyntheticEuroSAT(per_class=args.samples_per_class, seed=args.seed)
        labels = np.array([i // ds.per_class for i in range(len(ds))])
        ids = [f"synthetic/{i:05d}" for i in range(len(ds))]
        return (lambda i: np.array(ds[i][0])), labels, "synthetic EuroSAT-like patches", ids
    root = args.image_folder or download_eurosat(log=lambda m: progress.emit("info", message=m))
    files, labels = [], []
    for c, name in enumerate(EUROSAT_CLASSES):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            raise SystemExit(f"missing class folder {folder}")
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                files.append(os.path.join(folder, f))
                labels.append(c)
    ids = [os.path.relpath(f, root).replace(os.sep, "/") for f in files]
    return (lambda i: np.array(Image.open(files[i]).convert("RGB"))), np.array(labels), f"EuroSAT RGB ({root})", ids


def transforms_for(size: int):
    train = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    evaluate = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return train, evaluate


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, collect=False):
    model.eval()
    preds, labels, probs, feats, total_loss = [], [], [], [], 0.0
    captured = {}
    hook = model.model.avgpool.register_forward_hook(lambda m, i, o: captured.__setitem__("f", o.flatten(1))) if collect else None
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * len(y)
        p = logits.softmax(1)
        preds.append(p.argmax(1).cpu())
        probs.append(p.max(1).values.cpu())
        labels.append(y.cpu())
        if collect:
            feats.append(captured["f"].cpu())
    if hook:
        hook.remove()
    out = {"pred": torch.cat(preds).numpy(), "true": torch.cat(labels).numpy(), "conf": torch.cat(probs).numpy()}
    out["acc"] = float((out["pred"] == out["true"]).mean())
    out["loss"] = total_loss / max(len(out["true"]), 1)
    if collect:
        out["features"] = torch.cat(feats).numpy()
    return out


def grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["eurosat", "synthetic"], default="eurosat")
    p.add_argument("--image-folder", default=None, help="Use an extracted EuroSAT RGB folder")
    p.add_argument("--out", default="models/checkpoints/resnet50_eurosat.pth")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img-size", type=int, default=224, help="224 = ImageNet size; 64 = EuroSAT native (fast on CPU)")
    p.add_argument("--max-samples", type=int, default=0, help="Stratified subsample of the dataset (0 = all)")
    p.add_argument("--samples-per-class", type=int, default=300, help="Synthetic dataset size")
    p.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2,  # Windows workers cannot pickle the loaders
                   help="DataLoader worker processes")
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

    # 1. Data. The test split is sealed: identified by a manifest + hash but never loaded here.
    load, labels, source, ids = load_source(args, progress)
    train_idx, val_idx, test_idx = stratified_split(labels, max_samples=args.max_samples or None, seed=args.seed)
    manifest = split_manifest(ids, test_idx)
    analytics.write_json(an_dir, "test_split.json", {**manifest, "seed": args.seed, "dataset": args.dataset,
                                                     "note": "sealed: evaluate only via training.confirm"})
    progress.emit("info", message=f"Dataset: {source}; {len(train_idx)} train / {len(val_idx)} val / "
                                  f"{len(test_idx)} test (sealed, sha256 {manifest['sha256'][:12]})")
    stats = describe_classification(load, labels, {"train": train_idx, "val": val_idx, "test": test_idx}, source,
                                    an_dir, pool=np.concatenate([train_idx, val_idx]))
    analytics.write_json(an_dir, "dataset.json", stats)
    progress.emit("dataset", **{k: stats[k] for k in ("total", "split_sizes", "channel_mean", "channel_std")})

    train_tf, eval_tf = transforms_for(args.img_size)
    mk = lambda idx, tf, shuffle: DataLoader(ImageList(load, labels, idx, tf), batch_size=args.batch_size,  # noqa: E731
                                             shuffle=shuffle, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_dl, val_dl = mk(train_idx, train_tf, True), mk(val_idx, eval_tf, False)

    # 2. Model
    model, pretrained = build_with_pretrained_fallback(
        lambda pre: SceneClassifier(num_classes=len(EUROSAT_CLASSES), pretrained=pre), not args.no_pretrained, progress)
    model = model.to(device)
    net = model.model
    arch = analytics.architecture_summary(
        model, (torch.zeros(1, 3, args.img_size, args.img_size, device=device),),
        [(n, getattr(net, n)) for n in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc")],
        "ResNet-50 scene classifier")
    arch["notes"] = {"stem": "7Ã—7 conv, stride 2 â†’ batch norm â†’ ReLU â†’ 3Ã—3 max-pool",
                     "stages": "4 stages of bottleneck blocks (1Ã—1 â†’ 3Ã—3 â†’ 1Ã—1 conv + skip connection): 3, 4, 6, 3 blocks",
                     "head": f"global average pooling â†’ fully connected 2048 â†’ {len(EUROSAT_CLASSES)} classes (softmax)",
                     "initialisation": "ImageNet" if pretrained else "random"}
    analytics.write_json(an_dir, "architecture.json", arch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_dl))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    hyper = {"optimizer": "AdamW", "weight_decay": 1e-4, "max_lr": args.lr, "schedule": "one-cycle (cosine)",
             "loss": "cross-entropy, label smoothing 0.05", "batch_size": args.batch_size, "epochs": args.epochs,
             "img_size": args.img_size, "augmentation": "flips, 90Â° rotation, colour jitter",
             "mixed_precision": device.type == "cuda"}
    progress.emit("start", model="classifier", dataset=args.dataset, epochs=args.epochs, steps_per_epoch=len(train_dl),
                  train_size=len(train_idx), val_size=len(val_idx), test_size=len(test_idx), test_sealed=True, device=str(device),
                  pretrained=pretrained, params=arch["total_params"], hyperparameters=hyper)

    # 3. Train
    best_acc, history, step = -1.0, [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total_loss, correct, seen = time.time(), 0.0, 0, 0
        for i, (x, y) in enumerate(tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}"), 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = grad_norm(model)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            hits = int((logits.argmax(1) == y).sum())
            total_loss += loss.item() * len(y)
            correct += hits
            seen += len(y)
            progress.batch(epoch=epoch, step=i, global_step=step, loss=loss.item(), batch_acc=hits / len(y),
                           lr=scheduler.get_last_lr()[0], grad_norm=gnorm)
        val = evaluate(model, val_dl, device, criterion)
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "train_acc": correct / seen,
                        "val_loss": val["loss"], "val_acc": val["acc"], "lr": scheduler.get_last_lr()[0],
                        "seconds": round(time.time() - start, 1)})
        progress.emit("epoch", **history[-1])
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            torch.save(model.state_dict(), args.out)

    # 4. Evaluate the best checkpoint on the VALIDATION split and explain it (the test split stays sealed)
    progress.emit("info", message="Evaluating the best checkpoint on the validation set and generating analytics")
    model.load_state_dict(torch.load(args.out, map_location=device))
    ev = evaluate(model, val_dl, device, criterion, collect=True)
    report = analytics.classification_report(ev["true"], ev["pred"], EUROSAT_CLASSES)
    report["embedding"] = analytics.pca_2d(ev["features"], ev["true"])
    report["split"] = "val"
    report["val_loss"] = ev["loss"]
    analytics.write_json(an_dir, "evaluation.json", report)
    analytics.confusion_png(report, os.path.join(an_dir, "confusion_matrix.png"))
    analytics.filters_png(net.conv1.weight, os.path.join(an_dir, "filters.png"),
                          "The 64 learned 7x7 kernels of the first convolution")

    sample = load(val_idx[0])
    captured = {}
    h = net.layer1.register_forward_hook(lambda m, i, o: captured.__setitem__("a", o[0]))
    model.eval()
    with torch.no_grad():
        model(eval_tf(Image.fromarray(sample)).unsqueeze(0).to(device))
    h.remove()
    analytics.feature_maps_png(sample, captured["a"], os.path.join(an_dir, "feature_maps.png"), "layer1")

    wrong = np.flatnonzero(ev["pred"] != ev["true"])[:8]
    right = np.flatnonzero(ev["pred"] == ev["true"])
    order = np.concatenate([wrong, right])[:16]
    analytics.predictions_png([load(val_idx[k]) for k in order], ev["true"][order], ev["pred"][order],
                              ev["conf"][order], EUROSAT_CLASSES, os.path.join(an_dir, "predictions.png"))
    analytics.curves_png(history, read_batches(args.progress_file), os.path.join(an_dir, "training_curves.png"),
                         "val_acc", "validation accuracy")

    metrics = {"dataset": source if args.dataset == "eurosat" else "synthetic EuroSAT-like",
               "evaluation_split": "val", "best_val_acc": best_acc, "val_acc": report["accuracy"],
               "val_macro_f1": report["macro_f1"],
               "per_class_val_recall": {c: v["recall"] for c, v in report["per_class"].items()},
               "test_split": {k: manifest[k] for k in ("count", "sha256")},
               "pretrained": pretrained, "params": arch["total_params"], "hyperparameters": hyper,
               "history": history, "args": vars(args)}
    with open(args.out.replace(".pth", ".metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    progress.emit("done", checkpoint=args.out, val_acc=report["accuracy"], val_macro_f1=report["macro_f1"],
                  best_val_acc=best_acc, per_class=metrics["per_class_val_recall"])


if __name__ == "__main__":
    main()
