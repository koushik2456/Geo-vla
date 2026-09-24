"""
training/analytics.py — Everything needed to *explain* a trained network.

Saved into <job>/analytics/ by the training scripts and shown in the studio:

  architecture.json     layer-by-layer output shapes and parameter counts
  filters.png           the 64 learned first-layer convolution kernels (7×7 RGB)
  feature_maps.png      activations of an early layer for one input image
  evaluation.json       confusion matrix, per-class precision/recall/F1, 2-D embedding
                        projection (PCA of penultimate features), threshold sweep
  predictions.png       test images with true vs predicted labels (✓ / ✗)
  training_curves.png   loss / accuracy / learning-rate curves (for slides)
"""
import json
import os

import numpy as np

DARK = "#0b0f17"
INK = "#cfe3ff"
MUTED = "#7f93b0"
C1, C2, C3 = "#3987e5", "#d95926", "#199e70"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    plt = _plt()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def write_json(out_dir: str, name: str, data: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(data, f, indent=1, default=float)


# -- architecture ----------------------------------------------------------------------------

def architecture_summary(model, inputs: tuple, modules: list, title: str) -> dict:
    """Run one forward pass with hooks on `modules` [(name, module)] and record shapes/params."""
    import torch

    shapes = {}
    hooks = []

    def shape_of(out):
        if isinstance(out, torch.Tensor):
            return list(out.shape)
        if isinstance(out, (list, tuple)):
            return [shape_of(o) for o in out if isinstance(o, torch.Tensor)]
        return None

    def recorder(name):
        def hook(_module, _inputs, output):  # must return None: a returned value replaces the output
            shapes.setdefault(name, shape_of(output))
        return hook

    for name, module in modules:
        hooks.append(module.register_forward_hook(recorder(name)))
    model.eval()
    with torch.no_grad():
        model(*inputs)
    for h in hooks:
        h.remove()
    rows = []
    for name, module in modules:
        params = sum(p.numel() for p in module.parameters())
        rows.append({"name": name, "type": type(module).__name__, "output_shape": shapes.get(name),
                     "params": int(params),
                     "trainable": int(sum(p.numel() for p in module.parameters() if p.requires_grad))})
    total = sum(p.numel() for p in model.parameters())
    return {"title": title, "input_shape": [list(x.shape) for x in inputs], "layers": rows,
            "total_params": int(total),
            "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "size_mb": round(total * 4 / 1e6, 1)}


# -- learned representations -------------------------------------------------------------------

def filters_png(weight, path: str, title: str = "First-layer convolution kernels") -> None:
    """weight: (out, 3, k, k) tensor → grid of RGB kernels, each min-max normalised."""
    plt = _plt()
    w = weight.detach().cpu().numpy()[:, :3]
    n = w.shape[0]
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.8, rows * 0.8 + 0.4), facecolor=DARK)
    for ax in axes.ravel():
        ax.axis("off")
    for k, ax in zip(range(n), axes.ravel()):
        f = w[k].transpose(1, 2, 0)
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        ax.imshow(f, interpolation="nearest")
    fig.suptitle(title, color=INK, fontsize=9)
    _save(fig, path)


def feature_maps_png(image: np.ndarray, activation, path: str, layer_name: str, n: int = 15) -> None:
    plt = _plt()
    act = activation.detach().cpu().numpy()
    order = np.argsort(-act.reshape(act.shape[0], -1).std(1))[:n]   # most active channels first
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.2), facecolor=DARK)
    for ax in axes.ravel():
        ax.axis("off")
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("input", color=INK, fontsize=7)
    for ax, ch in zip(axes.ravel()[1:], order):
        ax.imshow(act[ch], cmap="magma")
        ax.set_title(f"ch {ch}", color=MUTED, fontsize=6)
    fig.suptitle(f"Feature maps after {layer_name} ({act.shape[0]} channels, {act.shape[1]}×{act.shape[2]})",
                 color=INK, fontsize=9)
    _save(fig, path)


def pca_2d(features: np.ndarray, labels: np.ndarray, max_points: int = 1200) -> dict:
    """Project penultimate-layer embeddings to 2-D with PCA (SVD) for a scatter plot."""
    idx = np.random.default_rng(0).permutation(len(features))[:max_points]
    x = features[idx] - features[idx].mean(0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    pts = x @ vt[:2].T
    var = (s ** 2) / (s ** 2).sum()
    return {"points": np.round(pts, 3).tolist(), "labels": labels[idx].astype(int).tolist(),
            "explained_variance": np.round(var[:2], 4).tolist(), "dim": int(features.shape[1])}


# -- evaluation ----------------------------------------------------------------------------------

def classification_report(y_true: np.ndarray, y_pred: np.ndarray, classes: list) -> dict:
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i].sum() if cm[i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                        "f1": round(float(f1), 4), "support": int(cm[i].sum())}
    supported = [v for v in per_class.values() if v["support"]]
    return {"accuracy": round(float((y_true == y_pred).mean()), 4), "confusion_matrix": cm.tolist(),
            "classes": classes, "per_class": per_class,
            "macro_f1": round(float(np.mean([v["f1"] for v in supported])), 4) if supported else 0.0}


def predictions_png(images: list, y_true, y_pred, conf, classes, path: str) -> None:
    plt = _plt()
    n = len(images)
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.75), facecolor=DARK)
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, img, t, p, c in zip(np.atleast_1d(axes).ravel(), images, y_true, y_pred, conf):
        ax.imshow(img)
        ok = t == p
        ax.set_title(f"{'✓' if ok else '✗'} {classes[p][:12]}\n{c:.0%} · true {classes[t][:10]}",
                     fontsize=6, color="#57d98a" if ok else "#ff6b6b", pad=2)
    fig.suptitle("Test predictions (✓ correct, ✗ wrong)", color=INK, fontsize=9)
    _save(fig, path)


def change_predictions_png(samples: list, path: str) -> None:
    """samples: [(A, B, label, prob)] uint8/float arrays."""
    plt = _plt()
    fig, axes = plt.subplots(len(samples), 4, figsize=(8, 2.1 * len(samples)), facecolor=DARK)
    axes = np.atleast_2d(axes)
    for row, (a, b, y, prob) in zip(axes, samples):
        for ax, img, title in zip(row, [a, b, y, prob], ["Before", "After", "Ground truth", "Predicted probability"]):
            ax.imshow(img, cmap="magma" if img.ndim == 2 else None, vmin=0, vmax=1 if img.ndim == 2 else None)
            ax.set_title(title, color=INK, fontsize=7)
            ax.axis("off")
    _save(fig, path)


def threshold_sweep(probs: np.ndarray, labels: np.ndarray) -> list:
    out = []
    for t in np.round(np.arange(0.1, 0.95, 0.1), 2):
        pred = probs > t
        tp = float((pred & labels).sum())
        fp = float((pred & ~labels).sum())
        fn = float((~pred & labels).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out.append({"threshold": float(t), "precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
                    "iou": round(tp / (tp + fp + fn), 4) if tp + fp + fn else 0.0})
    return out


def curves_png(history: list, batches: list, path: str, metric_key: str, metric_label: str) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), facecolor=DARK)
    for ax in axes:
        ax.set_facecolor(DARK)
        ax.tick_params(colors=MUTED, labelsize=7)
        for s in ax.spines.values():
            s.set_color("#243044")
        ax.grid(color="#1a2433", lw=0.6)
    ep = [h["epoch"] for h in history]
    if batches:
        axes[0].plot([b["global_step"] for b in batches], [b["loss"] for b in batches], color=C1, lw=1, alpha=0.8)
    axes[0].set_title("Training loss per batch", color=INK, fontsize=9)
    axes[1].plot(ep, [h["train_loss"] for h in history], "o-", color=C1, lw=2, ms=4, label="train")
    if history and "val_loss" in history[0]:
        axes[1].plot(ep, [h["val_loss"] for h in history], "o-", color=C2, lw=2, ms=4, label="validation")
    axes[1].set_title("Loss per epoch", color=INK, fontsize=9)
    axes[1].legend(fontsize=7, facecolor=DARK, labelcolor=INK, frameon=False)
    if history and metric_key in history[0]:
        axes[2].plot(ep, [100 * h[metric_key] for h in history], "o-", color=C3, lw=2, ms=4, label=metric_label)
        if "train_acc" in history[0] and metric_key == "val_acc":
            axes[2].plot(ep, [100 * h["train_acc"] for h in history], "o-", color=C1, lw=2, ms=4, label="train accuracy")
    axes[2].set_title(f"{metric_label} (%)", color=INK, fontsize=9)
    from matplotlib.ticker import MaxNLocator
    for ax in axes[1:]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("epoch", color=MUTED, fontsize=7)
    axes[0].set_xlabel("step", color=MUTED, fontsize=7)
    axes[2].legend(fontsize=7, facecolor=DARK, labelcolor=INK, frameon=False)
    _save(fig, path)


def confusion_png(report: dict, path: str) -> None:
    plt = _plt()
    cm = np.array(report["confusion_matrix"], dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), facecolor=DARK)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    classes = report["classes"]
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=7, color=INK)
    ax.set_yticks(range(len(classes)), classes, fontsize=7, color=INK)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "#1c2430")
    ax.set_xlabel("Predicted", color=INK, fontsize=8)
    ax.set_ylabel("True", color=INK, fontsize=8)
    ax.set_title(f"Confusion matrix — accuracy {100 * report['accuracy']:.1f}%", color=INK, fontsize=9)
    _save(fig, path)

