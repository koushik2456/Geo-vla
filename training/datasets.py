"""
training/datasets.py — Dataset download, loading, splitting and description.

EuroSAT  (land-cover classifier): downloaded automatically from the first mirror
         that works; any folder of 10 class sub-folders is also accepted.
LEVIR-CD (change detector): import a downloaded archive or folder
         (python -m training.download_data levir --from /path/LEVIR-CD.zip).
Synthetic stand-ins for both are always available (training/synthetic_data.py).

`describe_*` functions compute the dataset statistics shown in the training
studio and saved with every job: class balance, per-channel statistics,
histograms, per-class mean colour ("spectral signature"), and sample images.
"""
import io
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import EUROSAT_CLASSES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EUROSAT_DIR = os.path.join(ROOT, "data", "eurosat")
LEVIR_DIR = os.path.join(ROOT, "data", "levir_cd")

# Tried in order; override with EUROSAT_URL. Each archive contains one folder per class.
EUROSAT_MIRRORS = [u for u in [
    os.getenv("EUROSAT_URL"),
    "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1",
    "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
    "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip",
] if u]


# -- EuroSAT ------------------------------------------------------------------------------

def find_eurosat_root(base: str = EUROSAT_DIR):
    """Directory whose sub-folders are the 10 EuroSAT classes (searched recursively)."""
    if not os.path.isdir(base):
        return None
    for dirpath, dirnames, _ in os.walk(base):
        if set(EUROSAT_CLASSES) <= set(dirnames):
            return dirpath
    return None


def _download(url: str, dest: str, log=print) -> None:
    import requests
    from tqdm import tqdm
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Geo-VLA"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def download_eurosat(base: str = EUROSAT_DIR, log=print) -> str:
    """Download and extract EuroSAT RGB (~90 MB) unless it is already present."""
    root = find_eurosat_root(base)
    if root:
        log(f"EuroSAT already available at {root}")
        return root
    os.makedirs(base, exist_ok=True)
    errors = []
    for url in EUROSAT_MIRRORS:
        log(f"Downloading EuroSAT from {url.split('?')[0]} …")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                archive = os.path.join(tmp, "eurosat.zip")
                _download(url, archive, log)
                with zipfile.ZipFile(archive) as z:
                    z.extractall(base)
            root = find_eurosat_root(base)
            if root:
                log(f"EuroSAT ready at {root}")
                return root
            errors.append(f"{url}: archive did not contain the 10 class folders")
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            log(f"  failed ({type(exc).__name__}); trying the next mirror")
    raise RuntimeError("Could not download EuroSAT from any mirror:\n  " + "\n  ".join(errors) +
                       "\nDownload EuroSAT_RGB.zip manually and extract it into data/eurosat/.")


def stratified_split(labels, val_frac=0.1, test_frac=0.1, max_samples=None, seed=42):
    """Per-class shuffled split so every class appears in train/val/test in proportion."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    classes = np.unique(labels)
    per_class_cap = None if not max_samples else max(3, max_samples // len(classes))
    for c in classes:
        idx = rng.permutation(np.flatnonzero(labels == c))
        if per_class_cap:
            idx = idx[:per_class_cap]
        n_val, n_test = max(1, int(len(idx) * val_frac)), max(1, int(len(idx) * test_frac))
        test.extend(idx[:n_test])
        val.extend(idx[n_test:n_test + n_val])
        train.extend(idx[n_test + n_val:])
    return [rng.permutation(np.array(s)) for s in (train, val, test)]


# -- LEVIR-CD ---------------------------------------------------------------------------------

def levir_ready(base: str = LEVIR_DIR) -> bool:
    return all(os.path.isdir(os.path.join(base, split, sub)) and os.listdir(os.path.join(base, split, sub))
               for split in ("train", "val", "test") for sub in ("A", "B", "label"))


def import_levir(source: str, base: str = LEVIR_DIR, log=print) -> str:
    """Import LEVIR-CD from a .zip or a folder into data/levir_cd/{train,val,test}/{A,B,label}.
    Accepts the official layout (train/val/test each with A, B, label) at any depth."""
    if levir_ready(base):
        log(f"LEVIR-CD already available at {base}")
        return base
    with tempfile.TemporaryDirectory() as tmp:
        src = source
        if source.endswith(".zip"):
            log(f"Extracting {source} …")
            with zipfile.ZipFile(source) as z:
                z.extractall(tmp)
            src = tmp
        found = {}
        for dirpath, dirnames, _ in os.walk(src):
            if {"A", "B", "label"} <= set(dirnames):
                split = os.path.basename(dirpath).lower()
                split = {"training": "train", "validation": "val", "testing": "test"}.get(split, split)
                if split in ("train", "val", "test"):
                    found[split] = dirpath
        missing = {"train", "val", "test"} - set(found)
        if missing:
            raise RuntimeError(f"could not find LEVIR-CD splits {sorted(missing)} (need <split>/A, B, label)")
        for split, path in found.items():
            for sub in ("A", "B", "label"):
                dest = os.path.join(base, split, sub)
                os.makedirs(dest, exist_ok=True)
                for name in os.listdir(os.path.join(path, sub)):
                    shutil.copy2(os.path.join(path, sub, name), os.path.join(dest, name))
    log(f"LEVIR-CD ready at {base}")
    return base


# -- dataset description (studio "Dataset explorer") -----------------------------------------------

def _grid_png(tiles: list, labels: list, cols: int, scale: int = 2) -> bytes:
    """Sample grid with a caption under each tile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = int(np.ceil(len(tiles) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25 * scale / 2, rows * 1.45 * scale / 2), facecolor="#0b0f17")
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, tile, label in zip(np.atleast_1d(axes).ravel(), tiles, labels):
        ax.imshow(tile, interpolation="nearest")
        ax.set_title(label, fontsize=6.5, color="#cfe3ff", pad=2)
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def describe_classification(images, labels, splits: dict, source: str, out_dir: str = None, max_stats: int = 1500) -> dict:
    """images: callable(i) -> HxWx3 uint8; labels: array of class indices."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(labels))[:max_stats]
    sample = np.stack([images(i) for i in idx]).astype(np.float32) / 255.0      # N,H,W,3
    lab = labels[idx]
    mean, std = sample.mean(axis=(0, 1, 2)), sample.std(axis=(0, 1, 2))
    hist = {ch: np.histogram(sample[..., k], bins=32, range=(0, 1))[0].tolist() for k, ch in enumerate("RGB")}
    per_class = {}
    for c, name in enumerate(EUROSAT_CLASSES):
        m = lab == c
        if m.any():
            px = sample[m]
            per_class[name] = {"mean_rgb": [round(float(v), 4) for v in px.mean(axis=(0, 1, 2))],
                               "std_rgb": [round(float(v), 4) for v in px.std(axis=(0, 1, 2))],
                               "brightness": round(float(px.mean()), 4),
                               "greenness_index": round(float(((px[..., 1] - px[..., 0]) / (px[..., 1] + px[..., 0] + 1e-6)).mean()), 4)}
    counts = {split: {name: int((labels[ix] == c).sum()) for c, name in enumerate(EUROSAT_CLASSES)}
              for split, ix in splits.items()}
    stats = {
        "task": "scene classification", "source": source, "classes": EUROSAT_CLASSES,
        "total": int(len(labels)), "image_shape": list(images(0).shape),
        "split_sizes": {k: int(len(v)) for k, v in splits.items()}, "class_counts": counts,
        "channel_mean": [round(float(v), 4) for v in mean], "channel_std": [round(float(v), 4) for v in std],
        "histograms": {"bins": np.linspace(0, 1, 33).round(4).tolist(), **hist},
        "per_class": per_class, "stats_sample_size": int(len(idx)),
        "normalisation": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "note": "ImageNet statistics"},
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tiles, caps = [], []
        for c, name in enumerate(EUROSAT_CLASSES):
            for i in np.flatnonzero(labels == c)[:6]:
                tiles.append(images(i))
                caps.append(name if len(caps) % 6 == 0 else "")
        with open(os.path.join(out_dir, "dataset_samples.png"), "wb") as f:
            f.write(_grid_png(tiles, caps, cols=6))
        stats["sample_image"] = "dataset_samples.png"
    return stats


def describe_change(pairs, n: int, splits: dict, source: str, out_dir: str = None, max_stats: int = 200) -> dict:
    """pairs: callable(i) -> (A uint8 HxWx3, B uint8, label bool HxW)."""
    idx = np.random.default_rng(0).permutation(n)[:max_stats]
    ratios, a_means, b_means = [], [], []
    for i in idx:
        a, b, y = pairs(i)
        ratios.append(float(y.mean()))
        a_means.append(a.reshape(-1, 3).mean(0) / 255)
        b_means.append(b.reshape(-1, 3).mean(0) / 255)
    ratios = np.array(ratios)
    a0, b0, y0 = pairs(0)
    stats = {
        "task": "binary change detection", "source": source, "total": int(n), "image_shape": list(a0.shape),
        "split_sizes": {k: int(len(v)) if hasattr(v, "__len__") else int(v) for k, v in splits.items()},
        "change_pixel_ratio": {"mean": round(float(ratios.mean()), 4), "median": round(float(np.median(ratios)), 4),
                               "max": round(float(ratios.max()), 4),
                               "pairs_without_change": round(float((ratios < 1e-4).mean()), 4)},
        "change_ratio_histogram": {"bins": np.linspace(0, max(ratios.max(), 1e-3), 21).round(4).tolist(),
                                   "counts": np.histogram(ratios, bins=20, range=(0, max(ratios.max(), 1e-3)))[0].tolist()},
        "class_balance": {"unchanged": round(float(1 - ratios.mean()), 4), "changed": round(float(ratios.mean()), 4)},
        "channel_mean_t1": [round(float(v), 4) for v in np.mean(a_means, 0)],
        "channel_mean_t2": [round(float(v), 4) for v in np.mean(b_means, 0)],
        "stats_sample_size": int(len(idx)),
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tiles, caps = [], []
        for i in idx[:4]:
            a, b, y = pairs(i)
            tiles += [a, b, np.stack([y * 255] * 3, -1).astype(np.uint8)]
            caps += ["Before (T1)", "After (T2)", f"Change {100 * y.mean():.1f}%"]
        with open(os.path.join(out_dir, "dataset_samples.png"), "wb") as f:
            f.write(_grid_png(tiles, caps, cols=3, scale=3))
        stats["sample_image"] = "dataset_samples.png"
    return stats
