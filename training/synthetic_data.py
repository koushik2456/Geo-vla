"""
training/synthetic_data.py — Procedural stand-ins for EuroSAT and LEVIR-CD.

They let the whole training pipeline (studio, progress charts, model registry,
promotion) be exercised without downloading gigabytes of data. Models trained
on them are only useful on the synthetic demo world — for real use, train on
EuroSAT / LEVIR-CD.
"""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import synthetic
from models import EUROSAT_CLASSES
from synthetic import _World

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _noise(rng, shape, scale):
    return rng.normal(0, scale, shape)


def eurosat_like_patch(cls: str, rng: np.random.Generator, size: int = 64) -> np.ndarray:
    """A 64×64 RGB uint8 patch with a class-typical colour and texture."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size, 3), dtype=np.float32)
    green = np.array([70, 110, 60])
    if cls == "Forest":
        img[:] = [25, 60, 30]
        img += _noise(rng, (size, size, 1), 12)
    elif cls == "HerbaceousVegetation":
        img[:] = [110, 140, 80]
        img += _noise(rng, (size, size, 1), 18)
    elif cls == "Pasture":
        img[:] = [130, 165, 95]
        img += _noise(rng, (size, size, 1), 5)
    elif cls == "AnnualCrop":
        theta = rng.uniform(0, np.pi)
        period = rng.uniform(6, 14)
        stripes = np.sin((xx * np.cos(theta) + yy * np.sin(theta)) / period * 2 * np.pi) > 0
        img[:] = np.where(stripes[..., None], [150, 130, 80], [90, 130, 60])
    elif cls == "PermanentCrop":
        img[:] = [120, 110, 75]
        step = rng.integers(6, 10)
        dots = ((xx % step) < 3) & ((yy % step) < 3)
        img[dots] = [50, 90, 40]
    elif cls == "Residential":
        img[:] = [150, 140, 130]
        step = rng.integers(7, 11)
        roofs = ((xx % step) < step - 3) & ((yy % step) < step - 3)
        img[roofs] = [185, 95, 80]
    elif cls == "Industrial":
        img[:] = [120, 120, 120]
        for _ in range(rng.integers(2, 5)):
            x0, y0 = rng.integers(0, size - 20, 2)
            w, h = rng.integers(14, 30, 2)
            img[y0:y0 + h, x0:x0 + w] = rng.integers(180, 235)
    elif cls == "Highway":
        img[:] = green
        theta = rng.uniform(0, np.pi)
        d = np.abs((xx - size / 2) * np.sin(theta) - (yy - size / 2) * np.cos(theta))
        img[d < rng.uniform(3, 6)] = [105, 105, 110]
    elif cls == "River":
        img[:] = green
        phase, amp = rng.uniform(0, 2 * np.pi), rng.uniform(4, 12)
        centre = size / 2 + amp * np.sin(yy / size * 2 * np.pi + phase)
        img[np.abs(xx - centre) < rng.uniform(4, 9)] = [40, 70, 110]
    elif cls == "SeaLake":
        img[:] = [20, 45, 80]
        img += _noise(rng, (size, size, 1), 4)
    img += _noise(rng, (size, size, 3), 6) + rng.uniform(-15, 15)   # sensor noise + illumination
    return np.clip(img, 0, 255).astype(np.uint8)


class SyntheticEuroSAT(Dataset):
    classes = EUROSAT_CLASSES

    def __init__(self, per_class: int = 300, transform=None, seed: int = 0):
        self.n = per_class * len(EUROSAT_CLASSES)
        self.per_class, self.transform, self.seed = per_class, transform, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        label = i // self.per_class
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        img = Image.fromarray(eurosat_like_patch(EUROSAT_CLASSES[label], rng))
        return (self.transform(img) if self.transform else img), label


class SyntheticChangePairs(Dataset):
    """Bi-temporal 256×256 pairs from the synthetic world; label = land cover changed."""

    def __init__(self, n: int = 400, size: int = 256, seed: int = 0, augment: bool = False):
        self.n, self.size, self.seed, self.augment = n, size, seed, augment

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        lon, lat = rng.uniform(70, 90), rng.uniform(8, 30)
        bbox = [lon, lat, lon + 0.03, lat + 0.03]
        dates = sorted(f"{rng.integers(2018, 2026)}-{rng.integers(1, 13):02d}-15" for _ in range(2))
        shape = (self.size, self.size)
        world = _World(bbox)
        c1, c2 = (world.land_cover(shape, synthetic._year(d)) for d in dates)
        label = (c1 != c2).astype(np.float32)
        imgs = []
        for d in dates:
            bands = synthetic.synthetic_sentinel2(bbox, d, shape)
            rgb = np.clip(np.stack([bands["B04"], bands["B03"], bands["B02"]], -1) * 2.5, 0, 1)
            imgs.append(rgb.astype(np.float32))
        a, b = imgs
        if self.augment and rng.random() < 0.5:
            a, b, label = a[:, ::-1], b[:, ::-1], label[:, ::-1]
        to_t = lambda x: torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy())  # noqa: E731
        return to_t(a), to_t(b), torch.from_numpy(label.copy())
