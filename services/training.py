"""
services/training.py — Training studio: jobs, live progress and the model registry.

Jobs run the training scripts as subprocesses (so a crash or out-of-memory
never takes the API down) and write JSON-lines progress that the studio polls.
A finished job registers a new model *version* with its metrics; an admin
compares versions and promotes one, which copies it into the active
checkpoint path and hot-reloads it — no restart needed. Checkpoints trained
elsewhere (e.g. the Colab notebook) can be uploaded as versions too.
"""
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading

import config
import geotools
from services import db

log = logging.getLogger("geo-vla.training")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "classifier": {"title": "Land-cover classifier (ResNet-50)", "script": "training.train_classifier",
                   "checkpoint": geotools.CHECKPOINT_FILES["classifier"], "score": ("test_acc", "Test accuracy"),
                   "datasets": ["eurosat", "synthetic"]},
    "change_detector": {"title": "Change detector (Siamese U-Net)", "script": "training.train_change_detector",
                        "checkpoint": geotools.CHECKPOINT_FILES["change"], "score": ("test_f1", "Test F1"),
                        "datasets": ["levir", "synthetic"]},
}
_GEOTOOLS_KEY = {"classifier": "classifier", "change_detector": "change"}
_procs: dict = {}
_threads: dict = {}
_lock = threading.Lock()
_downloads: dict = {}   # dataset id -> {"status", "message"}


# -- datasets ------------------------------------------------------------------------------

def datasets() -> list:
    from training.datasets import find_eurosat_root, levir_ready
    has_eurosat = find_eurosat_root() is not None
    items = [
        {"id": "eurosat", "model": "classifier", "title": "EuroSAT RGB (27,000 patches, 10 classes)",
         "available": has_eurosat, "auto_download": True,
         "how": "Downloaded automatically on first training run (needs internet), or extract EuroSAT_RGB into data/eurosat/."},
        {"id": "levir", "model": "change_detector", "title": "LEVIR-CD (637 image pairs → 10,192 patches)",
         "available": levir_ready(), "auto_download": False,
         "how": "Download from justchenhao.github.io/LEVIR and extract to data/levir_cd/{train,val,test}/{A,B,label}."},
        {"id": "synthetic", "model": "both", "title": "Synthetic demo data (generated on the fly)",
         "available": True, "auto_download": False,
         "how": "Always available. Exercises the full pipeline; models only suit the offline demo world."},
    ]
    for d in items:
        d["download"] = _downloads.get(d["id"])
    return items


def start_download(dataset_id: str) -> dict:
    """Download EuroSAT in the background (LEVIR-CD must be imported manually)."""
    if dataset_id != "eurosat":
        raise ValueError("only EuroSAT can be downloaded automatically; import LEVIR-CD with "
                         "python -m training.download_data levir --from <zip>")
    if _downloads.get(dataset_id, {}).get("status") == "running":
        return _downloads[dataset_id]
    state = _downloads[dataset_id] = {"status": "running", "message": "starting"}

    def work():
        from training.datasets import download_eurosat
        try:
            download_eurosat(log=lambda m: state.__setitem__("message", m))
            state.update(status="done", message="EuroSAT is ready")
        except Exception as exc:
            state.update(status="failed", message=str(exc))
    threading.Thread(target=work, daemon=True, name=f"download-{dataset_id}").start()
    return state


def explore_dir(dataset_id: str) -> str:
    return os.path.join(config.DATA_DIR, "dataset_stats", dataset_id)


def explore_dataset(dataset_id: str, refresh: bool = False) -> dict:
    """Dataset statistics + sample gallery for the studio's dataset explorer (cached)."""
    out = explore_dir(dataset_id)
    cached = os.path.join(out, "dataset.json")
    if os.path.exists(cached) and not refresh:
        return json.load(open(cached))
    import numpy as np
    from PIL import Image
    from training import datasets as ds
    from models import EUROSAT_CLASSES
    if dataset_id in ("synthetic", "eurosat"):
        if dataset_id == "synthetic":
            from training.synthetic_data import SyntheticEuroSAT
            synth = SyntheticEuroSAT(per_class=200)
            labels = np.array([i // synth.per_class for i in range(len(synth))])
            load, source = (lambda i: np.array(synth[i][0])), "synthetic EuroSAT-like patches"
        else:
            root = ds.find_eurosat_root()
            if not root:
                raise ValueError("EuroSAT is not downloaded yet")
            files, lab = [], []
            for c, name in enumerate(EUROSAT_CLASSES):
                for f in sorted(os.listdir(os.path.join(root, name))):
                    files.append(os.path.join(root, name, f))
                    lab.append(c)
            labels = np.array(lab)
            load, source = (lambda i: np.array(Image.open(files[i]).convert("RGB"))), "EuroSAT RGB"
        tr, va, te = ds.stratified_split(labels)
        stats = ds.describe_classification(load, labels, {"train": tr, "val": va, "test": te}, source, out)
    elif dataset_id in ("synthetic_change", "levir"):
        if dataset_id == "levir":
            if not ds.levir_ready():
                raise ValueError("LEVIR-CD is not imported yet")
            from training.train_change_detector import LevirCDPatches
            pairs = LevirCDPatches(ds.LEVIR_DIR, "train")
            sizes = {s: len(LevirCDPatches(ds.LEVIR_DIR, s)) for s in ("train", "val", "test")}
            source = "LEVIR-CD"
        else:
            from training.synthetic_data import SyntheticChangePairs
            pairs, sizes, source = SyntheticChangePairs(400), {"train": 400, "val": 50, "test": 50}, "synthetic change pairs"
        stats = ds.describe_change(pairs.raw, len(pairs), sizes, source, out, max_stats=60)
    else:
        raise KeyError(dataset_id)
    os.makedirs(out, exist_ok=True)
    with open(cached, "w") as f:
        json.dump(stats, f, default=float)
    return stats


ARTIFACTS = {"dataset.json", "architecture.json", "evaluation.json", "dataset_samples.png", "filters.png",
             "feature_maps.png", "predictions.png", "training_curves.png", "confusion_matrix.png"}


def artifact_path(job_id: int, name: str) -> str:
    if name not in ARTIFACTS:
        raise KeyError(name)
    path = os.path.join(_job_dir(job_id), "analytics", name)
    if not os.path.exists(path):
        raise KeyError(name)
    return path


def version_artifact_path(model: str, version: str, name: str) -> str:
    if name not in ARTIFACTS or model not in MODELS or not version.startswith("v") or not version[1:].isdigit():
        raise KeyError(name)
    path = os.path.join(config.MODEL_REGISTRY_DIR, model, f"{version}_analytics", name)
    if not os.path.exists(path):
        raise KeyError(name)
    return path


def job_analytics(job_id: int) -> dict:
    d = os.path.join(_job_dir(job_id), "analytics")
    out = {}
    for name in ("dataset.json", "architecture.json", "evaluation.json"):
        path = os.path.join(d, name)
        if os.path.exists(path):
            out[name.removesuffix(".json")] = json.load(open(path))
    out["images"] = sorted(n for n in ARTIFACTS if n.endswith(".png") and os.path.exists(os.path.join(d, n)))
    return out


# -- jobs ------------------------------------------------------------------------------------------

def _job_dir(job_id: int) -> str:
    return os.path.join(config.TRAINING_JOBS_DIR, str(int(job_id)))


def _command(model: str, dataset: str, p: dict, job_id: int) -> list:
    d = _job_dir(job_id)
    cmd = [sys.executable, "-m", MODELS[model]["script"], "--dataset", dataset,
           "--epochs", str(p["epochs"]), "--batch-size", str(p["batch_size"]), "--lr", str(p["lr"]),
           "--workers", str(p.get("workers", 0)), "--out", os.path.join(d, "model.pth"),
           "--progress-file", os.path.join(d, "progress.jsonl")]
    if not p.get("pretrained", True):
        cmd.append("--no-pretrained")
    if dataset == "synthetic":
        cmd += (["--samples-per-class", str(p.get("samples", 200))] if model == "classifier"
                else ["--synthetic-pairs", str(p.get("samples", 200))])
    if p.get("max_samples"):
        cmd += ["--max-samples", str(p["max_samples"])]
    if model == "classifier" and p.get("img_size"):
        cmd += ["--img-size", str(p["img_size"])]
    return cmd


def validate_params(model: str, dataset: str, p: dict) -> dict:
    if model not in MODELS:
        raise ValueError(f"model must be one of {list(MODELS)}")
    if dataset not in MODELS[model]["datasets"]:
        raise ValueError(f"dataset for {model} must be one of {MODELS[model]['datasets']}")
    out = {"epochs": int(p.get("epochs", 5)), "batch_size": int(p.get("batch_size", 16)),
           "lr": float(p.get("lr", 3e-4 if model == "classifier" else 1e-4)),
           "pretrained": bool(p.get("pretrained", True)), "samples": int(p.get("samples", 200)),
           "workers": int(p.get("workers", 0)), "max_samples": int(p.get("max_samples", 0) or 0),
           "img_size": int(p.get("img_size", 224 if dataset != "synthetic" else 64))}
    if not (1 <= out["epochs"] <= 500 and 1 <= out["batch_size"] <= 512 and 0 < out["lr"] < 1
            and 4 <= out["samples"] <= 20000 and 32 <= out["img_size"] <= 512 and out["max_samples"] >= 0):
        raise ValueError("epochs 1-500, batch size 1-512, learning rate 0-1, samples 4-20000, image size 32-512")
    return out


def start_job(model: str, dataset: str, params: dict, user: dict) -> dict:
    params = validate_params(model, dataset, params)
    with _lock:
        if any(p.poll() is None for p in _procs.values()):
            raise ValueError("a training job is already running — wait for it or cancel it")
        job_id = db.execute(
            "INSERT INTO training_jobs (model, dataset, params, status, created_by, created_at, started_at) "
            "VALUES (?, ?, ?, 'running', ?, ?, ?)",
            (model, dataset, db.dumps(params), user["id"] if user else None, db.now(), db.now()))
        d = _job_dir(job_id)
        os.makedirs(d, exist_ok=True)
        log_file = open(os.path.join(d, "log.txt"), "w")
        proc = subprocess.Popen(_command(model, dataset, params, job_id), cwd=ROOT, stdout=log_file,
                                stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                start_new_session=True)
        _procs[job_id] = proc
        db.execute("UPDATE training_jobs SET pid = ? WHERE id = ?", (proc.pid, job_id))
    watcher = threading.Thread(target=_watch, args=(job_id, proc, log_file), daemon=True, name=f"train-{job_id}")
    watcher.start()
    _threads[job_id] = watcher
    return get_job(job_id)


def wait_for_job(job_id: int, timeout: float = None) -> dict:
    """Block until a job (and its registration) has finished — used by the CLI pipeline."""
    thread = _threads.get(job_id)
    if thread:
        thread.join(timeout)
    return get_job(job_id)


def _watch(job_id: int, proc: subprocess.Popen, log_file) -> None:
    code = proc.wait()
    log_file.close()
    job = db.one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job["status"] == "cancelled":
        return
    ckpt = os.path.join(_job_dir(job_id), "model.pth")
    if code == 0 and os.path.exists(ckpt):
        try:
            version = register_version(job["model"], ckpt, source="studio", job_id=job_id, dataset=job["dataset"])
            db.execute("UPDATE training_jobs SET status = 'done', version = ?, finished_at = ? WHERE id = ?",
                       (version["version"], db.now(), job_id))
            return
        except Exception as exc:
            error = f"could not register the model: {exc}"
    else:
        events = _events(job_id)
        error = next((e["message"] for e in reversed(events) if e["event"] == "error"), f"exit code {code}")
    db.execute("UPDATE training_jobs SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
               (error, db.now(), job_id))


def _events(job_id: int) -> list:
    path = os.path.join(_job_dir(job_id), "progress.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # partially written last line
    return out


def _log_tail(job_id: int, n: int = 40) -> str:
    path = os.path.join(_job_dir(job_id), "log.txt")
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as f:
        lines = [ln for ln in f.read().replace("\r", "\n").split("\n") if ln.strip()]
    return "\n".join(lines[-n:])


def get_job(job_id: int, detail: bool = True) -> dict:
    job = db.one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if not job:
        raise KeyError(job_id)
    job["params"] = db.loads(job["params"], {})
    events = _events(job_id)
    start = next((e for e in events if e["event"] == "start"), None)
    epochs = [e for e in events if e["event"] == "epoch"]
    last_batch = next((e for e in reversed(events) if e["event"] == "batch"), None)
    done = next((e for e in events if e["event"] == "done"), None)
    job["progress"] = {
        "epochs_total": start["epochs"] if start else job["params"].get("epochs"),
        "epochs_done": len(epochs),
        "steps_per_epoch": start["steps_per_epoch"] if start else None,
        "current": last_batch,
        "fraction": _fraction(start, epochs, last_batch, job["status"]),
        "device": start["device"] if start else None,
        "pretrained": start["pretrained"] if start else None,
    }
    if detail:
        job["history"] = epochs
        job["batches"] = [e for e in events if e["event"] == "batch"][-300:]
        job["warnings"] = [e["message"] for e in events if e["event"] == "warning"]
        job["result"] = done
        job["log"] = _log_tail(job_id)
        job["info"] = [e["message"] for e in events if e["event"] == "info"]
        job["hyperparameters"] = start.get("hyperparameters") if start else None
        job["params_count"] = start.get("params") if start else None
        job["analytics"] = job_analytics(job_id)
    return job


def _fraction(start, epochs, last_batch, status) -> float:
    if status == "done":
        return 1.0
    if not start:
        return 0.0
    total = start["epochs"] * max(start["steps_per_epoch"], 1)
    done = len(epochs) * start["steps_per_epoch"]
    if last_batch and last_batch["epoch"] > len(epochs):
        done += last_batch["step"]
    return round(min(done / total, 0.99), 3)


def list_jobs() -> list:
    return [get_job(r["id"], detail=False) for r in
            db.query("SELECT id FROM training_jobs ORDER BY created_at DESC LIMIT 50")]


def cancel_job(job_id: int) -> None:
    proc = _procs.get(job_id)
    db.execute("UPDATE training_jobs SET status = 'cancelled', finished_at = ? WHERE id = ? AND status = 'running'",
               (db.now(), job_id))
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)


def recover_after_restart() -> None:
    """Jobs marked running whose process is gone (server restarted) are failed."""
    for job in db.query("SELECT id FROM training_jobs WHERE status = 'running'"):
        if job["id"] not in _procs:
            db.execute("UPDATE training_jobs SET status = 'failed', error = 'server restarted during training', "
                       "finished_at = ? WHERE id = ?", (db.now(), job["id"]))


# -- model registry ----------------------------------------------------------------------------------------

def _flat_metrics(model: str, metrics: dict) -> dict:
    if model == "classifier":
        return {"test_acc": metrics.get("test_acc"), "best_val_acc": metrics.get("best_val_acc"),
                "per_class": metrics.get("per_class_test_acc")}
    test = metrics.get("test", {})
    return {"test_f1": test.get("f1"), "test_iou": test.get("iou"), "test_precision": test.get("precision"),
            "test_recall": test.get("recall"), "best_val_f1": metrics.get("best_val_f1")}


def _load_state_dict(model: str, path: str):
    """Load a checkpoint into its architecture to prove it is usable (raises otherwise)."""
    import torch
    if model == "classifier":
        from models.classifier import SceneClassifier
        net = SceneClassifier(pretrained=False)
    else:
        from models.change_detector import SiameseChangeDetector
        net = SiameseChangeDetector(pretrained=False)
    net.load_state_dict(torch.load(path, map_location="cpu"))


def register_version(model: str, checkpoint: str, source: str, job_id: int = None, dataset: str = None,
                     metrics: dict = None, notes: str = None, validate: bool = False) -> dict:
    if model not in MODELS:
        raise ValueError(f"unknown model {model}")
    if validate:
        _load_state_dict(model, checkpoint)
    n = db.one("SELECT COUNT(*) AS n FROM model_versions WHERE model = ?", (model,))["n"] + 1
    version = f"v{n}"
    dest_dir = os.path.join(config.MODEL_REGISTRY_DIR, model)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{version}.pth")
    shutil.copyfile(checkpoint, dest)
    analytics_src = os.path.join(os.path.dirname(checkpoint), "analytics")
    if os.path.isdir(analytics_src):
        shutil.copytree(analytics_src, os.path.join(dest_dir, f"{version}_analytics"), dirs_exist_ok=True)
    if metrics is None:
        side = checkpoint.replace(".pth", ".metrics.json")
        metrics = json.load(open(side)) if os.path.exists(side) else {}
    flat = {**_flat_metrics(model, metrics), "dataset_name": metrics.get("dataset"),
            "pretrained": metrics.get("pretrained"), "epochs": len(metrics.get("history", [])) or None}
    db.execute("INSERT INTO model_versions (model, version, path, dataset, metrics, source, job_id, notes, created_at) "
               "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
               (model, version, dest, dataset, db.dumps(flat), source, job_id, notes, db.now()))
    return db.one("SELECT * FROM model_versions WHERE model = ? AND version = ?", (model, version))


def versions() -> dict:
    out = {}
    for key, spec in MODELS.items():
        rows = db.query("SELECT * FROM model_versions WHERE model = ? ORDER BY id DESC", (key,))
        for r in rows:
            r["metrics"] = db.loads(r["metrics"], {})
            r["active"] = bool(r["active"])
            r.pop("path", None)
        out[key] = {"title": spec["title"], "score": {"key": spec["score"][0], "label": spec["score"][1]},
                    "active_checkpoint": geotools.model_version(_GEOTOOLS_KEY[key]), "versions": rows}
    return out


def promote(model: str, version: str) -> dict:
    row = db.one("SELECT * FROM model_versions WHERE model = ? AND version = ?", (model, version))
    if not row:
        raise KeyError(version)
    _load_state_dict(model, row["path"])
    os.makedirs(config.MODEL_CHECKPOINT_DIR, exist_ok=True)
    target = os.path.join(config.MODEL_CHECKPOINT_DIR, MODELS[model]["checkpoint"])
    tmp = target + ".tmp"
    shutil.copyfile(row["path"], tmp)
    os.replace(tmp, target)  # atomic swap: in-flight runs keep the model they already loaded
    sidecar = {"model": model, "version": version, "promoted_at": db.now()}
    arch = os.path.join(os.path.dirname(row["path"]), f"{version}_analytics", "architecture.json")
    if os.path.exists(arch):  # remember the training input size so inference resizes patches the same way
        with open(arch) as f:
            shapes = json.load(f).get("input_shape") or [[]]
        if len(shapes[0]) == 4:
            sidecar["input_size"] = shapes[0][-1]
    with open(target + ".version.json", "w") as f:
        json.dump(sidecar, f)
    db.execute("UPDATE model_versions SET active = (version = ?) WHERE model = ?", (version, model))
    geotools.reload_models()
    return row


def deactivate(model: str) -> None:
    """Remove the active checkpoint so the tools fall back to the classical methods."""
    target = os.path.join(config.MODEL_CHECKPOINT_DIR, MODELS[model]["checkpoint"])
    for path in (target, target + ".version.json"):
        if os.path.exists(path):
            os.remove(path)
    db.execute("UPDATE model_versions SET active = 0 WHERE model = ?", (model,))
    geotools.reload_models()


def delete_version(model: str, version: str) -> None:
    row = db.one("SELECT * FROM model_versions WHERE model = ? AND version = ?", (model, version))
    if not row:
        raise KeyError(version)
    if row["active"]:
        raise ValueError("cannot delete the active version — promote another one first")
    if os.path.exists(row["path"]):
        os.remove(row["path"])
    db.execute("DELETE FROM model_versions WHERE id = ?", (row["id"],))
