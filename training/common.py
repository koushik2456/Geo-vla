"""training/common.py — Shared helpers for the training scripts."""
import json
import os
import time


class Progress:
    """Writes JSON-lines events that the in-app training studio reads live.

    Events: start, batch (throttled), epoch, warning, done, error.
    With no file configured it only prints, so the scripts work standalone too.
    """

    def __init__(self, path: str = None, batch_every: float = 2.0):
        self.path = path
        self.batch_every = batch_every
        self._last_batch = 0.0
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def emit(self, event: str, **data) -> None:
        record = {"event": event, "time": round(time.time(), 2), **data}
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(record, default=float) + "\n")
        if event != "batch":
            print(json.dumps(record, default=float), flush=True)

    def batch(self, **data) -> None:
        now = time.time()
        if now - self._last_batch >= self.batch_every:
            self._last_batch = now
            self.emit("batch", **data)


def build_with_pretrained_fallback(factory, pretrained: bool, progress: Progress):
    """Try ImageNet-initialised weights; if they cannot be downloaded (offline,
    firewall), fall back to random initialisation and say so."""
    if not pretrained:
        return factory(False), False
    try:
        return factory(True), True
    except Exception as exc:
        progress.emit("warning", message=f"ImageNet weights unavailable ({type(exc).__name__}); "
                                         "training from random initialisation — expect lower accuracy")
        return factory(False), False
