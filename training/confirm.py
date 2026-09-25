"""
training/confirm.py — One-shot confirmation of a trained model on its sealed test split.

Training never loads the test split (it evaluates on validation only). This command is the
only code path that does, and it refuses to run unless every integrity condition holds:

  1. eval/confirmation/<name>/PREREG.md exists and its SHA-256 matches PREREG.sha256
     (the pre-registration is frozen: hypotheses, metrics, pass bars, split identity).
  2. The test split rebuilt from the dataset has exactly the SHA-256 recorded in PREREG.md.
  3. APPROVAL.md exists and quotes the PREREG hash, the checkpoint hash and the split hash
     (the user's explicit go-ahead for THIS model on THIS split).
  4. SPENT.json does not exist (the split has not already been used for a confirmation).

It then evaluates once, writes results.json (every metric plus pass/fail against the frozen
bars), and writes SPENT.json. A spent split may still be used for development, never again for
confirmation. Nothing in a completed confirmation folder is ever rewritten.

    python -m training.confirm --name classifier-eurosat --version v3
    python -m training.confirm --name classifier-eurosat --version v3 --check   # verify only, no evaluation
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from models import EUROSAT_CLASSES  # noqa: E402

CONF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "confirmation")


class Refused(SystemExit):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prereg_spec(prereg_text: str) -> dict:
    """The machine-readable block of PREREG.md (```json prereg ... ```)."""
    m = re.search(r"```json prereg\s*\n(.*?)\n```", prereg_text, re.S)
    if not m:
        raise Refused("PREREG.md has no ```json prereg``` block")
    return json.loads(m.group(1))


def checkpoint_path(model: str, version: str) -> str:
    path = os.path.join(config.MODEL_REGISTRY_DIR, model, f"{version}.pth")
    if not os.path.exists(path):
        raise Refused(f"no registered checkpoint {path}")
    return path


def eurosat_test_split(spec: dict):
    """Rebuild the sealed EuroSAT test split exactly as training defines it."""
    from training.datasets import find_eurosat_root, split_manifest, stratified_split
    root = find_eurosat_root()
    if not root:
        raise Refused("EuroSAT is not downloaded")
    files, labels = [], []
    for c, name in enumerate(EUROSAT_CLASSES):
        for f in sorted(os.listdir(os.path.join(root, name))):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                files.append(os.path.join(root, name, f))
                labels.append(c)
    labels = np.array(labels)
    ids = [os.path.relpath(f, root).replace(os.sep, "/") for f in files]
    _, _, test_idx = stratified_split(labels, seed=spec["split_seed"])
    return files, labels, test_idx, split_manifest(ids, test_idx)


def gate(name: str, model: str, version: str) -> dict:
    """Check every integrity condition; return the facts needed to run. Raises Refused."""
    folder = os.path.join(CONF_DIR, name)
    prereg = os.path.join(folder, "PREREG.md")
    if not os.path.exists(prereg) or not os.path.exists(prereg + ".sha256"):
        raise Refused(f"{prereg} and PREREG.md.sha256 must exist (pre-register and freeze first)")
    prereg_hash = sha256_file(prereg)
    frozen = open(prereg + ".sha256").read().split()[0]
    if prereg_hash != frozen:
        raise Refused(f"PREREG.md changed after freezing ({prereg_hash} != {frozen}); record changes in DEVIATIONS "
                      "and re-freeze only with the user's agreement")
    spec = prereg_spec(open(prereg, encoding="utf-8").read())
    if spec["model"] != model:
        raise Refused(f"PREREG is for model {spec['model']}, not {model}")
    ckpt = checkpoint_path(model, version)
    ckpt_hash = sha256_file(ckpt)
    if spec["dataset"] != "eurosat":
        raise Refused(f"confirmation for dataset {spec['dataset']} is not implemented")
    files, labels, test_idx, manifest = eurosat_test_split(spec)
    if manifest["sha256"] != spec["test_split_sha256"]:
        raise Refused(f"rebuilt test split {manifest['sha256']} does not match PREREG {spec['test_split_sha256']}")
    facts = {"folder": folder, "prereg_sha256": prereg_hash, "checkpoint": ckpt, "checkpoint_sha256": ckpt_hash,
             "test_split_sha256": manifest["sha256"], "test_count": manifest["count"], "spec": spec,
             "files": files, "labels": labels, "test_idx": test_idx}
    approval = os.path.join(folder, "APPROVAL.md")
    missing = [k for k in ("prereg_sha256", "checkpoint_sha256", "test_split_sha256")
               if not os.path.exists(approval) or facts[k] not in open(approval, encoding="utf-8").read()]
    if missing:
        raise Refused("APPROVAL.md must exist and quote these hashes:\n" +
                      "\n".join(f"  {k}: {facts[k]}" for k in ("prereg_sha256", "checkpoint_sha256", "test_split_sha256")))
    if os.path.exists(os.path.join(folder, "SPENT.json")):
        raise Refused(f"{name} is SPENT: its test split was already used for a confirmation")
    return facts


def evaluate(facts: dict, batch_size: int = 128) -> dict:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader

    from models.classifier import SceneClassifier
    from training import analytics
    from training.train_classifier import ImageList, transforms_for

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SceneClassifier(num_classes=len(EUROSAT_CLASSES), pretrained=False)
    model.load_state_dict(torch.load(facts["checkpoint"], map_location=device))
    model.to(device).eval()
    files = facts["files"]
    _, eval_tf = transforms_for(facts["spec"]["img_size"])
    dl = DataLoader(ImageList(lambda i: np.array(Image.open(files[i]).convert("RGB")), facts["labels"], facts["test_idx"],
                              eval_tf), batch_size=batch_size, shuffle=False, num_workers=0)
    preds, trues = [], []
    with torch.no_grad():
        for x, y in dl:
            preds.append(model(x.to(device)).argmax(1).cpu().numpy())
            trues.append(y.numpy())
    report = analytics.classification_report(np.concatenate(trues), np.concatenate(preds), EUROSAT_CLASSES)
    return report


def decide(report: dict, bars: dict) -> dict:
    """Apply the frozen pass bars. Returns {bar: {value, threshold, passed}}."""
    values = {"accuracy": report["accuracy"], "macro_f1": report["macro_f1"],
              "min_class_recall": min(v["recall"] for v in report["per_class"].values() if v["support"])}
    return {k: {"value": round(values[k], 4), "threshold": t, "passed": values[k] >= t} for k, t in bars.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="confirmation folder under eval/confirmation/")
    p.add_argument("--model", default="classifier")
    p.add_argument("--version", required=True, help="registered model version, e.g. v3")
    p.add_argument("--check", action="store_true", help="verify the gate and print the hashes; do not evaluate")
    args = p.parse_args(argv)
    facts = None
    try:
        facts = gate(args.name, args.model, args.version)
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(json.dumps({k: facts[k] for k in ("prereg_sha256", "checkpoint_sha256", "test_split_sha256", "test_count")},
                     indent=2))
    if args.check:
        print("gate OK (check only, nothing evaluated)")
        return 0
    report = evaluate(facts)
    decision = decide(report, facts["spec"]["pass_bars"])
    results = {"name": args.name, "model": args.model, "version": args.version,
               "ran_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               **{k: facts[k] for k in ("prereg_sha256", "checkpoint_sha256", "test_split_sha256", "test_count")},
               "decision": decision, "passed_all": all(d["passed"] for d in decision.values()), "report": report}
    out = os.path.join(facts["folder"], "results.json")
    with open(out, "x") as f:  # "x": never overwrite a completed confirmation
        json.dump(results, f, indent=2)
    with open(os.path.join(facts["folder"], "SPENT.json"), "x") as f:
        json.dump({"spent_at": results["ran_at"], "results_sha256": sha256_file(out), "version": args.version,
                   "checkpoint_sha256": facts["checkpoint_sha256"]}, f, indent=2)
    print(json.dumps({"decision": decision, "passed_all": results["passed_all"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
