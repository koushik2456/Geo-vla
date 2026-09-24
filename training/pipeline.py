"""
training/pipeline.py — One command from nothing to a trained, registered, promoted model.

    python -m training.pipeline                      # quick demo: both models on synthetic data (CPU, ~5 min)
    python -m training.pipeline --real               # EuroSAT classifier (auto-download) + LEVIR-CD if imported
    python -m training.pipeline --model classifier --dataset eurosat --epochs 10 --img-size 224
    python -m training.pipeline --model classifier --dataset eurosat --max-samples 3000 --img-size 64 --epochs 6

Each run goes through the same job system as the web training studio, so it
appears there with live curves and full analytics, is registered in the model
registry as a new version, and (with --promote, the default) goes into production.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUICK = {
    "classifier": {"dataset": "synthetic", "epochs": 6, "batch_size": 32, "lr": 1e-3, "samples": 100, "img_size": 64,
                   "pretrained": True},
    "change_detector": {"dataset": "synthetic", "epochs": 4, "batch_size": 4, "lr": 3e-4, "samples": 64,
                        "pretrained": True},
}
REAL = {
    "classifier": {"dataset": "eurosat", "epochs": 10, "batch_size": 64, "lr": 3e-4, "img_size": 224, "pretrained": True},
    "change_detector": {"dataset": "levir", "epochs": 50, "batch_size": 8, "lr": 1e-4, "pretrained": True},
}


def run_one(model: str, spec: dict, promote: bool) -> dict:
    from services import db, training
    db.init()
    spec = dict(spec)
    dataset = spec.pop("dataset")
    if dataset == "eurosat":
        from training.datasets import download_eurosat
        download_eurosat()
    if dataset == "levir":
        from training.datasets import levir_ready
        if not levir_ready():
            print("LEVIR-CD not imported — skipping the change detector. Import it with:\n"
                  "  python -m training.download_data levir --from /path/to/LEVIR-CD.zip")
            return {}
    job = training.start_job(model, dataset, spec, user=None)
    print(f"\n▶ job #{job['id']}: {model} on {dataset}  {spec}")
    last = None
    while True:
        j = training.get_job(job["id"], detail=True)
        if j["history"] and j["history"][-1] != last:
            last = j["history"][-1]
            print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in last.items() if k not in ("event", "time")))
        if j["status"] != "running":
            break
        time.sleep(2)
    j = training.wait_for_job(job["id"])
    if j["status"] != "done":
        print(f"✗ job #{j['id']} {j['status']}: {j.get('error')}\n{j.get('log', '')[-2000:]}")
        return j
    print(f"✓ registered {model} {j['version']}")
    if promote:
        training.promote(model, j["version"])
        print(f"✓ promoted {model} {j['version']} to production")
    return j


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", action="store_true", help="Train on EuroSAT / LEVIR-CD instead of synthetic data")
    p.add_argument("--model", choices=["classifier", "change_detector", "both"], default="both")
    p.add_argument("--dataset", choices=["synthetic", "eurosat", "levir"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--img-size", type=int)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--samples", type=int, help="synthetic dataset size")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-promote", action="store_true")
    args = p.parse_args()

    base = REAL if args.real else QUICK
    models = ["classifier", "change_detector"] if args.model == "both" else [args.model]
    for model in models:
        spec = dict(base[model])
        overrides = {"dataset": args.dataset, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                     "img_size": args.img_size, "max_samples": args.max_samples, "samples": args.samples}
        spec.update({k: v for k, v in overrides.items() if v is not None})
        if args.no_pretrained:
            spec["pretrained"] = False
        if model == "change_detector":
            spec.pop("img_size", None)
            if spec["dataset"] == "eurosat":
                spec["dataset"] = "levir"
        run_one(model, spec, promote=not args.no_promote)
    print("\nOpen the Training studio in the web app to see curves, confusion matrices and all analytics.")


if __name__ == "__main__":
    main()
