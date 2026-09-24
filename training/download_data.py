"""
training/download_data.py — Get the training datasets.

    python -m training.download_data eurosat                      # automatic (~90 MB)
    python -m training.download_data levir --from ~/Downloads/LEVIR-CD.zip
    python -m training.download_data status

LEVIR-CD has no stable direct-download link (it is distributed via Google
Drive / Baidu from https://justchenhao.github.io/LEVIR/ and mirrored on
Kaggle). Download the archive once, then import it with --from; the importer
accepts the official layout at any depth inside the zip or folder.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.datasets import download_eurosat, find_eurosat_root, import_levir, levir_ready  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["eurosat", "levir", "status"])
    p.add_argument("--from", dest="source", help="LEVIR-CD .zip or folder to import")
    args = p.parse_args()

    if args.dataset == "status":
        print(f"EuroSAT : {find_eurosat_root() or 'not downloaded — run: python -m training.download_data eurosat'}")
        print(f"LEVIR-CD: {'ready' if levir_ready() else 'not imported — see --help'}")
        print("Synthetic datasets: always available")
    elif args.dataset == "eurosat":
        download_eurosat()
    else:
        if not args.source:
            src = os.getenv("LEVIR_CD_PATH")
            if not src:
                raise SystemExit("Download LEVIR-CD from https://justchenhao.github.io/LEVIR/ (or Kaggle), then:\n"
                                 "  python -m training.download_data levir --from /path/to/LEVIR-CD.zip")
            args.source = src
        import_levir(args.source)


if __name__ == "__main__":
    main()
