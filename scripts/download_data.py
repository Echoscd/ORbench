#!/usr/bin/env python3
"""
Download ORBench test data from HuggingFace Hub.

Usage:
  # First-time: pip install huggingface_hub

  # Default: download small (fastest, 97 MB)
  python3 scripts/download_data.py

  # Specific size
  python3 scripts/download_data.py small           # 97 MB
  python3 scripts/download_data.py medium          # 3.6 GB
  python3 scripts/download_data.py large           # 8.2 GB
  python3 scripts/download_data.py all             # 12 GB

  # Use a specific repo
  python3 scripts/download_data.py --repo-id myuser/orbench-data small
"""
import argparse, os, tarfile, sys, json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "Cosmoscd/AccelEval"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("size", nargs="?", default="small",
                    choices=["small", "medium", "large", "all"])
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--verify", action="store_true",
                    help="Verify SHA256 against manifest after download")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        sys.exit(1)

    sizes = ["small", "medium", "large"] if args.size == "all" else [args.size]
    patterns = [f"{s}.tar.gz" for s in sizes] + ["manifest.json"]

    print(f"[download] repo={args.repo_id}, sizes={sizes}")
    cache_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
    )
    cache_dir = Path(cache_dir)

    # Verify
    if args.verify:
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        for s in sizes:
            tarball = cache_dir / f"{s}.tar.gz"
            expected = manifest["tarballs"][f"{s}.tar.gz"]["sha256"]
            actual = sha256(tarball)
            if actual != expected:
                print(f"  ❌ {s}.tar.gz sha256 mismatch")
                sys.exit(1)
            print(f"  ✅ {s}.tar.gz sha256 verified")

    # Extract
    tasks_dir = ROOT / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    for s in sizes:
        tarball = cache_dir / f"{s}.tar.gz"
        print(f"[extract] {tarball.name} → {tasks_dir}/")
        with tarfile.open(tarball) as tar:
            tar.extractall(tasks_dir)

    print(f"\n✅ Done. Data extracted to {tasks_dir}/")


if __name__ == "__main__":
    main()
