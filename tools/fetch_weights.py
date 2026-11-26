"""Fetch the trained CabinQ weights from the GitHub release, verify SHA-256, and place them.

The weights are too large for the git repository (about 350 MB each), so they are attached to a
GitHub release. Usage, from the repository root:

    python -I tools/fetch_weights.py                       # public repo: download over HTTPS
    python -I tools/fetch_weights.py --from-dir weights_dl # files already downloaded, e.g. with
                                                           #   gh release download weights-v1 -R <owner>/<repo> -D weights_dl
    python -I tools/fetch_weights.py --only v2             # just the final model

Each file is checked against the SHA-256 logged when it was submitted (provenance/submission_log.txt)
and copied to:
    submission/weights/model.safetensors        (v2, the final system)
    work/train_<v>/soup/model.safetensors        (every variant; used by tools/check_weights.py)
    work/submission_v2/weights/model.safetensors (v2; used by analysis/compute_metrics.py)
"""
import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "invst-git/cabinq-isu-track-b"  # change if the repository is renamed or forked
TAG = "weights-v1"
WEIGHTS = {  # variant: (release asset name, SHA-256, size in bytes)
    "v1": ("cabinq-v1.safetensors", "8660e2d7281b8c91bb7ea1cf8ce1ec131fd7e0e03341692d283b0cdf64c4a3a8", 356090392),
    "v2": ("cabinq-v2.safetensors", "8031f1b441dd850826c660050ec3dc90fb30843889a0017375e4885f94f8aa64", 356090392),
    "v2s1": ("cabinq-v2s1.safetensors", "75bec17a437c3f83cfe4fb4e9fa09314c0c58eb65a599807d83a75aae8b12d00", 356090392),
    "v4": ("cabinq-v4.safetensors", "d15e49f0c9600cd2713d32d0916f6b091f908963da8956d1cf562fbb34284b59", 352323664),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--from-dir", type=Path, help="take the assets from this folder instead of downloading")
    ap.add_argument("--only", nargs="*", choices=sorted(WEIGHTS), help="variants to fetch (default: all available)")
    args = ap.parse_args()
    cache = args.from_dir or ROOT / "weights_dl"
    cache.mkdir(parents=True, exist_ok=True)
    ok = True
    for v in args.only or WEIGHTS:
        name, digest, size = WEIGHTS[v]
        src = cache / name
        if not src.exists():
            if args.from_dir:
                print(f"[{v}] {name} not in {cache}; skipped")
                continue
            url = f"https://github.com/{args.repo}/releases/download/{args.tag}/{name}"
            print(f"[{v}] downloading {url}")
            try:
                urllib.request.urlretrieve(url, src)
            except urllib.error.HTTPError as exc:
                src.unlink(missing_ok=True)
                if exc.code == 404 and not args.only and v != "v2":  # v2 (the final model) is required
                    print(f"[{v}] not attached to release {args.tag}; skipped")
                    continue
                print(f"[{v}] download failed ({exc}); for a private repository use 'gh release download' and --from-dir")
                ok = False
                continue
            except Exception as exc:
                src.unlink(missing_ok=True)
                print(f"[{v}] download failed ({exc})")
                ok = False
                continue
        got = sha256(src)
        if got != digest or src.stat().st_size != size:
            print(f"[{v}] CHECKSUM MISMATCH: {got} (expected {digest})")
            ok = False
            continue
        targets = [ROOT / f"work/train_{v}/soup/model.safetensors"]
        if v == "v2":
            targets += [ROOT / "submission/weights/model.safetensors", ROOT / "work/submission_v2/weights/model.safetensors"]
        for t in targets:
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, t)
        print(f"[{v}] verified {digest[:16]}... -> {', '.join(str(t.relative_to(ROOT)) for t in targets)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
