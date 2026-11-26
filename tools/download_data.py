"""Download the ISU-Challenge Track B data, pinned to fixed dataset commits.

Usage:
    python -I tools/download_data.py --out data --mode local   # small subset for CPU smoke tests
    python -I tools/download_data.py --out /tmp/isu --mode full --tar-to /kaggle/working
On Kaggle with no arguments it runs the second form, so the file can be pushed as a script
kernel unchanged; training kernels then attach its output (isu_main.tar, isu_extended.tar).

Only RGB images, real photos, labels, instance masks and manifests are fetched; depth and
Canny maps (about 17 GB of the 21 GB) are skipped.

Integrity: both revisions are pinned to commits published after 2026-10-26, when the withheld
real photos sample_00030-00059 were removed. The script refuses to finish if real/ holds
anything other than sample_00000-00029. Never point it at an older revision.
"""
import argparse
import datetime as dt
import json
import os
import sys
import tarfile
import time
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "ISU-Test/isu-challenge-dataset"
MAIN_SHA = "e9a8eb3ca955e5ef94631139a4044569a7b64163"  # main, 2026-10-27
EXT_SHA = "35834ed5417d9fda4cc6f97a0154ccdeb145f81e"  # extended, 2026-10-21 (no real photos)

COMMON = ["labels/*.json", "scene_manifest.json", "README.md", "instance_seg/*.json"]

# Local subset: the 30 real photos, their 30 twin renders, 67 spread-out synthetic renders,
# all labels and all main-branch instance masks (~115 MB in total).
LOCAL_SIM = [f"images/sample_{i:05d}_sim.png" for i in range(30)] + [
    f"images/sample_{i:05d}_sim.png" for i in range(30, 1030, 15)
]
PATTERNS = {
    "local": {
        "main": COMMON + ["real/*.jpg", "instance_seg/*.png"] + LOCAL_SIM,
        "extended": COMMON,
    },
    "full": {
        "main": COMMON + ["real/*.jpg", "instance_seg/*.png", "images/*.png"],
        "extended": COMMON + ["instance_seg/*.png", "images/*.png"],
    },
}
EXPECTED_REAL = {f"sample_{i:05d}_real.jpg" for i in range(30)}


def fetch(branch: str, sha: str, target: Path, patterns: list[str], workers: int) -> None:
    """snapshot_download with retries; finished files are skipped on retry (pinned SHA)."""
    # The Hub allows 5,000 file resolves per 5 minutes signed in (fewer anonymously), and the
    # full set is ~18k files, so a 429 is expected: wait out the window and resume.
    for attempt in range(1, 31):
        try:
            snapshot_download(REPO, repo_type="dataset", revision=sha, local_dir=target,
                              allow_patterns=patterns, max_workers=workers)
            return
        except Exception as exc:  # rate limits (429) and transient network errors
            wait = 310 if "429" in str(exc) else min(30 * attempt, 300)
            print(f"[{branch}] attempt {attempt} failed: {str(exc)[:160]!r}; retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"[{branch}] download failed after retries")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--mode", choices=("local", "full"), default="local")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tar-to", type=Path, help="also pack each branch into <dir>/isu_<branch>.tar")
    on_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ and len(sys.argv) == 1
    args = ap.parse_args(["--out", "/tmp/isu", "--mode", "full", "--tar-to", "/kaggle/working"]
                         if on_kaggle else None)
    if args.out is None:
        ap.error("--out is required")
    if on_kaggle:  # tqdm bars flood the kernel log
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()

    out = args.out.resolve()
    for branch, sha in (("main", MAIN_SHA), ("extended", EXT_SHA)):
        target = out / branch
        print(f"[{branch}] {sha[:10]} -> {target}", flush=True)
        fetch(branch, sha, target, PATTERNS[args.mode][branch], args.workers)

    # Integrity gate: only the 30 public real photos may exist anywhere in the download.
    real = {p.name for p in (out / "main" / "real").glob("*")}
    stray = sorted(p.relative_to(out).as_posix() for p in out.rglob("*_real.*")
                   if p.name not in EXPECTED_REAL and ".cache" not in p.parts)
    if real != EXPECTED_REAL or stray or (out / "extended" / "real").exists():
        print(f"INTEGRITY FAILURE: real={sorted(real ^ EXPECTED_REAL)} stray={stray}", file=sys.stderr)
        return 1

    counts = {}
    for branch in ("main", "extended"):
        for sub in ("images", "real", "labels", "instance_seg"):
            d = out / branch / sub
            if d.is_dir():
                counts[f"{branch}/{sub}"] = sum(1 for _ in d.iterdir())
    provenance = {
        "repo": REPO,
        "main_sha": MAIN_SHA,
        "extended_sha": EXT_SHA,
        "mode": args.mode,
        "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "file_counts": counts,
        "real_photos": sorted(real),
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance["file_counts"], indent=2))
    print("OK: integrity check passed (real/ holds exactly sample_00000-00029)")

    if args.tar_to:
        # Uncompressed tar: PNG/JPG are already compressed, and a few large files beat
        # ~18k small ones for Kaggle kernel output and Drive copies.
        args.tar_to.mkdir(parents=True, exist_ok=True)
        for branch in ("main", "extended"):
            dest = args.tar_to / f"isu_{branch}.tar"
            # dereference: older huggingface_hub may leave symlinks into its cache
            with tarfile.open(dest, "w", dereference=True) as tar:
                for sub in sorted((out / branch).iterdir()):
                    if sub.name != ".cache":
                        tar.add(sub, arcname=f"{branch}/{sub.name}")
            print(f"packed {dest} ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)
        (args.tar_to / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
