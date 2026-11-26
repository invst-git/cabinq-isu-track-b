"""Build a submission folder that runs a mean-logit ensemble of trained soups.

Usage: python -I tools/build_ensemble_submission.py --out work/submission_v2v4 work/train_v2/soup work/train_v4/soup
Copies submission/system.py + submission/isu_core.py, each soup into weights/<variant>/, writes
weights/ensemble.json, and a Dockerfile that adds timm (pinned, --no-deps) only when a member needs it.
Refuses to overwrite an existing --out folder.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMM_PIN = "1.0.31"  # keep equal to train/train_isu.py TIMM_PIN

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("soups", nargs="+")
args = ap.parse_args()

out = Path(args.out)
if out.exists():
    raise SystemExit(f"{out} exists; remove it first")
names, needs_timm = [], False
for s in map(Path, args.soups):
    meta = json.loads((s / "meta.json").read_text())
    for f in ("model.safetensors", "config.json", "meta.json"):
        assert (s / f).is_file(), f"{s / f} missing"
    names.append(meta.get("variant") or s.parent.name)
    needs_timm |= isinstance(meta.get("backbone"), dict) and meta["backbone"].get("kind") == "timm"
assert len(set(names)) == len(names), f"duplicate member names {names}"

(out / "weights").mkdir(parents=True)
for f in ("system.py", "isu_core.py"):
    shutil.copy2(ROOT / "submission" / f, out / f)
for n, s in zip(names, map(Path, args.soups)):
    (out / "weights" / n).mkdir()
    for f in ("model.safetensors", "config.json", "meta.json"):
        shutil.copy2(s / f, out / "weights" / n / f)
(out / "weights" / "ensemble.json").write_text(json.dumps({"members": names, "combine": "mean_logit"}, indent=1))

docker = [
    "FROM isu-challenge-base",
    "",
    "# torch, torchvision, transformers, safetensors, Pillow and numpy come from the base image.",
    "# The submission folder (system.py, isu_core.py, weights/) is mounted read-only at /submission.",
]
check = "import torch, transformers, safetensors; print('torch', torch.__version__, '| transformers', transformers.__version__)"
if needs_timm:
    docker += [
        "# timm builds the DINOv3 member; --no-deps keeps the base image's torch/torchvision build intact.",
        f"RUN pip install --no-cache-dir --no-deps timm=={TIMM_PIN}",
    ]
    check = check.replace("safetensors;", "safetensors, timm;").replace(
        "transformers.__version__)", "transformers.__version__, '| timm', timm.__version__)")
docker.append(f'RUN python -c "{check}"')
(out / "Dockerfile").write_text("\n".join(docker) + "\n")
size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
print(f"wrote {out}: members {names}, timm={needs_timm}, {size / 1e6:.1f} MB")
