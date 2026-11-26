"""Load a submission folder exactly as the evaluator does and dump probs + answers for some images.

Usage: python -I dump_probs.py <submission_dir> <data_main_dir> <out_json>
"""
import json
import sys
from pathlib import Path

from PIL import Image

sub, data, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
sys.path.insert(0, str(sub))
import system  # noqa: E402

s = system.create_system("cpu")
s.set_prompt("x")
imgs = [data / "real" / f"sample_{i:05d}_real.jpg" for i in range(30)]
sims = sorted((data / "images").glob("sample_*_sim.png"))
imgs += sims[:: max(1, len(sims) // 10)][:10]
res = {}
for p in imgs:
    im = Image.open(p).convert("RGB")
    res[p.name] = {"probs": [round(float(v), 6) for v in s.engine.probs(im)], "pred": s.predict(im)}
out.write_text(json.dumps(res))
print(f"{sub.name}: {len(res)} images, calls={s.calls} failures={s.failures} device={s.device}")
