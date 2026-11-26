"""Run submission/system.py through the official Track B evaluator (CPU), plus a preflight-style check.

Usage: python harness_system.py <submission_dir> <starterkit_track_b_dir> <data_main_dir> <out_json> [image_kind]
"""
import sys
import time
from pathlib import Path

sub, tb, data, out = (Path(a) for a in sys.argv[1:5])
kind = sys.argv[5] if len(sys.argv) > 5 else "both"
sys.path[:0] = [str(sub), str(tb)]

from PIL import Image  # noqa: E402

import system  # noqa: E402  (the submission's system.py)
from evaluator_cli import EvaluatorTrackB  # noqa: E402
from track_b_evaluator import TRACK_B_FEATURES  # noqa: E402

t0 = time.perf_counter()
s = system.create_system("cpu")
print(f"create_system: {time.perf_counter() - t0:.1f}s, base={type(s).__mro__[1].__name__}, engine={'yes' if s.engine else 'NO'}")

# preflight imitation: one scene, prompt for that scene only, must answer features without raising
s.set_prompt("dummy prompt listing one scene's options")
img = Image.open(data / "images" / "sample_00000_sim.png").convert("RGB")
pred = s.predict(img)
missing = [f for f in TRACK_B_FEATURES if f not in pred]
print(f"preflight: {len(pred)} features answered, missing {missing}")
assert not missing and all(isinstance(v, str) for v in pred.values())

# robustness: odd inputs must still produce 31 answers
for name, im in [("tiny", Image.new("RGB", (17, 9))), ("portrait", Image.new("RGB", (540, 960), (200, 10, 10))),
                 ("RGBA", Image.new("RGBA", (960, 540))), ("L", Image.new("L", (1920, 1080)))]:
    p = s.predict(im)
    assert len(p) == 31, (name, len(p))
print("odd-input checks: ok")

report = EvaluatorTrackB(data, s, kind, output_path=out, team_name="local_test").evaluate()
m = report["metrics"]
print(f"RESULT scenes={m['scenes']} acc={m.get('accuracy')} latency_mean={m['latency_seconds_mean']:.3f}s")
