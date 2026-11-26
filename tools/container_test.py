"""Self-test inside the built submission image under the official runtime restrictions
(no network, read-only file system except /tmp, non-root user), as in the starter kit's track_b README.

Example, from the repository root, after building the base image (starter kit: docker build -t isu-challenge-base docker/)
and the submission image (docker build -t cabinq submission/):
    docker run --rm --network none --read-only --tmpfs /tmp --user 1000:1000 -e HOME=/tmp \
        -v "$PWD/submission:/submission:ro" -v "<starter-kit clone>:/starterkit:ro" \
        -v "<folder with images/, real/, labels/>:/data:ro" -v "$PWD/runs/container:/out" \
        -v "$PWD/tools:/test:ro" --entrypoint python cabinq /test/container_test.py
Add --gpus all on a machine with an NVIDIA GPU. Mounts: /submission (ro), /starterkit (ro), /data (ro), /out (rw), /test (ro).
"""
import os
import sys
import time

sys.path[:0] = ["/submission", "/starterkit/track_b"]
print(f"uid={os.getuid()} cwd={os.getcwd()} HOME={os.environ.get('HOME')} "
      f"HF_HOME={os.environ.get('HF_HOME')} cpus={os.cpu_count()}", flush=True)
for probe in ("/submission/x", "/opt/x", "/tmp/x"):
    try:
        open(probe, "w").close()
        print(f"writable: {probe}")
    except OSError as e:
        print(f"read-only: {probe} ({type(e).__name__})")

t0 = time.perf_counter()
from system import create_system  # noqa: E402

system = create_system(os.environ.get("ISU_DEVICE", "cpu"))
print(f"create_system took {time.perf_counter() - t0:.1f}s (limit 900s)", flush=True)

from evaluator_cli import EvaluatorTrackB  # noqa: E402

orig = system.predict
times = []


def timed(image):
    t = time.perf_counter()
    out = orig(image)
    times.append(time.perf_counter() - t)
    return out


system.predict = timed
EvaluatorTrackB(data_dir="/data", isu_system=system, output_path="/out/report.json").evaluate()
print(f"predict: n={len(times)} max={max(times):.2f}s mean={sum(times) / len(times):.2f}s "
      f"(limit 30s) failures={system.failures}", flush=True)
