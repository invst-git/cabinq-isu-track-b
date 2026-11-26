"""ISU-Challenge Track B submission: DINOv2 multi-head classifier inside a reliability shell.

Contract (track_b/README.md): create_system(device) once, set_prompt(prompt) once, then
predict(image) per image.
- Loading: model on the requested device, else on CPU; if neither loads, create_system raises, so
  the free preflight rejects the submission instead of a priors-only run using a scored evaluation.
- Predicting: a raise or a >30 s call restarts the container, and after 3 restarts the remaining
  scenes are skipped, so predict() never raises and always answers all 31 features with valid
  option strings: model -> model reloaded on CPU (after a CUDA failure) -> majority answers (PRIORS).
"""
import os
import sys
import time
import traceback
from pathlib import Path

# No network at evaluation; the official runner is not documented to set these.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if not os.access(os.environ.get("HOME") or "/nonexistent", os.W_OK):
    os.environ["HOME"] = "/tmp"
os.environ.setdefault("USER", "isu")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:  # the evaluation makes track_b/ importable; do not depend on it
    from evaluator_cli import ISUSystem as _Base
except Exception:
    _Base = object

# Majority non-null value per feature over all 6,030 published labels (work/priors.json).
PRIORS = {
    "driver_gender": "MALE", "driver_tshirt_color": "WHITE", "driver_emotion": "HAPPY",
    "driver_phone": "NO", "driver_safety_belt": "NO",
    "passenger_codriver": "NO", "passenger_codriver_gender": "FEMALE",
    "passenger_codriver_tshirt_color": "BLACK", "passenger_codriver_emotion": "SERIOUS",
    "codriver_safety_belt": "NO",
    "passenger_rear_seat_left": "YES", "passenger_rear_left_gender": "MALE",
    "passenger_rear_left_tshirt_color": "BLACK", "passenger_rear_left_emotion": "HAPPY",
    "passenger_rear_left_safety_belt": "NO",
    "passenger_rear_seat_right": "NO", "passenger_rear_right_gender": "MALE",
    "passenger_rear_right_tshirt_color": "WHITE", "passenger_rear_right_emotion": "SERIOUS",
    "passenger_rear_right_safety_belt": "YES",
    "phone_codriver_seat": "NO", "phone_codriver_seat_color": "WHITE",
    "colabottle_codriver_seat": "NO", "colacan_codriver_seat": "NO",
    "suitcase": "NO", "suitcase_color": "ANTRACITE", "suitcase_location": "REAR_SEAT",
    "suitcase_pose": "FLAT", "baby_seat": "NO", "baby_seat_orientation": "REAR_FACING", "baby": "YES",
}
ALLOWED = {
    f: {"YES", "NO"} for f in PRIORS
}
for _f in PRIORS:
    if _f.endswith("gender"):
        ALLOWED[_f] = {"MALE", "FEMALE"}
    elif _f.endswith("tshirt_color") or _f == "phone_codriver_seat_color":
        ALLOWED[_f] = {"BLACK", "WHITE"}
    elif _f.endswith("emotion"):
        ALLOWED[_f] = {"HAPPY", "SERIOUS"}
ALLOWED.update({
    "suitcase_color": {"ANTRACITE", "YELLOW"},
    "suitcase_location": {"CO_DRIVER_SEAT", "REAR_SEAT"},
    "suitcase_pose": {"UPWARDS", "FLAT"},
    "baby_seat_orientation": {"FRONT_FACING", "REAR_FACING"},
})

WEIGHTS = HERE / "weights"


def _log(msg: str) -> None:
    print(f"[isu-system] {msg}", file=sys.stderr, flush=True)


class TrackBSystem(_Base):
    def __init__(self, device: str = "cuda"):
        self.prompt = None
        self.engine = None
        self.cpu_fallback_tried = False
        self.calls = 0
        self.failures = 0
        self.device = self._pick_device(device)
        started = time.perf_counter()
        self.engine = self._load(self.device)
        if self.engine is None and self.device != "cpu" and (WEIGHTS / "model.safetensors").exists():
            _log("loading on the GPU failed; retrying on CPU")
            self.cpu_fallback_tried = True
            self.device = "cpu"
            self.engine = self._load("cpu")
        if self.engine is None:
            # Fail loudly here: a preflight failure is shown on the PR and does not use one of the
            # 3 daily evaluations, whereas a priors-only system would pass it and waste a scored run.
            raise RuntimeError(f"model could not be loaded from {WEIGHTS} (see the traceback above)")
        _log(f"ready on {self.device} in {time.perf_counter() - started:.1f}s")

    @staticmethod
    def _pick_device(device):
        try:
            import torch
            if str(device).startswith("cuda") and not torch.cuda.is_available():
                _log("CUDA requested but unavailable; using CPU")
                return "cpu"
        except Exception:
            traceback.print_exc()
            return "cpu"
        return str(device or "cpu")

    @staticmethod
    def _load(device):
        if not (WEIGHTS / "model.safetensors").exists():
            _log(f"no weights in {WEIGHTS}")
            return None
        try:
            import torch
            from isu_core import Engine
            torch.set_grad_enabled(False)
            if device == "cpu":
                torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
            engine = Engine(WEIGHTS, device)
            engine.warmup()
            return engine
        except Exception:
            traceback.print_exc()
            return None

    def set_prompt(self, prompt):
        # The prompt lists the options seen in the current run (only one scene at preflight),
        # so it must never restrict the answers. Kept for the interface only.
        self.prompt = prompt

    def name(self):
        return "isu-tb-dinov2b-multihead"

    def predict(self, image):
        self.calls += 1
        out = dict(PRIORS)
        if self.engine is None:
            return out
        try:
            pred = self.engine.predict(image)
        except Exception as exc:
            self.failures += 1
            traceback.print_exc()
            if self.device != "cpu" and not self.cpu_fallback_tried:
                # A CUDA error poisons the context; never touch the GPU again, rebuild on CPU.
                self.cpu_fallback_tried = True
                _log(f"GPU inference failed ({type(exc).__name__}); reloading on CPU")
                self.device = "cpu"
                self.engine = self._load("cpu")
                if self.engine is not None:
                    try:
                        pred = self.engine.predict(image)
                    except Exception:
                        traceback.print_exc()
                        self.engine = None
                        return out
                else:
                    return out
            else:
                return out
        for key, value in pred.items():
            if key in out:
                v = str(value).strip().upper()
                if v in ALLOWED[key]:
                    out[key] = v
        return out


def create_system(device="cuda"):
    return TrackBSystem(device)
