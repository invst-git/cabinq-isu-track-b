"""Shared code for ISU-Challenge Track B: label coding, preprocessing, model, inference engine.

Training (Kaggle) and the submission (system.py) both import this file, so preprocessing and
the network cannot drift apart. Keep it free of training-only dependencies.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

FEATURES = (
    "driver_gender", "driver_tshirt_color", "driver_emotion", "driver_phone", "driver_safety_belt",
    "passenger_codriver", "passenger_codriver_gender", "passenger_codriver_tshirt_color",
    "passenger_codriver_emotion", "codriver_safety_belt",
    "passenger_rear_seat_left", "passenger_rear_left_gender", "passenger_rear_left_tshirt_color",
    "passenger_rear_left_emotion", "passenger_rear_left_safety_belt",
    "passenger_rear_seat_right", "passenger_rear_right_gender", "passenger_rear_right_tshirt_color",
    "passenger_rear_right_emotion", "passenger_rear_right_safety_belt",
    "phone_codriver_seat", "phone_codriver_seat_color",
    "colabottle_codriver_seat", "colacan_codriver_seat",
    "suitcase", "suitcase_color", "suitcase_location", "suitcase_pose",
    "baby_seat", "baby_seat_orientation", "baby",
)


def _values(feature: str) -> tuple[str, str]:
    """(value for logit < 0, value for logit > 0). Saved in meta.json; inference decodes from there."""
    if feature.endswith("gender"):
        return ("FEMALE", "MALE")
    if feature.endswith("tshirt_color") or feature == "phone_codriver_seat_color":
        return ("BLACK", "WHITE")
    if feature.endswith("emotion"):
        return ("SERIOUS", "HAPPY")
    return {
        "suitcase_color": ("ANTRACITE", "YELLOW"),
        "suitcase_location": ("REAR_SEAT", "CO_DRIVER_SEAT"),
        "suitcase_pose": ("FLAT", "UPWARDS"),
        "baby_seat_orientation": ("REAR_FACING", "FRONT_FACING"),
    }.get(feature, ("NO", "YES"))


VALUES = {f: _values(f) for f in FEATURES}

# Every image is first brought to the render size (real photos are 1920x1080, renders 960x540),
# then resized on the device to the model input. The input size is stored per set of weights in
# meta.json ("input_size"); IN_W x IN_H is the default: 644x364 = 46x26 patches of 14 px, ~16:9.
CACHE_W, CACHE_H = 960, 540
IN_W, IN_H = 644, 364
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def prep(image: Image.Image) -> np.ndarray:
    """PIL image as the evaluator opens it (Image.open(p).convert("RGB"), no EXIF transpose)
    -> uint8 array (CACHE_H, CACHE_W, 3). The training cache and predict() both call this."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (CACHE_W, CACHE_H):
        image = image.resize((CACHE_W, CACHE_H), Image.BILINEAR)
    return np.array(image, dtype=np.uint8)  # a writable copy: torch.from_numpy warns on read-only arrays


def amp_ctx(enabled: bool):
    """fp16 autocast on CUDA (training and inference alike); a no-op context on CPU."""
    return torch.autocast("cuda", dtype=torch.float16) if enabled else contextlib.nullcontext()


def u8_to_float(batch: torch.Tensor) -> torch.Tensor:
    """uint8 (B, H, W, 3) -> float (B, 3, H, W) in [0, 1]."""
    return batch.permute(0, 3, 1, 2).float().div_(255.0)


def to_model_input(x: torch.Tensor, in_wh=(IN_W, IN_H)) -> torch.Tensor:
    """float (B, 3, H, W) in [0, 1] -> resized, normalized model input (B, 3, in_h, in_w)."""
    in_w, in_h = int(in_wh[0]), int(in_wh[1])
    if x.shape[-2:] != (in_h, in_w):
        x = F.interpolate(x, size=(in_h, in_w), mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class ISUNet(nn.Module):
    """DINOv2 backbone + one learned query per feature attending over all tokens.

    Per-feature queries let each answer look at its own part of the cabin (seat, belt, object),
    which plain CLS/mean pooling would blur. A CLS linear term is added for a stable start.
    """

    def __init__(self, backbone_config, n_out: int = len(FEATURES), n_heads: int = 8):
        super().__init__()
        from transformers import Dinov2Model

        try:
            self.backbone = Dinov2Model._from_config(backbone_config, attn_implementation="sdpa")
        except Exception:  # older/newer transformers without the sdpa switch
            self.backbone = Dinov2Model(backbone_config)
        d = backbone_config.hidden_size
        self.queries = nn.Parameter(torch.randn(n_out, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.w = nn.Parameter(torch.randn(n_out, d) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_out))
        self.cls_head = nn.Linear(d, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(pixel_values=x).last_hidden_state  # (B, 1 + N, d), final layernorm applied
        q = self.queries.unsqueeze(0).expand(h.shape[0], -1, -1)
        a, _ = self.attn(q, h, h, need_weights=False)
        z = self.norm(a + q)
        return (z * self.w).sum(-1) + self.b + self.cls_head(h[:, 0])


def save_weights(net: nn.Module, out_dir: Path, meta: dict) -> None:
    """Write config.json, model.safetensors (fp32) and meta.json (features, value coding, ...)."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    net = net.module if hasattr(net, "module") else net
    net.backbone.config.to_json_file(str(out_dir / "config.json"))
    state = {k: v.detach().float().cpu().contiguous() for k, v in net.state_dict().items()}
    save_file(state, str(out_dir / "model.safetensors"))
    meta = {"features": list(FEATURES), "values": {f: list(VALUES[f]) for f in FEATURES},
            "cache_size": [CACHE_W, CACHE_H], "input_size": [IN_W, IN_H], **meta}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


class Engine:
    """Inference: one image -> probabilities -> option strings, decoded from meta.json."""

    def __init__(self, weights_dir: Path, device: str = "cuda"):
        from safetensors.torch import load_file
        from transformers import Dinov2Config

        weights_dir = Path(weights_dir)
        self.meta = json.loads((weights_dir / "meta.json").read_text())
        if tuple(self.meta["cache_size"]) != (CACHE_W, CACHE_H):
            raise ValueError(f"weights were trained with another cache size: {self.meta['cache_size']}")
        self.in_wh = tuple(int(v) for v in self.meta["input_size"])
        if len(self.in_wh) != 2 or any(v % 14 for v in self.in_wh):
            raise ValueError(f"input_size must be two multiples of the 14 px patch: {self.in_wh}")
        self.features = self.meta["features"]
        self.values = self.meta["values"]
        self.thresholds = self.meta.get("thresholds", {})
        cfg = Dinov2Config.from_json_file(str(weights_dir / "config.json"))
        net = ISUNet(cfg, n_out=len(self.features))
        net.load_state_dict(load_file(str(weights_dir / "model.safetensors")), strict=True)
        self.device = torch.device(device)
        self.net = net.eval().to(self.device)
        self.amp = self.device.type == "cuda"

    @torch.inference_mode()
    def probs(self, image: Image.Image) -> np.ndarray:
        x = torch.from_numpy(prep(image)).unsqueeze(0).to(self.device, non_blocking=True)
        x = to_model_input(u8_to_float(x), self.in_wh)
        with amp_ctx(self.amp):
            logits = self.net(x)
        return torch.sigmoid(logits.float())[0].cpu().numpy()

    def predict(self, image: Image.Image) -> dict[str, str]:
        p = self.probs(image)
        return {f: self.values[f][int(p[i] >= self.thresholds.get(f, 0.5))]
                for i, f in enumerate(self.features)}

    def warmup(self, n: int = 3) -> None:
        blank = Image.new("RGB", (CACHE_W, CACHE_H), (90, 90, 90))
        for _ in range(n):
            self.probs(blank)
