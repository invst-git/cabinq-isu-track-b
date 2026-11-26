"""Step 0d checks on Kaggle (T4 x2). Each check is independent; results go to /kaggle/working/checks.json.

A. v2 soup on the 500-render synthetic hold-out: per-image probabilities, errors, near-threshold decisions.
B. CUDA-graph replay of the v2 inference path: bit-equality with eager, and per-image timing (T4).
C. DINOv3-B/16 (timm, not gated): fp16 autocast forward/backward finiteness on real photos and renders at
   608x1088 and 544x960, fp16-vs-fp32 feature agreement, activation maxima, and train step time (bs 32, 2 GPUs).
D. DINOv2-L/14 train step time at 364x644 (random init; timing only) for the fallback plan.
Bundled with submission/isu_core.py by kaggle/build_check_kernel.py.
"""
import json
import math
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from isu_core import (CACHE_H, CACHE_W, FEATURES, MEAN, STD, VALUES, Engine, amp_ctx, prep,  # BUNDLE-STRIP
                      to_model_input, u8_to_float)  # BUNDLE-STRIP

OUT = Path("/kaggle/working")
RES = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save():
    (OUT / "checks.json").write_text(json.dumps(RES, indent=2))


def run(name, fn):
    t0 = time.time()
    try:
        RES[name] = {"ok": True, **fn()}
    except Exception as exc:  # keep going: every check is independent
        RES[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-3000:]}
    RES[name]["seconds"] = round(time.time() - t0, 1)
    log(f"{name}: {json.dumps(RES[name])[:1500]}")
    save()
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ setup
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==__TRANSFORMERS_PIN__", "timm>=1.0.20"],
               check=False)
import timm  # noqa: E402
import transformers  # noqa: E402

RES["env"] = {"torch": torch.__version__, "transformers": transformers.__version__, "timm": timm.__version__,
              "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
log(RES["env"])

root = Path("/tmp/isu")
for t in sorted(Path("/kaggle/input").rglob("isu_*.tar")):
    branch = t.stem.replace("isu_", "")
    if not (root / branch / "labels").is_dir():
        log(f"extracting {t}")
        with tarfile.open(t) as tf:
            tf.extractall(root, filter="data")
v2_dir = next(p.parent for p in Path("/kaggle/input").rglob("soup/model.safetensors"))
splits = json.loads((v2_dir.parent / "splits.json").read_text())
log(f"v2 soup at {v2_dir}; holdout {len(splits['holdout'])}")


def key_path(key):
    branch, rest = key.split("/")
    stem, kind = rest.rsplit("_", 1)
    sub = "images" if kind == "sim" else "real"
    return root / branch / sub / f"{stem}_{kind}.{'png' if kind == 'sim' else 'jpg'}", root / branch / "labels" / f"{stem}.json"


def load_rgb(p):
    with Image.open(p) as im:
        return im.convert("RGB")


real_keys = [f"main/sample_{i:05d}_real" for i in range(30)]


# ------------------------------------------------------------------ A
def check_a():
    eng = Engine(v2_dir, "cuda")
    eng.warmup()
    rows, wrong, near05, near1 = {}, [], 0, 0
    min_margin = (1e9, None)
    for key in splits["holdout"] + real_keys:
        ip, lp = key_path(key)
        p = eng.probs(load_rgb(ip))
        label = json.loads(lp.read_text())
        lg = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
        rows[key] = [round(float(v), 5) for v in lg]
        for i, f in enumerate(FEATURES):
            v = label.get(f)
            if v is None:
                continue
            y = VALUES[f].index(str(v).strip().upper())
            if (lg[i] > 0) != bool(y):
                wrong.append((key, f, round(float(lg[i]), 3)))
            if key.endswith("_sim"):
                m = abs(float(lg[i]))
                near05 += m < 0.05
                near1 += m < 1.0
                if m < min_margin[0]:
                    min_margin = (m, f"{key}:{f}")
    (OUT / "v2_logits.json").write_text(json.dumps(rows))
    return dict(n_images=len(rows), wrong=wrong, holdout_scored_near_0p05=int(near05),
                holdout_scored_near_1p0=int(near1), holdout_min_margin=[round(min_margin[0], 4), min_margin[1]])


# ------------------------------------------------------------------ B
class GraphRunner:
    """Replays the whole post-copy inference path (uint8 -> float -> resize -> normalise -> net) as one graph."""

    def __init__(self, eng):
        self.eng, dev = eng, eng.device
        self.u8 = torch.zeros((1, CACHE_H, CACHE_W, 3), dtype=torch.uint8, device=dev)
        self.mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                self.fwd()
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.g):
            self.out = self.fwd()

    def fwd(self):
        x = u8_to_float(self.u8)
        in_w, in_h = self.eng.in_wh
        if x.shape[-2:] != (in_h, in_w):
            x = F.interpolate(x, size=(in_h, in_w), mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        # cache_enabled=False: required for autocast inside graph capture; same casts, same numerics
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            logits = self.eng.net(x)
        return logits.float()

    @torch.inference_mode()
    def logits(self, image):
        self.u8.copy_(torch.from_numpy(prep(image)).unsqueeze(0))
        self.g.replay()
        return self.out.clone()


def check_b():
    eng = Engine(v2_dir, "cuda")
    eng.warmup()
    gr = GraphRunner(eng)
    keys = splits["holdout"][:60] + real_keys
    imgs = [load_rgb(key_path(k)[0]) for k in keys]
    eq_engine, max_diff, flips = 0, 0.0, 0
    for im in imgs:
        with torch.inference_mode():
            pe = torch.from_numpy(eng.probs(im))
        pg = torch.sigmoid(gr.logits(im))[0].cpu()
        eq_engine += bool(torch.equal(pe, pg))
        max_diff = max(max_diff, float((pe - pg).abs().max()))
        flips += int(((pe >= 0.5) != (pg >= 0.5)).sum())

    def timeit(fn, n=3):
        ts = []
        for _ in range(n):
            for im in imgs:
                torch.cuda.synchronize()
                t = time.perf_counter()
                fn(im)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
        return round(1000 * float(np.mean(ts)), 2), round(1000 * float(np.percentile(ts, 99)), 2)

    eager_ms = timeit(lambda im: eng.probs(im))
    graph_ms = timeit(lambda im: torch.sigmoid(gr.logits(im))[0].cpu().numpy())
    return dict(n=len(imgs), probs_bit_equal=f"{eq_engine}/{len(imgs)}", max_abs_prob_diff=max_diff,
                decision_flips=flips, eager_ms_mean_p99=eager_ms, graph_ms_mean_p99=graph_ms)


# ------------------------------------------------------------------ C
def dinov3(img_hw):
    name = "vit_base_patch16_dinov3.lvd1689m"
    try:
        m = timm.create_model(name, pretrained=True, num_classes=0, img_size=img_hw)
    except TypeError:
        m = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True)
    return m


def batch_for(hw, n_real=8, n_sim=8, mean=MEAN, std=STD):
    keys = real_keys[:n_real] + splits["holdout"][:n_sim]
    x = torch.stack([torch.from_numpy(prep(load_rgb(key_path(k)[0]))) for k in keys]).cuda()
    x = u8_to_float(x)
    x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False, antialias=True)
    mt = torch.tensor(mean, device="cuda").view(1, 3, 1, 1)
    st = torch.tensor(std, device="cuda").view(1, 3, 1, 1)
    return (x - mt) / st


def check_c():
    out = {}
    for hw in ((608, 1088), (544, 960)):
        m = dinov3(hw).cuda()
        cfg = m.pretrained_cfg
        x = batch_for(hw, mean=cfg.get("mean", MEAN), std=cfg.get("std", STD))
        maxima = []
        hooks = [b.register_forward_hook(lambda mod, i, o: maxima.append(float(o.detach().float().abs().max())))
                 for b in m.blocks]
        r = {"num_prefix_tokens": int(getattr(m, "num_prefix_tokens", -1)), "embed_dim": int(m.embed_dim),
             "mean_std": [cfg.get("mean"), cfg.get("std")]}
        m.eval()
        with torch.no_grad():
            f32 = m.forward_features(x.float())
            maxima_fp32 = list(maxima)
            maxima.clear()
            with torch.autocast("cuda", dtype=torch.float16):
                f16 = m.forward_features(x)
        r["tokens"] = list(f32.shape)
        r["fp32_block_absmax_max"] = round(max(maxima_fp32), 1)
        r["fp16_block_absmax_max"] = round(max(maxima), 1) if maxima else None
        r["fp16_forward_finite"] = bool(torch.isfinite(f16).all())
        cos = F.cosine_similarity(f16.float().flatten(1), f32.flatten(1), dim=1)
        r["fp16_vs_fp32_cos_min"] = round(float(cos.min()), 6)
        for h in hooks:
            h.remove()
        # backward under fp16 autocast with a 31-logit head, grad checkpointing on
        m.train()
        m.set_grad_checkpointing(True)
        head = nn.Linear(m.embed_dim, 31).cuda()
        opt = torch.optim.AdamW(list(m.parameters()) + list(head.parameters()), lr=1e-5)
        scaler = torch.amp.GradScaler("cuda")
        finite_steps = []
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.float16):
                feats = m.forward_features(x)
                logits = head(feats[:, 0])
            loss = F.binary_cross_entropy_with_logits(logits.float(), torch.zeros_like(logits.float()))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            finite_steps.append(bool(torch.isfinite(loss)) and bool(torch.isfinite(gn)))
            scaler.step(opt)
            scaler.update()
        r["fp16_train_steps_finite"] = finite_steps
        r["grad_scale_after"] = float(scaler.get_scale())
        # timing: batch 32 over 2 GPUs (DataParallel), grad checkpointing, AdamW
        r["s_per_step_bs32"] = time_train(m, head, hw, cfg)
        out[f"{hw[1]}x{hw[0]}"] = r
        del m, head, opt
        torch.cuda.empty_cache()
    return out


def time_train(backbone, head, hw, cfg, steps=6):
    class Wrap(nn.Module):
        def __init__(self, b, h):
            super().__init__()
            self.b, self.h = b, h

        def forward(self, x):
            f = self.b.forward_features(x) if hasattr(self.b, "forward_features") else self.b(x)
            return self.h(f[:, 0])

    net = Wrap(backbone, head).cuda().train()
    model = nn.DataParallel(net) if torch.cuda.device_count() > 1 else net
    opt = torch.optim.AdamW(net.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(32, 3, *hw, device="cuda")
    ts = []
    for s in range(steps + 2):
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        loss = logits.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        if s >= 2:
            ts.append(time.perf_counter() - t)
    return round(float(np.mean(ts)), 2)


# ------------------------------------------------------------------ D
def check_d():
    from transformers import Dinov2Config
    from isu_core import ISUNet  # BUNDLE-STRIP
    cfg = Dinov2Config.from_pretrained("facebook/dinov2-large", revision="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c")
    out = {}
    for hw, ckpt in (((364, 644), True), ((532, 952), True)):
        net = ISUNet(cfg)
        if ckpt:
            net.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        class B(nn.Module):  # adapter so time_train can call forward_features
            def __init__(self, n):
                super().__init__()
                self.n = n

            def forward(self, x):
                return self.n(x).unsqueeze(1).expand(-1, 1, -1)

        head = nn.Identity()
        out[f"dinov2_l_{hw[1]}x{hw[0]}_ckpt{int(ckpt)}"] = time_train(B(net), head, hw, {})
        del net
        torch.cuda.empty_cache()
    return out


run("A_v2_holdout_margins", check_a)
run("B_cuda_graph", check_b)
run("C_dinov3_b_fp16", check_c)
run("D_dinov2_l_timing", check_d)
log("done")
