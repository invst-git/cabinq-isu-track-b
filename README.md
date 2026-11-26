# CabinQ: ISU-Challenge @ ICSE 2027, Track B

CabinQ answers the 31 Track B questions about a car cabin (occupants, seat belts, phone use,
clothing color, emotion, objects) from one RGB image in a single forward pass. A DINOv2-B/14
backbone reads the image at 952×532 pixels and one learned query per question attends to the
image tokens. Training has two stages: synthetic renders only, then renders mixed with the 30
public real photos and their synthetic twins, once per cross-validation fold; the five fold
models are averaged into the submitted model. The method and results are described in our
ISU-Challenge report.

Author: Dandu Venkata Nitish Chandra Teja (Amrita Vishwa Vidyapeetham, Coimbatore, India).

## Official results (NVIDIA L4)

| Variant | Pull request | Commit | Correct (of 10,749) | Real (of 479) | Synthetic (of 10,270) | Latency |
|---|---|---|---|---|---|---|
| v1 | [#34](https://github.com/isu-challenge/Starterkit-ICSE-2027/pull/34) | `8a05f77` | 10,747 | 477 | 10,270 | 32 ms |
| **v2 (final)** | [#35](https://github.com/isu-challenge/Starterkit-ICSE-2027/pull/35) | `079c843` | **10,748** | **478** | 10,270 | 41 ms |
| v2 + v2s1 | [#38](https://github.com/isu-challenge/Starterkit-ICSE-2027/pull/38) | `4bc2a56` | 10,748 | 478 | 10,270 | 67 ms |
| v2 + v4 | [#39](https://github.com/isu-challenge/Starterkit-ICSE-2027/pull/39) | `b47bc3f` | 10,748 | 478 | 10,270 | 67 ms |

Source: the evaluation comments on the pull requests, saved in
[`analysis/official_results.json`](analysis/official_results.json).

## Variants and provenance

| Variant | What changes (all else as v2) | Kaggle kernel / bundle digest | Run artefacts | Weights (release asset) |
|---|---|---|---|---|
| v1 | input 644×364, 12 stage-1 epochs | `isu-track-b-train` / `e804f64b5422` | `work/train_v1/` | `cabinq-v1.safetensors` |
| v2 | final: input 952×532, 6 stage-1 epochs | `isu-track-b-train-v2` / `e08208eecd98` | `work/train_v2/` | `cabinq-v2.safetensors` |
| v2s1 | v2 with training seed 1 (same data splits) | `isu-track-b-train-v2s1` / `511138d8a9d2` | `work/train_v2s1/` | `cabinq-v2s1.safetensors` |
| v4 | DINOv3-B/16 backbone at 1088×608, 3 stage-1 epochs | `isu-track-b-train-v4` / `a5320aad0c2b` | `work/train_v4/` | `cabinq-v4.safetensors` |

A planned v3 (DINOv2-L) was never trained. Ensembles (`v2 + v4`) average the logits of their members.

- `kaggle/isu-train*/train_kernel.py` are the exact scripts that ran on Kaggle: each was generated
  by `kaggle/build_train_kernel.py` less than a minute before its run started and is unchanged since.
  The digest in their first line is the SHA-256 prefix of the bundled sources.
- `provenance/pr*/` holds the exact code, Dockerfile and weight metadata of every scored submission;
  `provenance/submission_log.txt` lists the SHA-256 of every submitted file, including the weights.
- `submission/` is the final system: the current code (which adds ensemble support and produces
  outputs identical to the code of PR #35 for a single model) with v2's weight metadata.

## Repository layout

```
submission/    final system: system.py (evaluation entry point), isu_core.py (model, preprocessing,
               inference engine shared with training), Dockerfile, weights/ (config + metadata; weights via release)
train/         train_isu.py: stage 1, stage 2 per fold, fold soup, reports (VARIANTS defines v1-v5)
kaggle/        build_train_kernel.py, build_check_kernel.py (bundle code into Kaggle script kernels),
               isu-data/ (dataset kernel), isu-train*/ (exact training bundles), isu-checks/, isu-enscheck/
tools/         download_data.py, fetch_weights.py, check_weights.py, ensemble_cv.py, ens_combos.py,
               build_ensemble_submission.py, harness_system.py, container_test.py, dump_probs.py,
               kaggle_checks.py, kaggle_ens_check.py, emotion_expert_gate.py, crop_geometry.py, analyze_labels.py
work/          run artefacts: train_<variant>/{reports.json, splits.json, train_log.txt, Kaggle log, soup/meta.json},
               enscheck/ (synthetic hold-out logits of v2, v2s1, v4), checks_0d/, label_profile.txt, priors.json
provenance/    code of every scored submission and the hash log
analysis/      compute_metrics.py (recomputes every number in the report from work/), metrics.json, official_results.json
```

## Setup

Python 3.11 or 3.12. CPU is enough for everything except training.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Data.** The ISU-Challenge dataset is on Hugging Face (`ISU-Test/isu-challenge-dataset`) and is not
redistributed here. `tools/download_data.py` pins the dataset commits used in all experiments and stops
if any withheld real photo (`sample_00030`-`00059`) were present.

```bash
python -I tools/download_data.py --out data --mode local   # ~115 MB: labels, the 30 real photos, 97 renders
python -I tools/download_data.py --out data --mode full    # ~4.2 GB: all renders, needed only for training
```

**Weights.** The trained models (about 350 MB each) are attached to the GitHub release `weights-v1`.
`tools/fetch_weights.py` downloads them, checks the SHA-256 against the submission log and copies them
into place; a model that is not attached to the release (e.g. v4) is skipped. For a private repository,
download with the GitHub CLI first:

```bash
python -I tools/fetch_weights.py                                   # public repository
gh release download weights-v1 -R invst-git/cabinq-isu-track-b -D weights_dl
python -I tools/fetch_weights.py --from-dir weights_dl              # private repository
```

## Run the final system

**Local evaluation (CPU).** Clone the starter kit and run the official Track B evaluator on the local data:

```bash
git clone https://github.com/isu-challenge/Starterkit-ICSE-2027
python -I -X utf8 tools/harness_system.py submission Starterkit-ICSE-2027/track_b data/main runs/eval.json both
```

The harness also imitates the preflight check (one scene, a prompt listing only that scene's options)
and feeds malformed inputs to `predict()`. The photos in `data/main` were used in training, so this run
checks the plumbing, not generalization.

**Weight parity.** `tools/check_weights.py` re-scores the 30 real photos and the local hold-out renders with
the released weights and compares them with the errors reported by the Kaggle run:

```bash
python -I tools/check_weights.py --weights work/train_v2/soup --reports work/train_v2/reports.json \
    --splits work/train_v2/splits.json --data data/main
```

**Docker.** Build the official base image from the starter kit (`docker build -t isu-challenge-base docker/`),
then `docker build -t cabinq submission/`. `tools/container_test.py` runs the evaluator inside the image
with no network, a read-only file system and a non-root user; its docstring contains the full command.

**Submitting an ensemble.** `tools/build_ensemble_submission.py --out <dir> work/train_v2/soup work/train_v4/soup`
writes a submission folder with `weights/ensemble.json` and adds a pinned `timm` install to the Dockerfile
when a DINOv3 member is present. Submissions are encrypted with the starter kit's `tools/encrypt_submission.py`.

## Reproduce training

**On Kaggle (as run, two NVIDIA T4 GPUs, about 4 h per variant).**

1. Dataset kernel: in `kaggle/isu-data/kernel-metadata.json`, replace `nitishxdandu` with your Kaggle
   user name, then `kaggle kernels push -p kaggle/isu-data` (CPU, internet on). Its output holds
   `isu_main.tar` and `isu_extended.tar`.
2. Training kernel. To rerun exactly what ran, edit the `id` and `kernel_sources` in
   `kaggle/isu-train-v2/kernel-metadata.json` and push that folder:
   `kaggle kernels push -p kaggle/isu-train-v2 --accelerator NvidiaTeslaT4`.
   To train from the current sources instead (same v2 configuration), regenerate the bundle:
   `python -I kaggle/build_train_kernel.py --transformers 4.57.6 --slug isu-track-b-train-v2 --variant v2 --user <you> --data-kernel isu-track-b-data`
   (this overwrites the committed bundle folder).
3. Download the outputs: `kaggle kernels output <you>/isu-track-b-train-v2 -p work/train_v2`, then run the
   parity check above.

**On a local GPU.** `train_isu.py` reads a data root with `main/` and `extended/` (from `--mode full`):

```bash
PYTHONPATH=submission python train/train_isu.py --variant v2 --data data --out runs/v2
PYTHONPATH=submission python train/train_isu.py --variant v2 --data data --out runs/dry --dry-run   # CPU smoke test
```

A real run needs Linux (or WSL), a CUDA GPU and about 10 GB free in `/tmp`, where it caches all
images as uint8 at 960×540 (9.4 GB); the dry run writes its small cache to `--out`. The runs reported
here were made only on Kaggle (2×T4, DataParallel, fp16, gradient checkpointing); other GPUs were not tested.

## Reproduce the development results

```bash
# real-photo cross-validation of single variants and logit-averaged ensembles (needs data/main/labels)
python -I tools/ensemble_cv.py work/train_v2/reports.json work/train_v4/reports.json \
    --splits work/train_v2/splits.json work/train_v4/splits.json
# synthetic hold-out errors and margins of every combination of v2, v2s1 and v4 (needs data/*/labels)
python -I tools/ens_combos.py work/enscheck/ens_logits.json --data data
```

`work/enscheck/ens_logits.json` holds the per-member logits of the 500 hold-out renders, computed on a T4 by
the kernel in `kaggle/isu-enscheck/` (`kaggle/build_check_kernel.py --script tools/kaggle_ens_check.py`).

## Recompute the numbers in the report

```bash
python -I analysis/compute_metrics.py   # needs the dataset (--mode local) and the weights
```

`analysis/metrics.json` holds every number, table entry and plotted value in the report. The script
recomputes it from the run artefacts in `work/`, the released weights and `analysis/official_results.json`
(the official results copied from the pull-request comments); a rerun reproduces the committed file.

## Reproducibility notes

- Seeds and data splits are fixed (`splits.json` is identical for all variants), and the library versions of
  the model code are pinned (Transformers 4.57.6, timm 1.0.31). Kaggle supplied PyTorch 2.11; the
  evaluation image uses PyTorch 2.6. The parity check gave identical errors for every released model.
- GPU training is not bit-deterministic. Retraining reproduces the procedure, not necessarily the exact
  error counts: v2 and v2s1 differ only in their seed and make 1 and 6 cross-validation errors.
  The reported numbers come from the released weights and the saved `reports.json` files.
- Training used only the 30 public real photos (`sample_00000`-`00029`). The withheld photos were never
  downloaded. Their scenes' renders are part of the published synthetic data, and 24 of these 30 renders
  were in the training set.

## Third-party components

- ISU-Challenge dataset: MIT license (see the dataset card).
- DINOv2 (`facebook/dinov2-base`, Apache-2.0) is the backbone of v1, v2 and v2s1.
- v4 uses DINOv3 (`vit_base_patch16_dinov3.lvd1689m` via timm) and is subject to Meta's DINOv3 license.
- Hugging Face Transformers and timm (Apache-2.0).
- The facial-expression experiment (`tools/emotion_expert_gate.py`) uses the YuNet face detector
  (`face_detection_yunet_2023mar.onnx`, OpenCV Zoo) and the HSEmotion/EmotiEffLib model `enet_b2_8.onnx`.
  They are not part of the submitted system and not included; place them in `work/experts/` to rerun it.

## Citation

The dataset and benchmark:

```bibtex
@inproceedings{sorokin2026isutest,
  author    = {Lev Sorokin and Chen Yang and Ken E. Friedl and Andrea Stocco},
  title     = {Search-based Testing of Vision Language Models for In-Car Scene Understanding},
  booktitle = {Proceedings of the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE 2026), Industry Track},
  year      = {2026},
  doi       = {10.1145/3832783.3834506}
}
```
