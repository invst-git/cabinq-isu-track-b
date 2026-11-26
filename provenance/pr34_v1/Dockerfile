FROM isu-challenge-base

# Everything the system needs (torch, transformers, safetensors, Pillow, numpy) is already in
# the base image; installing nothing keeps its torch/CUDA build intact. The submission folder
# (system.py, isu_core.py, weights/) is mounted read-only at /submission during evaluation.
RUN python -c "import torch, transformers, safetensors; print('torch', torch.__version__, '| transformers', transformers.__version__)"
