FROM isu-challenge-base

# torch, torchvision, transformers, safetensors, Pillow and numpy come from the base image.
# The submission folder (system.py, isu_core.py, weights/) is mounted read-only at /submission.
RUN python -c "import torch, transformers, safetensors; print('torch', torch.__version__, '| transformers', transformers.__version__)"
