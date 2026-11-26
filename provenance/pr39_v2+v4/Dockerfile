FROM isu-challenge-base

# torch, torchvision, transformers, safetensors, Pillow and numpy come from the base image.
# The submission folder (system.py, isu_core.py, weights/) is mounted read-only at /submission.
# timm builds the DINOv3 member; --no-deps keeps the base image's torch/torchvision build intact.
RUN pip install --no-cache-dir --no-deps timm==1.0.31
RUN python -c "import torch, transformers, safetensors, timm; print('torch', torch.__version__, '| transformers', transformers.__version__, '| timm', timm.__version__)"
