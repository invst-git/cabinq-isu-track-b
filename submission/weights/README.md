# Weights of the final system (v2)

`config.json` (DINOv2-B backbone configuration) and `meta.json` (features, answer coding, input size,
training configuration) are versioned here. `model.safetensors` (356 MB, SHA-256
`8031f1b441dd850826c660050ec3dc90fb30843889a0017375e4885f94f8aa64`) is attached to the GitHub release
`weights-v1` as `cabinq-v2.safetensors`; `python -I tools/fetch_weights.py --only v2` downloads, verifies
and places it here.
