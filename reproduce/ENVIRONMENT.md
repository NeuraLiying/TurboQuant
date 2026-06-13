# Environment Notes

Created: 2026-06-10

## Active Reproduction Environment

- Conda env: `turboquant`
- Creation method used today: cloned from existing validated env `layer_skip`
- Reason: creating directly from the first `environment.yml` stalled during pip installation because PyTorch wheels were specified without the `+cu121` build tag and CUDA wheel index.

Command used:

```bash
conda create -n turboquant --clone layer_skip -y
```

Validation command:

```bash
conda run -n turboquant python scripts/inspect_assets.py \
  --paths configs/paths.yaml \
  --output reproduce/logs/asset_report_turboquant_env.json
```

## Validated Package Versions

- `torch`: 2.2.1+cu121
- `torchvision`: 0.17.1+cu121
- `torchaudio`: 2.2.1+cu121
- `transformers`: 4.53.0
- `datasets`: 3.6.0
- `accelerate`: 1.8.1
- `flash-attn`: 2.5.6
- `pyarrow`: 19.0.0
- `numpy`: 1.26.4
- `scipy`: 1.15.1

## GPU Validation

- CUDA visible: yes
- Visible GPUs: 8 x NVIDIA GeForce RTX 4090
- Initial free GPUs from `asset_report_turboquant_env.json`: GPU 0, 1, 4, 5 were mostly idle at validation time.

## Recreate From Spec

The project `environment.yml` has been updated to use CUDA 12.1 PyTorch wheel tags and the PyTorch CUDA wheel index. For exact continuity, cloning `layer_skip` remains the fastest known-good option on this machine.
