## WAN 2.1 (T2V 1.3B) PTQ with DeepCompressor

This document shows how to quantize WAN 2.1 (text-to-video, 1.3B) to SVDQuant INT4 using DeepCompressor, and explains the WAN-specific adaptations in this repository.

### Quickstart

1) Generate the calibration set

```bash
python -m deepcompressor.app.diffusion.dataset.collect.calib \
  /workspace/pkgs/deepcompressor/examples/diffusion/configs/model/wan2.1-t2v-1.3b.yaml \
  /workspace/pkgs/deepcompressor/examples/diffusion/configs/collect/qdiff.yaml
```

- You can change how many samples to collect by editing `examples/diffusion/configs/collect/qdiff.yaml` (key: `collect.num_samples`).

2) Run SVDQuant PTQ and inference

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m deepcompressor.app.diffusion.ptq \
  /workspace/pkgs/deepcompressor/examples/diffusion/configs/model/wan2.1-t2v-1.3b.yaml \
  /workspace/pkgs/deepcompressor/examples/diffusion/configs/svdquant/int4.yaml \
  /workspace/pkgs/deepcompressor/examples/diffusion/configs/collect/overwrite_s32.yaml \
  --save-model true
```

- To choose a different calibration split, swap `overwrite_s32.yaml` for another file under `examples/diffusion/configs/collect/` (e.g., `overwrite_s4.yaml`). These overwrite files control `quant.calib.num_samples` and the cache `path` that PTQ loads from.
- After PTQ, the script automatically runs diffusers-based generation using the SVDQuant-quantized WAN pipeline (i.e., you will get outputs with the quantized model applied).

### Where to tweak calibration size and path

- Collect step: `examples/diffusion/configs/collect/qdiff.yaml` → `collect.num_samples` controls how many prompts are collected.
- PTQ step: `examples/diffusion/configs/collect/overwrite_sXXX.yaml` → sets `quant.calib.num_samples` and the calibration cache `path` that PTQ reads.

### What we adapted for WAN

WAN 2.1 is a video diffusion pipeline. The following changes ensure calibration, quantization, and inference work smoothly:

- Pipeline construction for WAN
  - WAN models are loaded via diffusers with `trust_remote_code` so custom pipeline classes are honored. The code then applies model "surgery" to make layers quantization-friendly, including replacing fused linears and certain convs with concat-compatible variants and optionally shifting activations for better range.

- Frames instead of images
  - WAN generation returns videos (frames), not single images. During calibration collection, the pipeline output is checked for `frames`; a thumbnail (first frame) is saved for reference, while the full set of intermediate tensors is captured via hooks for calibration.

- WAN-specific calibration hooks
  - Calibration attaches a forward hook on the core model (`transformer` for WAN). The hook normalizes inputs specifically for WAN’s 3D transformer: it moves `hidden_states` to the expected positional argument and broadcasts `timestep` to the batch. It then records per-step inputs/outputs that PTQ uses for range estimation and low-rank fitting.

- Automatic SVDQuant inference
  - After quantization (weights and activations, INT4), the same script runs generation/evaluation using the quantized pipeline, so you can immediately see outputs from SVDQuant WAN without another command.

### Key example configs

- Model: `examples/diffusion/configs/model/wan2.1-t2v-1.3b.yaml` (sets WAN path, eval resolution/steps, and default quant calib knobs)
- SVDQuant INT4: `examples/diffusion/configs/svdquant/int4.yaml` (enables W4A4 INT4 and activation shifting)
- Calibration overwrite examples: `examples/diffusion/configs/collect/overwrite_s4.yaml`, `overwrite_s32.yaml` (choose dataset size/path for PTQ)

### Notes

- Use `--save-model true` to save a quantized checkpoint; this can be converted and deployed via Nunchaku later if desired.
- Keep the WAN weights accessible in your environment as referenced by the model config (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`).


