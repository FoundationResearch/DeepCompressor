import os
import math
import torch
import torch.nn as nn

from model import TwoLayerMLP
from nunchaku.models.linear import SVDQW4A4Linear

DEBUG_PRINT = True

@torch.no_grad()
def truncated_svd_lowrank(weight: torch.Tensor, rank: int):
    # weight: (N, K) = (out_features, in_features)
    U, S, Vh = (x.to(weight.dtype) for x in torch.linalg.svd(weight.float(), full_matrices=False))

    r = min(rank, U.shape[1])
    U_r = U[:, :r].contiguous()
    S_r = S[:r].contiguous()
    V_r = Vh[:r, :].contiguous()  # (r, K)
    sr_sqrt = S_r.sqrt()
    lora_up = U_r * sr_sqrt.unsqueeze(0)        # (N, r)
    lora_down = V_r.t() * sr_sqrt.unsqueeze(0)  # (K, r)
    recon = (lora_up @ lora_down.t())           # (N, K)
    return lora_down, lora_up, recon


@torch.no_grad()
def compute_group_scales_sym_int4(residual: torch.Tensor, group_size: int = 64, dtype=torch.bfloat16):
    # residual: (N, K)
    N, K = residual.shape
    assert K % group_size == 0
    G = K // group_size
    res = residual.abs().reshape(N, G, group_size).amax(dim=-1)  # (N, G)
    scales = res.transpose(0, 1).clamp_min(1e-8).to(dtype)       # (G, N)
    return scales


@torch.no_grad()
def quantize_residual_to_int4(residual: torch.Tensor, scales: torch.Tensor, group_size: int = 64):
    # residual: (N, K); scales: (G, N)
    N, K = residual.shape
    G = K // group_size
    s_exp = scales.transpose(0, 1).repeat_interleave(group_size, dim=1)  # (N, K)
    q = (residual / s_exp).round().clamp_(-8, 7).to(torch.int8)
    return q  # (N, K)


@torch.no_grad()
def pack_int4_k_major(q: torch.Tensor) -> torch.Tensor:
    # q: (N, K) int8 in [-8,7]; pack along K: 2 int4 -> 1 int8
    assert q.dtype == torch.int8 and q.shape[1] % 2 == 0
    lo = (q[:, 0::2] & 0xF).to(torch.uint8)
    hi = (q[:, 1::2] & 0xF).to(torch.uint8)
    packed = (lo | (hi << 4)).to(torch.int8)
    return packed  # (N, K//2)


@torch.no_grad()
def convert_linear_to_svdq(
    linear: nn.Linear,
    *,
    rank: int = 32,
    precision: str = "int4",
    act_unsigned: bool = False,
):
    assert precision in ("int4",), "This test converter focuses on INT4 (W4A4)."
    in_features = linear.in_features
    out_features = linear.out_features
    assert in_features % 64 == 0, "INT4 requires in_features to be divisible by 64."

    torch_dtype = torch.bfloat16 if linear.weight.dtype == torch.bfloat16 else torch.float16
    device = linear.weight.device

    # 1) Low-rank branch via truncated SVD
    r = min(rank, in_features, out_features)
    lora_down, lora_up, recon = truncated_svd_lowrank(linear.weight.data, rank=r)

    # 2) Symmetric INT4 quantization on residual, grouped by K every 64
    residual = (linear.weight.data - recon).contiguous()
    wscales = compute_group_scales_sym_int4(residual, group_size=64, dtype=torch_dtype)  # (K//64, N)
    q = quantize_residual_to_int4(residual, wscales, group_size=64)                      # (N, K)
    qweight = pack_int4_k_major(q)                                                       # (N, K//2)

    # 3) Build SVDQ layer and fill parameters
    svdq = SVDQW4A4Linear.from_linear(
        linear,
        rank=r,
        precision=precision,
        act_unsigned=act_unsigned,
    )
    svdq.qweight.copy_(qweight)
    svdq.wscales.copy_(wscales)
    svdq.proj_down.copy_(lora_down.to(torch_dtype))
    svdq.proj_up.copy_(lora_up.to(torch_dtype))
    if linear.bias is not None:
        svdq.bias.copy_(linear.bias.data.to(torch_dtype))
    svdq.smooth_factor.fill_(1)
    svdq.smooth_factor_orig.fill_(1)
    
    if DEBUG_PRINT:
        # print original weight shape and svdq packed weight shape
        print(f"original weight shape: {linear.weight.shape}")
        print(f"svdq weight shape: {svdq.qweight.shape}")  # (N, K//2) packed int4
        print(f"wscales shape: {svdq.wscales.shape}")      # (K//64, N)
        # print project up and down shape
        print(f"project up shape: {svdq.proj_up.shape}")
        print(f"project down shape: {svdq.proj_down.shape}")
    
    return svdq


@torch.no_grad()
def replace_module_linear_with_svdq(module: nn.Module, ranks: dict[str, int] | None = None) -> nn.Module:
    ranks = ranks or {}
    for name, child in list(module.named_children()):
        full_name = name
        if isinstance(child, nn.Linear):
            # Heuristic: first layer before ReLU -> act_unsigned=False; after ReLU -> act_unsigned=True
            r = ranks.get(full_name, 32)
            act_unsigned = (full_name != "layer1")  # layer2 gets unsigned activations after ReLU
            setattr(module, name, convert_linear_to_svdq(child, rank=r, act_unsigned=act_unsigned))
        else:
            replace_module_linear_with_svdq(child, ranks=ranks)
    return module


def quantize_and_save(
    ckpt_in: str = "./ckpt/mlp_demo.pt",
    ckpt_out: str = "./ckpt/mlp_demo_svdq.pt",
    device: str | torch.device = "cuda",
):
    # Load BF16 model checkpoint
    ckpt = torch.load(ckpt_in, map_location=device)
    cfg = ckpt["config"]

    # Build FP16/BF16 model and load weights
    model = TwoLayerMLP(**cfg).to(device).to(torch.bfloat16)
    model.load_state_dict(ckpt["state_dict"])  # trained bf16 weights
    model.eval()

    # Replace nn.Linear with SVDQW4A4Linear
    qmodel = replace_module_linear_with_svdq(model, ranks={"layer1": 16, "layer2": 16}).to(device).eval()

    os.makedirs(os.path.dirname(ckpt_out) or ".", exist_ok=True)
    torch.save(
        {
            "config": cfg,
            "svdq": True,
            "precision": "int4",
            "rank": {"layer1": 16, "layer2": 16},
            "state_dict": qmodel.state_dict(),
        },
        ckpt_out,
    )
    print(f"[quantize] SVDQ W4A4 checkpoint saved to {ckpt_out}")
    return ckpt_out


if __name__ == "__main__":
    quantize_and_save()


