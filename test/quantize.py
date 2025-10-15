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
def compute_group_scales_sym_int4(
    residual: torch.Tensor,
    group_size: int = 64,
    dtype=torch.bfloat16,
    percentile: float | None = None,
):
    # residual: (N, K)
    N, K = residual.shape
    assert K % group_size == 0
    G = K // group_size
    res_block = residual.abs().reshape(N, G, group_size)  # (N, G, gs)
    if percentile is None or percentile >= 1.0:
        res = res_block.amax(dim=-1)  # (N, G)
    else:
        # per-(out,group) robust scale using percentile over group dimension
        q = torch.quantile(res_block.to(torch.float32), percentile, dim=-1)
        res = q.to(residual.dtype)
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
    w_percentile: float | None = 0.999,
    smooth_factor: torch.Tensor | None = None,
):
    assert precision in ("int4",), "This test converter focuses on INT4 (W4A4)."
    in_features = linear.in_features
    out_features = linear.out_features
    assert in_features % 64 == 0, "INT4 requires in_features to be divisible by 64."

    torch_dtype = torch.bfloat16 if linear.weight.dtype == torch.bfloat16 else torch.float16
    device = linear.weight.device

    # 1) Low-rank branch via truncated SVD; align rank to multiples of 16 for kernel
    r_base = min(rank, in_features, out_features)
    def _align16(x: int) -> int:
        return ((max(1, x) + 15) // 16) * 16
    r_aligned = _align16(r_base)
    lora_down_b, lora_up_b, recon = truncated_svd_lowrank(linear.weight.data, rank=r_base)
    if r_aligned != r_base:
        lora_down = torch.zeros(in_features, r_aligned, dtype=lora_down_b.dtype, device=lora_down_b.device)
        lora_up = torch.zeros(out_features, r_aligned, dtype=lora_up_b.dtype, device=lora_up_b.device)
        lora_down[:, :r_base] = lora_down_b
        lora_up[:, :r_base] = lora_up_b
    else:
        lora_down, lora_up = lora_down_b, lora_up_b

    # 2) Symmetric INT4 quantization on residual, grouped by K every 64 (percentile-based)
    residual = (linear.weight.data - recon).contiguous()
    wscales = compute_group_scales_sym_int4(
        residual,
        group_size=64,
        dtype=torch_dtype,
        percentile=w_percentile,
    )  # (K//64, N)
    q = quantize_residual_to_int4(residual, wscales, group_size=64)                      # (N, K)
    qweight = pack_int4_k_major(q)                                                       # (N, K//2)

    # 3) Build SVDQ layer and fill parameters
    svdq = SVDQW4A4Linear.from_linear(
        linear,
        rank=r_aligned,
        precision=precision,
        act_unsigned=act_unsigned,
    )
    svdq.qweight.copy_(qweight)
    svdq.wscales.copy_(wscales)
    svdq.proj_down.copy_(lora_down.to(torch_dtype))
    svdq.proj_up.copy_(lora_up.to(torch_dtype))
    if linear.bias is not None:
        svdq.bias.copy_(linear.bias.data.to(torch_dtype))
    if smooth_factor is not None:
        svdq.smooth_factor.copy_(smooth_factor.to(torch_dtype, non_blocking=True))
        svdq.smooth_factor_orig.copy_(smooth_factor.to(torch_dtype, non_blocking=True))
    else:
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
def replace_module_linear_with_svdq(
    module: nn.Module,
    ranks: dict[str, int] | None = None,
    w_percentile: float | None = 0.999,
    smooth_map: dict[str, torch.Tensor] | None = None,
) -> nn.Module:
    ranks = ranks or {}
    for name, child in list(module.named_children()):
        full_name = name
        if isinstance(child, nn.Linear):
            # Heuristic: first layer before ReLU -> act_unsigned=False; after ReLU -> act_unsigned=True
            r = ranks.get(full_name, 32)
            act_unsigned = (full_name != "layer1")  # layer2 gets unsigned activations after ReLU
            setattr(
                module,
                name,
                convert_linear_to_svdq(
                    child,
                    rank=r,
                    act_unsigned=act_unsigned,
                    w_percentile=w_percentile,
                    smooth_factor=(smooth_map.get(full_name) if smooth_map else None),
                ),
            )
        else:
            replace_module_linear_with_svdq(child, ranks=ranks, w_percentile=w_percentile, smooth_map=smooth_map)
    return module


@torch.no_grad()
def collect_layer_inputs(model: nn.Module, inputs: torch.Tensor, device: str | torch.device) -> dict[str, torch.Tensor]:
    """Collect pre-activation inputs for each nn.Linear child as calibration data."""
    buffers: dict[str, list[torch.Tensor]] = {}

    def pre_hook(name):
        def hook(mod, inp):
            x = inp[0]
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            buffers.setdefault(name, []).append(x.detach())
        return hook

    handles = []
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            handles.append(child.register_forward_pre_hook(pre_hook(name)))
        else:
            # recurse
            for subname, subchild in child.named_children():
                if isinstance(subchild, nn.Linear):
                    handles.append(subchild.register_forward_pre_hook(pre_hook(subname)))

    _ = model(inputs)
    for h in handles:
        h.remove()

    out: dict[str, torch.Tensor] = {}
    for k, vs in buffers.items():
        out[k] = torch.cat(vs, dim=0)
    return out


@torch.no_grad()
def compute_smooth_factors(
    model_fp: nn.Module,
    calib_inputs: torch.Tensor,
    alpha: float = 0.5,
    clamp_exp: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Compute SmoothQuant per-channel factors for each Linear input.

    s_i = ((A_i)/(W_i + eps))^alpha, then normalized to geometric mean 1, then clamped to [2^-clamp_exp, 2^clamp_exp].
    """
    model_fp.eval()
    layer_inputs = collect_layer_inputs(model_fp, calib_inputs, calib_inputs.device)
    smooth: dict[str, torch.Tensor] = {}
    for name, child in model_fp.named_children():
        if isinstance(child, nn.Linear):
            x = layer_inputs.get(name)
            if x is None:
                continue
            A = x.abs().amax(dim=0).to(torch.float32) + 1e-8
            W = child.weight.abs().amax(dim=0).to(torch.float32) + 1e-8
            s = (A / W).pow(alpha)
            # normalize geometric mean to 1
            gm = torch.exp(torch.mean(torch.log(s)))
            s = s / gm
            # clamp to 2^[−clamp_exp, +clamp_exp]
            lo, hi = 2.0 ** (-clamp_exp), 2.0 ** (clamp_exp)
            s = s.clamp(min=lo, max=hi)
            smooth[name] = s.to(calib_inputs.device)
        else:
            for subname, subchild in child.named_children():
                if isinstance(subchild, nn.Linear):
                    x = layer_inputs.get(subname)
                    if x is None:
                        continue
                    A = x.abs().amax(dim=0).to(torch.float32) + 1e-8
                    W = subchild.weight.abs().amax(dim=0).to(torch.float32) + 1e-8
                    s = (A / W).pow(alpha)
                    gm = torch.exp(torch.mean(torch.log(s)))
                    s = s / gm
                    lo, hi = 2.0 ** (-clamp_exp), 2.0 ** (clamp_exp)
                    s = s.clamp(min=lo, max=hi)
                    smooth[subname] = s.to(calib_inputs.device)
    return smooth


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

    # === Calibration ===
    torch.manual_seed(123)
    calib_bs = 2048
    x_calib = torch.randn(calib_bs, cfg["in_features"], dtype=torch.bfloat16, device=device)
    # baseline outputs for objective
    with torch.inference_mode():
        y_ref = model(x_calib)

    # SmoothQuant factors (alpha=0.5, clamp 2^[-2,2])
    smooth_map = compute_smooth_factors(model, x_calib, alpha=0.5, clamp_exp=2.0)

    # Small grid over weight percentile
    candidate_ps = [0.999, 1.0]
    best_p, best_loss, best_state = None, float("inf"), None
    ranks_cfg = {"layer1": 32, "layer2": 32}
    for p in candidate_ps:
        cand = TwoLayerMLP(**cfg).to(device).to(torch.bfloat16)
        cand.load_state_dict(ckpt["state_dict"]).eval()
        cand = replace_module_linear_with_svdq(
            cand, ranks=ranks_cfg, w_percentile=p, smooth_map=smooth_map
        ).to(device).eval()
        with torch.inference_mode():
            y_q = cand(x_calib.unsqueeze(0)) if x_calib.ndim == 2 else cand(x_calib)
            if x_calib.ndim == 2:
                y_q = y_q.squeeze(0)
        loss = torch.mean((y_q - y_ref).float().pow(2)).item()
        if DEBUG_PRINT:
            print(f"[calib] percentile={p} MSE={loss:.6f}")
        if loss < best_loss:
            best_loss = loss
            best_p = p
            best_state = cand.state_dict()

    # Build final quantized model with best hyperparams
    qmodel = TwoLayerMLP(**cfg).to(device).to(torch.bfloat16)
    qmodel = replace_module_linear_with_svdq(qmodel, ranks=ranks_cfg, w_percentile=best_p, smooth_map=smooth_map)
    if best_state is not None:
        qmodel.load_state_dict(best_state)

    # Optional: save to disk
    if ckpt_out:
        os.makedirs(os.path.dirname(ckpt_out) or ".", exist_ok=True)
        torch.save(
            {
                "config": cfg,
                "svdq": True,
                "precision": "int4",
                "rank": ranks_cfg,
                "w_percentile": best_p,
                "smooth_alpha": 0.5,
                "smooth_clamp_exp": 2.0,
                "state_dict": qmodel.state_dict(),
            },
            ckpt_out,
        )
        print(f"[quantize] SVDQ W4A4 checkpoint saved to {ckpt_out}")

    # Direct inference without saving/loading
    torch.manual_seed(42)
    x = torch.randn(1024, cfg["in_features"], dtype=torch.bfloat16, device=device)
    # Ground truth: sum along features if out_features==1; otherwise first head tracks sum, second head = -sum
    if cfg.get("out_features", 1) == 1:
        y = torch.sum(x, dim=1, keepdim=True)
    else:
        s = torch.sum(x, dim=1, keepdim=True)
        y = torch.zeros(x.shape[0], cfg["out_features"], dtype=torch.bfloat16, device=device)
        y[:, 0:1] = s
        if cfg["out_features"] > 1:
            y[:, 1:2] = -s

    # SVDQ linear expects (B, S, C) — if 2D, lift to (1, N, C)
    squeeze_back = False
    if x.ndim == 2:
        x = x.unsqueeze(0)
        squeeze_back = True
    with torch.inference_mode():
        pred = qmodel(x.unsqueeze(0)) if x.ndim == 2 else qmodel(x)
        if x.ndim == 2:
            pred = pred.squeeze(0)
        if squeeze_back:
            pred = pred.squeeze(0)
    loss = nn.MSELoss()(pred, y)
    print(f"[quantize][direct infer] loss={loss.item():.6f}")
    for i in range(min(5, x.shape[0] if not squeeze_back else pred.shape[0])):
        if cfg.get("out_features", 1) == 1:
            print(f"  GT sum={y[i].item():.4f} | Pred={pred[i].item():.4f}")
        else:
            print(f"  GT: [{y[i,0].item():.4f}, {y[i,1].item():.4f}] | Pred: [{pred[i,0].item():.4f}, {pred[i,1].item():.4f}]")
    return ckpt_out


if __name__ == "__main__":
    quantize_and_save()


