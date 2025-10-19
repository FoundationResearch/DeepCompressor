import os
import sys
import math
import torch
import torch.nn as nn
import time

from model import TwoLayerMLP
from nunchaku.models.linear import SVDQW4A4Linear

# add repository root to sys.path so local 'deepcompressor' can be imported
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from deepcompressor.backend.nunchaku.utils import NunchakuWeightPacker, convert_to_nunchaku_w4x4y16_linear_weight

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
        res = res_block.amax(dim=-1) / 7.0  # (N, G)
    else:
        # per-(out,group) robust scale using percentile over group dimension
        q = torch.quantile(res_block.to(torch.float32), percentile, dim=-1)
        res = (q / 7.0).to(residual.dtype)
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
def pack_with_nunchaku_layout(q_int4: torch.Tensor, scales: torch.Tensor, group_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack weights and scales to Nunchaku MMA layout.

    Args:
        q_int4: (N, K) int8 in [-8, 7]
        scales: (K//group_size, N) bf16/fp16
        group_size: int, default 64 for INT4

    Returns:
        qweight_packed: (N, K//2) int8 but arranged in Nunchaku layout
        wscales_packed: (K//group_size, N) same dtype as input, arranged for kernel access
    """
    assert q_int4.dtype == torch.int8
    N, K = q_int4.shape
    assert K % group_size == 0 and scales.shape == (K // group_size, N)
    ng = K // group_size
    # Nunchaku packer expects (n=out_features, 1, ng, 1) for scales before padding/packing
    scales_4d = scales.t().reshape(N, 1, ng, 1)
    packer = NunchakuWeightPacker(bits=4)
    # Ensure weight tile alignment if needed
    q_i32 = q_int4.to(torch.int32)
    # For robustness, pad to required tiles before packing
    q_i32 = packer.pad_weight(q_i32)
    qweight_packed = packer.pack_weight(q_i32)
    scales_4d = packer.pad_scale(scales_4d.to(dtype=scales.dtype), group_size=group_size)
    wscales_packed = packer.pack_scale(scales_4d, group_size=group_size)
    # Final dtypes/shapes: int8 and bf16/fp16, (N, K//2) and (K//group_size, N)
    return qweight_packed, wscales_packed


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

    # 1) Apply SmoothQuant to weights (migrate outliers):
    #    kernel divides activations by s => to preserve y = x W^T,
    #    we must scale weights by s: W_hat = (s) * W
    if smooth_factor is not None:
        s = smooth_factor.view(1, in_features).to(linear.weight.device, dtype=linear.weight.dtype)
    else:
        s = torch.ones(1, in_features, device=linear.weight.device, dtype=linear.weight.dtype)
    W_hat = (linear.weight.data * s).contiguous()

    # 2) Low-rank branch via truncated SVD on W_hat; align rank to multiples of 16 for kernel
    r_base = min(rank, in_features, out_features)
    def _align16(x: int) -> int:
        return ((max(1, x) + 15) // 16) * 16
    r_aligned = _align16(r_base)
    lora_down_b, lora_up_b, recon = truncated_svd_lowrank(W_hat, rank=r_base)
    if r_aligned != r_base:
        lora_down = torch.zeros(in_features, r_aligned, dtype=lora_down_b.dtype, device=lora_down_b.device)
        lora_up = torch.zeros(out_features, r_aligned, dtype=lora_up_b.dtype, device=lora_up_b.device)
        lora_down[:, :r_base] = lora_down_b
        lora_up[:, :r_base] = lora_up_b
    else:
        lora_down, lora_up = lora_down_b, lora_up_b

    # 3) Build packed tensors via official Nunchaku converter (handles residual, scales, low-rank, smooth, bias)
    residual = (W_hat - recon).contiguous()
    wscales = compute_group_scales_sym_int4(
        residual,
        group_size=64,
        dtype=torch_dtype,
        percentile=w_percentile,
    )  # (K//64, N)
    ng = in_features // 64
    wscales_4d = wscales.t().reshape(out_features, 1, ng, 1)
    # smooth [K] -> [K,1]
    if smooth_factor is not None:
        sm_2d = smooth_factor.view(-1, 1).to(torch_dtype, non_blocking=True)
    else:
        sm_2d = torch.ones(in_features, 1, dtype=torch_dtype, device=device)
    # bias [N] -> [N,1]
    if linear.bias is not None:
        b_2d = linear.bias.view(-1, 1).to(torch_dtype)
    else:
        b_2d = torch.zeros(out_features, 1, dtype=torch_dtype, device=device)
    # low-rank tuple
    lora_tuple = (lora_down.to(torch_dtype), lora_up.to(torch_dtype))
    print(f"[quantize] lora_tuple shape: {lora_tuple[0].shape}, {lora_tuple[1].shape}")
    qweight, wscales_packed, bias_packed, smooth_packed, lora_packed, _ = convert_to_nunchaku_w4x4y16_linear_weight(
        weight=residual.to(torch_dtype),
        scale=wscales_4d.to(torch_dtype),
        bias=b_2d,
        smooth=sm_2d,
        lora=lora_tuple,
        float_point=False,
        subscale=None,
    )
    print(f"[quantize] qweight shape: {qweight.shape}")
    print(f"[quantize] wscales_packed shape: {wscales_packed.shape}, dtype={wscales_packed.dtype}")
    print(f"[quantize] bias_packed shape: {bias_packed.shape}, dtype={bias_packed.dtype}")
    print(f"[quantize] smooth_packed shape: {smooth_packed.shape}, dtype={smooth_packed.dtype}")
    print(f"[quantize] lora_packed shape: {lora_packed[0].shape}, {lora_packed[1].shape}, dtype={lora_packed[0].dtype}, {lora_packed[1].dtype}")

    # 4) Build SVDQ layer and fill parameters (all already packed for Nunchaku)
    svdq = SVDQW4A4Linear.from_linear(
        linear,
        rank=r_aligned,
        precision=precision,
        act_unsigned=act_unsigned,
    )
    svdq.qweight.copy_(qweight)
    svdq.wscales.copy_(wscales_packed)
    # low-rank packed
    if lora_packed is not None:
        svdq.proj_down.copy_(lora_packed[0])
        svdq.proj_up.copy_(lora_packed[1])
    
    # svdq.proj_down.copy_(lora_down.to(torch_dtype))
    # svdq.proj_up.copy_(lora_up.to(torch_dtype))
    
    # packed bias and smooth
    svdq.bias.copy_(linear.bias)
    svdq.smooth_factor.copy_(smooth_factor)
    # svdq.smooth_factor_orig.copy_(smooth_factor)

    # 5) Debug: reconstruct W_hat from (q, scales) + low-rank and report error
    try:
        # Use local q-debug (pre-pack) to measure approximation quality consistently
        N, K = residual.shape
        gs = 64
        s_rep = wscales.transpose(0, 1).repeat_interleave(gs, dim=1).to(residual.dtype)  # (N, K)
        q_dbg = quantize_residual_to_int4(residual, wscales, group_size=gs).to(residual.dtype)
        W_hat_recon = (q_dbg * s_rep)
        if r_aligned > 0:
            W_hat_recon = W_hat_recon + (lora_up @ lora_down.t()).to(residual.dtype)
        err = torch.mean((W_hat_recon - W_hat).float().pow(2))
        ref = torch.mean(W_hat.float().pow(2)) + 1e-12
        rel = (err / ref).item()
        print(f"[quantize][debug] layer recon rel-MSE={rel:.6e}")
    except Exception:
        pass

    print(f"[quantize] svdq.precision: {svdq.precision}")
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
        print(f"[quantize] ==== Processing layer: {name} ====")
        full_name = name
        if isinstance(child, nn.Linear):
            # Heuristic: first layer before ReLU -> act_unsigned=False; after ReLU -> act_unsigned=True
            r = ranks.get(full_name, 32)
            # act_unsigned = (full_name != "layer1")  # layer2 gets unsigned activations after ReLU
            act_unsigned = False
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
    t0 = time.time()
    print("[quantize] === Start ===")
    # Load BF16 model checkpoint
    ckpt = torch.load(ckpt_in, map_location=device)
    cfg = ckpt["config"]

    # Build FP16/BF16 model and load weights
    print("[quantize] Build BF16 model & load weights")
    model = TwoLayerMLP(**cfg).to(device).to(torch.bfloat16)
    model.load_state_dict(ckpt["state_dict"])  # trained bf16 weights
    model.eval()

    # === Calibration ===
    torch.manual_seed(123)
    calib_bs = 2048
    x_calib = torch.randn(calib_bs, cfg["in_features"], dtype=torch.bfloat16, device=device)
    # baseline outputs for objective
    t_ref = time.time()
    with torch.inference_mode():
        y_ref = model(x_calib)
    print(f"[quantize] Baseline forward done in {(time.time()-t_ref):.3f}s, shape={tuple(y_ref.shape)}")

    # SmoothQuant factors (alpha=0.5, clamp 2^[-2,2])
    t_sm = time.time()
    print("[quantize] Compute SmoothQuant factors (alpha=0.5, clamp=2^[-2,2])")
    smooth_map = compute_smooth_factors(model, x_calib, alpha=0.5, clamp_exp=2.0)
    print(f"[quantize] smooth_map: {smooth_map}")
    print(f"[quantize] SmoothQuant done in {(time.time()-t_sm):.3f}s")

    # Small grid over weight percentile
    candidate_ps = [0.999]
    best_p, best_loss, best_state = None, float("inf"), None
    best_model = None
    ranks_cfg = {"layer1": 128, "layer2": 128}
    print(f"[quantize] Percentile search over {candidate_ps}")
    for p in candidate_ps:
        t_p = time.time()
        cand = TwoLayerMLP(**cfg).to(device).to(torch.bfloat16)
        cand.load_state_dict(ckpt["state_dict"])
        cand.eval()
        cand = replace_module_linear_with_svdq(
            cand, ranks=ranks_cfg, w_percentile=p, smooth_map=smooth_map
        ).to(device).eval()
        with torch.inference_mode():
            y_q = cand(x_calib.unsqueeze(0)) if x_calib.ndim == 2 else cand(x_calib)
            if x_calib.ndim == 2:
                y_q = y_q.squeeze(0)
        loss = torch.mean((y_q - y_ref).float().pow(2)).item()
        print(f"[quantize][search] p={p} MSE={loss:.6f} time={(time.time()-t_p):.3f}s")
        if loss < best_loss:
            best_loss = loss
            best_p = p
            best_state = cand.state_dict()
            best_model = cand

    # Build final quantized model with best hyperparams
    print(f"[quantize] Build final quantized model (p={best_p})")
    # Directly use best candidate model without rebuilding
    qmodel = best_model if best_model is not None else cand

    # Optional: save to disk
    if ckpt_out:
        t_sv = time.time()
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
        print(f"[quantize] Saved checkpoint to {ckpt_out} in {(time.time()-t_sv):.3f}s")

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
    t_inf = time.time()
    with torch.inference_mode():
        pred = qmodel(x.unsqueeze(0)) if x.ndim == 2 else qmodel(x)
        if x.ndim == 2:
            pred = pred.squeeze(0)
        if squeeze_back:
            pred = pred.squeeze(0)
    loss = nn.MSELoss()(pred, y)
    print(f"[quantize] Direct-infer loss={loss.item():.6f} in {(time.time()-t_inf):.3f}s")
    for i in range(min(5, x.shape[0] if not squeeze_back else pred.shape[0])):
        if cfg.get("out_features", 1) == 1:
            print(f"  GT sum={y[i].item():.4f} | Pred={pred[i].item():.4f}")
        else:
            print(f"  GT: [{y[i,0].item():.4f}, {y[i,1].item():.4f}] | Pred: [{pred[i,0].item():.4f}, {pred[i,1].item():.4f}]")
    print(f"[quantize] === Done in {(time.time()-t0):.3f}s ===")
    return ckpt_out


if __name__ == "__main__":
    quantize_and_save()


