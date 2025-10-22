import torch
import torch.nn as nn
import cupy as cp


@torch.no_grad()
def quantize_activation_int4(x: torch.Tensor, group_size: int = 64, unsigned: bool = False, percentile: float | None = 1.0):
    """Quantize activations to int4 per input-channel group.

    Args:
        x: (B, K) or (B, S, K) tensor, bf16/fp16/fp32
        group_size: group size along K, default 64
        unsigned: if True, use [0, 15] with scale=max/15; else [-8, 7] with scale=max/7
        percentile: optional percentile for robust scaling; 1.0 uses amax

    Returns:
        x_q: int8 tensor with values in [0, 15] or [-8, 7], shape (B, K)
        ascales: per-group scales, shape (B, K//group_size)
    """
    if x.ndim == 3:
        b, s, k = x.shape
        x = x.reshape(b * s, k)
    else:
        b, k = x.shape
    assert k % group_size == 0
    g = k // group_size
    x_view = x.reshape(b, g, group_size)
    if percentile is None or percentile >= 1.0:
        a = x_view.abs().amax(dim=-1)  # (B, G)
    else:
        a = torch.quantile(x_view.abs().to(torch.float32), percentile, dim=-1).to(x.dtype)
    denom = 15.0 if unsigned else 7.0
    ascales = (a / denom).clamp_min(1e-8)
    x_q = (x_view / ascales.unsqueeze(-1)).round_()
    if unsigned:
        x_q = x_q.clamp_(0, 15)
    else:
        x_q = x_q.clamp_(-8, 7)
    x_q = x_q.reshape(b, k).to(torch.int8)
    return x_q, ascales  # (B, K), (B, G)


@torch.no_grad()
def svdlinear_forward_w4a4(
    x: torch.Tensor,
    q_int8: torch.Tensor,
    wscales: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    group_size: int = 64,
    act_unsigned: bool = False,
    with_a4: bool = True,
    a_percentile: float | None = 1.0,
) -> torch.Tensor:
    """Manual SVDQuant W4A4 inference in PyTorch with int8 accumulators.

    Args:
        x: input tensor, shape (B, K) or (B, S, K)
        q_int8: signed int8 quantized residual weights in int4 range, shape (N, K)
        wscales: per-group weight scales, shape (K//group_size, N), bf16/fp16
        lora_down: (K, R) bf16/fp16
        lora_up: (N, R) bf16/fp16
        bias: (N,) bf16/fp16 or None
        group_size: group size along K, default 64
        act_unsigned: if True, use unsigned activation quantization (0..15)
        with_a4: if True, quantize activations and use pure int32 matmul per group; else use fp activations
        a_percentile: optional percentile for activation scaling

    Returns:
        y: output tensor, shape (B, N), dtype out_dtype is the same as x.dtype
    """
    orig_shape = x.shape
    out_dtype = x.dtype
    if x.ndim == 3:
        b, s, k = x.shape
        x = x.reshape(b * s, k)
        b_eff = b * s
    else:
        b, k = x.shape
        s = 1
        b_eff = b
    n, k_w = q_int8.shape
    assert k == k_w, f"Input dim {k} != weight dim {k_w}"
    assert k % group_size == 0
    g = k // group_size
    assert wscales.shape == (g, n), f"wscales shape {tuple(wscales.shape)} != ({g}, {n})"

    # Residual branch
    y = torch.zeros(b_eff, n, dtype=torch.float32, device=x.device)
    if with_a4:
        x_q, a_scales = quantize_activation_int4(x, group_size=group_size, unsigned=act_unsigned, percentile=a_percentile)
        # Use CuPy int8 GEMM per-group, but accumulate in torch float32 to avoid shape issues
        dev_index = x.device.index if x.is_cuda else 0
        with cp.cuda.Device(dev_index):
            x_q_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x_q.contiguous()))
            q_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(q_int8.contiguous()))
            for gi in range(g):
                k0, k1 = gi * group_size, (gi + 1) * group_size
                # (B, gs) @ (gs, N) -> (B, N); cast to int32 to guarantee int32 accumulation and avoid overflow
                prod_cp = x_q_cp[:, k0:k1].astype(cp.int32) @ q_cp[:, k0:k1].astype(cp.int32).T
                prod_t = torch.utils.dlpack.from_dlpack(prod_cp.astype(cp.float32).toDlpack())
                # scales in torch: (B, N)
                scale_t = a_scales[:, gi].unsqueeze(1) * wscales[gi, :].unsqueeze(0)
                y.add_(prod_t * scale_t.to(torch.float32))
    else:
        # Float activations: accumulate per group as (x_g @ q_g^T) * wscale_g
        x_f = x.to(torch.float32)
        for gi in range(g):
            k0, k1 = gi * group_size, (gi + 1) * group_size
            acc = torch.matmul(x_f[:, k0:k1], q_int8[:, k0:k1].t().to(torch.float32))  # (B, N)
            y.add_(acc * wscales[gi, :].unsqueeze(0).to(torch.float32))

    # Low-rank branch in fp
    if lora_down is not None and lora_up is not None and lora_down.numel() > 0 and lora_up.numel() > 0:
        # Use the provided input `x` (expected pre-smoothed by caller) for low-rank branch
        h = torch.matmul(x.to(lora_down.dtype), lora_down)  # (B, R)
        y.add_(torch.matmul(h, lora_up.t()).to(y.dtype))

    if bias is not None:
        y.add_(bias.view(1, -1).to(y.dtype))

    if len(orig_shape) == 3:
        y = y.reshape(b, s, n)
    return y.to(out_dtype)


