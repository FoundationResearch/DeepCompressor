import os
import sys
import time
import torch
import torch.nn as nn

# Ensure repository root on path so sibling modules resolve
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Local utils (do NOT import deepcompressor backend here)
# from utils import (
#     NunchakuWeightPacker,
#     convert_to_nunchaku_w4x4y16_linear_weight,
# )
from svdlinear import svdlinear_forward_w4a4


@torch.no_grad()
def _truncated_svd_lowrank(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    U, S, Vh = (x.to(weight.dtype) for x in torch.linalg.svd(weight.float(), full_matrices=False))
    r = min(rank, U.shape[1])
    U_r = U[:, :r].contiguous()
    S_r = S[:r].contiguous()
    V_r = Vh[:r, :].contiguous()
    sr_sqrt = S_r.sqrt()
    lora_up = U_r * sr_sqrt.unsqueeze(0)        # (N, r)
    lora_down = V_r.t() * sr_sqrt.unsqueeze(0)  # (K, r)
    recon = (lora_up @ lora_down.t())           # (N, K)
    return lora_down, lora_up, recon


@torch.no_grad()
def _compute_group_scales_sym_int4(
    residual: torch.Tensor,
    group_size: int = 64,
    dtype=torch.bfloat16,
    percentile: float | None = None,
) -> torch.Tensor:
    N, K = residual.shape
    assert K % group_size == 0
    G = K // group_size
    res_block = residual.abs().reshape(N, G, group_size)
    if percentile is None or percentile >= 1.0:
        res = res_block.amax(dim=-1) / 7.0
    else:
        q = torch.quantile(res_block.to(torch.float32), percentile, dim=-1)
        res = (q / 7.0).to(residual.dtype)
    scales = res.transpose(0, 1).clamp_min(1e-8).to(dtype)  # (G, N)
    return scales


@torch.no_grad()
def _quantize_residual_to_int4(residual: torch.Tensor, scales: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    N, K = residual.shape
    s_exp = scales.transpose(0, 1).repeat_interleave(group_size, dim=1)
    q = (residual / s_exp).round().clamp_(-8, 7).to(torch.int8)
    return q


class SVDQuantLinearManual(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *, dtype=torch.bfloat16, act_unsigned: bool = False, rank: int = 0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.act_unsigned = act_unsigned
        self.group_size = 64

        # Buffers for manual simulation
        self.register_buffer("_manual_q_int8", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("_manual_wscales", torch.ones(in_features // self.group_size, out_features, dtype=dtype))
        r = int(rank) if rank is not None else 0
        self.register_buffer("_manual_lora_down", torch.zeros(in_features, r, dtype=dtype))
        self.register_buffer("_manual_lora_up", torch.zeros(out_features, r, dtype=dtype))
        self.register_buffer("_manual_smooth", torch.ones(in_features, dtype=dtype))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype))
        else:
            self.bias = None

    @staticmethod
    @torch.no_grad()
    def _compute_smooth_from_layer_inputs(x: torch.Tensor, weight: torch.Tensor, *, alpha: float = 0.5, clamp_exp: float = 2.0) -> torch.Tensor:
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        A = x.abs().amax(dim=0).to(torch.float32) + 1e-8
        W = weight.abs().amax(dim=0).to(torch.float32) + 1e-8
        s = (A / W).pow(alpha)
        gm = torch.exp(torch.mean(torch.log(s)))
        s = s / gm
        lo, hi = 2.0 ** (-clamp_exp), 2.0 ** (clamp_exp)
        s = s.clamp(min=lo, max=hi)
        return s

    @classmethod
    @torch.no_grad()
    def from_linear_and_inputs(
        cls,
        linear: nn.Linear,
        layer_inputs: torch.Tensor,
        *,
        rank: int = 32,
        w_percentile: float | None = 0.999,
        act_unsigned: bool = False,
    ) -> "SVDQuantLinearManual":
        in_features = linear.in_features
        out_features = linear.out_features
        assert in_features % 64 == 0, "INT4 requires in_features divisible by 64."
        dtype = torch.bfloat16 if linear.weight.dtype == torch.bfloat16 else torch.float16
        device = linear.weight.device

        mod = cls(in_features, out_features, bias=(linear.bias is not None), dtype=dtype, act_unsigned=act_unsigned, rank=0).to(device)

        # Smooth factor and smoothed weights
        s = cls._compute_smooth_from_layer_inputs(layer_inputs.to(device), linear.weight.data, alpha=0.5, clamp_exp=2.0).to(device=device, dtype=linear.weight.dtype)
        W_hat = (linear.weight.data * s.view(1, in_features)).contiguous()

        # Low-rank via SVD, align to multiples of 16
        r_base = min(rank, in_features, out_features)
        r_aligned = ((max(1, r_base) + 15) // 16) * 16
        lora_down_b, lora_up_b, recon = _truncated_svd_lowrank(W_hat, rank=r_base)
        if r_aligned != r_base:
            lora_down = torch.zeros(in_features, r_aligned, dtype=lora_down_b.dtype, device=lora_down_b.device)
            lora_up = torch.zeros(out_features, r_aligned, dtype=lora_up_b.dtype, device=lora_up_b.device)
            lora_down[:, :r_base] = lora_down_b
            lora_up[:, :r_base] = lora_up_b
        else:
            lora_down, lora_up = lora_down_b, lora_up_b

        # Residual and scaling
        residual = (W_hat - recon).contiguous()
        wscales = _compute_group_scales_sym_int4(residual, group_size=64, dtype=dtype, percentile=w_percentile)
        q_int8 = _quantize_residual_to_int4(residual, wscales, group_size=64).to(torch.int8)

        # Store manual tensors
        mod._manual_q_int8.copy_(q_int8)
        mod._manual_wscales.copy_(wscales)
        # Resize and fill low-rank buffers
        r_use = lora_down.shape[1]
        if mod._manual_lora_down.shape[1] != r_use:
            mod._buffers["_manual_lora_down"] = lora_down.to(dtype)
            mod._buffers["_manual_lora_up"] = lora_up.to(dtype)
        else:
            mod._manual_lora_down.copy_(lora_down.to(dtype))
            mod._manual_lora_up.copy_(lora_up.to(dtype))
        mod._manual_smooth.copy_(s.to(dtype))
        if linear.bias is not None:
            mod.bias.copy_(linear.bias.to(dtype))

        return mod

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_back = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze_back = True
        s = self._manual_smooth.view(1, 1, -1)
        x_div = x.to(self.dtype) / s
        y = svdlinear_forward_w4a4(
            x_div,
            self._manual_q_int8,
            self._manual_wscales,
            self._manual_lora_down,
            self._manual_lora_up,
            self.bias if hasattr(self, "bias") else None,
            group_size=self.group_size,
            act_unsigned=self.act_unsigned,
            with_a4=True,
        )
        if squeeze_back and y.ndim == 3 and y.shape[0] == 1:
            y = y.squeeze(0)
        return y

    @torch.no_grad()
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "class": "SVDQuantLinearManual"}, path)

    # ===== Model-level helpers =====
    @staticmethod
    @torch.no_grad()
    def collect_layer_inputs(model: nn.Module, inputs: torch.Tensor, device: str | torch.device) -> dict[str, torch.Tensor]:
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

    @staticmethod
    @torch.no_grad()
    def compute_smooth_factors(
        model_fp: nn.Module,
        calib_inputs: torch.Tensor,
        alpha: float = 0.5,
        clamp_exp: float = 2.0,
    ) -> dict[str, torch.Tensor]:
        model_fp.eval()
        layer_inputs = SVDQuantLinearManual.collect_layer_inputs(model_fp, calib_inputs, calib_inputs.device)
        smooth: dict[str, torch.Tensor] = {}
        for name, child in model_fp.named_children():
            if isinstance(child, nn.Linear):
                x = layer_inputs.get(name)
                if x is None:
                    continue
                s = SVDQuantLinearManual._compute_smooth_from_layer_inputs(x, child.weight, alpha=alpha, clamp_exp=clamp_exp)
                smooth[name] = s.to(calib_inputs.device)
            else:
                for subname, subchild in child.named_children():
                    if isinstance(subchild, nn.Linear):
                        x = layer_inputs.get(subname)
                        if x is None:
                            continue
                        s = SVDQuantLinearManual._compute_smooth_from_layer_inputs(x, subchild.weight, alpha=alpha, clamp_exp=clamp_exp)
                        smooth[subname] = s.to(calib_inputs.device)
        return smooth

    @staticmethod
    @torch.no_grad()
    def quantize_model(
        model: nn.Module,
        *,
        ranks: dict[str, int] | None = None,
        x_calib: torch.Tensor | None = None,
        calib_bs: int = 2048,
        w_percentile: float | None = 0.999,
        device: str | torch.device | None = None,
        alpha: float = 0.5,
        clamp_exp: float = 2.0,
    ) -> nn.Module:
        device = device or next(model.parameters()).device
        model = model.to(device).eval()

        # Infer input feature size from the first Linear
        in_features = None
        for _, m in model.named_modules():
            if isinstance(m, nn.Linear):
                in_features = m.in_features
                break
        assert in_features is not None, "No nn.Linear found in model."

        if x_calib is None:
            x_calib = torch.randn(calib_bs, in_features, dtype=torch.bfloat16, device=device)
        else:
            x_calib = x_calib.to(device)
        smooth_map = SVDQuantLinearManual.compute_smooth_factors(model, x_calib, alpha=alpha, clamp_exp=clamp_exp)

        ranks = ranks or {}
        for name, child in list(model.named_children()):
            full_name = name
            if isinstance(child, nn.Linear):
                r = ranks.get(full_name, 32)
                act_unsigned = False
                s = smooth_map.get(full_name)
                if s is None:
                    # Fallback: compute directly from a forward pass slice
                    s = SVDQuantLinearManual._compute_smooth_from_layer_inputs(x_calib, child.weight, alpha=alpha, clamp_exp=clamp_exp).to(x_calib.device)
                # Build per-layer inputs by passing x_calib through parents up to this layer is complex; use captured pre-hook instead
                # We already captured pre-activation inputs in compute_smooth_factors; reuse them if available
                layer_inputs = SVDQuantLinearManual.collect_layer_inputs(model, x_calib, device).get(full_name, x_calib)
                setattr(model, name, SVDQuantLinearManual.from_linear_and_inputs(child, layer_inputs, rank=r, w_percentile=w_percentile, act_unsigned=act_unsigned))
            else:
                # recurse one level
                for subname, subchild in list(child.named_children()):
                    if isinstance(subchild, nn.Linear):
                        r = ranks.get(subname, 32)
                        act_unsigned = False
                        layer_inputs = SVDQuantLinearManual.collect_layer_inputs(model, x_calib, device).get(subname, x_calib)
                        setattr(child, subname, SVDQuantLinearManual.from_linear_and_inputs(subchild, layer_inputs, rank=r, w_percentile=w_percentile, act_unsigned=act_unsigned))
        return model

    @staticmethod
    @torch.no_grad()
    def materialize_from_state_dict(model: nn.Module, state_dict: dict, *, device: str | torch.device | None = None) -> nn.Module:
        """Replace nn.Linear layers with SVDQuantLinearManual using ranks inferred from state_dict.

        Assumes keys like '<name>._manual_lora_down' exist to indicate rank.
        """
        device = device or next(model.parameters()).device
        model = model.to(device).eval()

        def infer_rank(prefix: str) -> int:
            key = f"{prefix}._manual_lora_down"
            if key in state_dict:
                return int(state_dict[key].shape[1])
            return 0

        for name, child in list(model.named_children()):
            full = name
            if isinstance(child, nn.Linear):
                r = infer_rank(full)
                mod = SVDQuantLinearManual(child.in_features, child.out_features, bias=(child.bias is not None), dtype=torch.bfloat16 if child.weight.dtype == torch.bfloat16 else torch.float16, act_unsigned=False, rank=r).to(device)
                setattr(model, name, mod)
            else:
                for subname, subchild in list(child.named_children()):
                    if isinstance(subchild, nn.Linear):
                        r = infer_rank(subname)
                        mod = SVDQuantLinearManual(subchild.in_features, subchild.out_features, bias=(subchild.bias is not None), dtype=torch.bfloat16 if subchild.weight.dtype == torch.bfloat16 else torch.float16, act_unsigned=False, rank=r).to(device)
                        setattr(child, subname, mod)
        return model


