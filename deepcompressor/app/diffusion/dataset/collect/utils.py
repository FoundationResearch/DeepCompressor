# -*- coding: utf-8 -*-
"""Common utilities for collecting data."""

import inspect
import typing as tp

import torch
import torch.nn as nn
from diffusers.models.transformers import (
    FluxTransformer2DModel,
    PixArtTransformer2DModel,
    SanaTransformer2DModel,
)
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel as WanTransformer3DModel_HF
from fastvideo.models.dits.wanvideo import WanTransformer3DModel as WanTransformer3DModel_FV
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from deepcompressor.utils.common import tree_map, tree_split

__all__ = ["CollectHook"]


class CollectHook:
    def __init__(self, caches: list[dict[str, tp.Any]] = None, zero_redundancy: bool = False) -> None:
        self.caches = [] if caches is None else caches
        self.zero_redundancy = zero_redundancy

    def __call__(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tuple[torch.Tensor, ...],
    ) -> tp.Any:
        new_args = []
        signature = inspect.signature(module.forward)
        bound_arguments = signature.bind(*input_args, **input_kwargs)
        arguments = bound_arguments.arguments
        args_to_kwargs = {k: v for k, v in arguments.items() if k not in input_kwargs}
        input_kwargs.update(args_to_kwargs)

        if isinstance(module, UNet2DConditionModel):
            sample = input_kwargs.pop("sample")
            new_args.append(sample)
            timestep = input_kwargs["timestep"]
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(sample.shape[0])
            input_kwargs["timestep"] = timesteps
        elif isinstance(module, (PixArtTransformer2DModel, SanaTransformer2DModel)):
            new_args.append(input_kwargs.pop("hidden_states"))
        elif isinstance(module, FluxTransformer2DModel):
            new_args.append(input_kwargs.pop("hidden_states"))
        elif isinstance(module, (WanTransformer3DModel_HF, WanTransformer3DModel_FV)):
            # Wan 3D transformer: first input is hidden_states; ensure timestep is broadcasted
            if "hidden_states" in input_kwargs:
                new_args.append(input_kwargs.pop("hidden_states"))
            else:
                # fallback to positional
                new_args.append(input_args[0])
            # Normalize timestep like UNet path
            if "timestep" in input_kwargs:
                sample = new_args[0]
                timesteps = input_kwargs["timestep"]
                if not torch.is_tensor(timesteps):
                    is_mps = sample.device.type == "mps"
                    if isinstance(timesteps, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
                elif len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(sample.device)
                timesteps = timesteps.expand(sample.shape[0])
                input_kwargs["timestep"] = timesteps
        else:
            raise ValueError(f"Unknown model: {module}")
        cache = tree_map(lambda x: x.cpu(), {"input_args": new_args, "input_kwargs": input_kwargs, "outputs": output})
        split_cache = tree_split(cache)

        if isinstance(module, PixArtTransformer2DModel) and self.zero_redundancy:
            for cache in split_cache:
                cache_kwargs = cache["input_kwargs"]
                encoder_hidden_states = cache_kwargs.pop("encoder_hidden_states")
                assert encoder_hidden_states.shape[0] == 1
                encoder_attention_mask = cache_kwargs.get("encoder_attention_mask", None)
                if encoder_attention_mask is not None:
                    encoder_hidden_states = encoder_hidden_states[:, : max(encoder_attention_mask.sum(), 1)]
                cache_kwargs["encoder_hidden_states"] = encoder_hidden_states

        self.caches.extend(split_cache)
