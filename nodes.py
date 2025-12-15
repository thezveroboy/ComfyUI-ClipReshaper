import os
from typing import Dict, List, Tuple, Any

import torch


# -----------------------------
# Helpers for ComfyUI conditioning
# -----------------------------
def _conditioning_items(cond) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    ComfyUI CONDITIONING is typically a list of [tensor, meta] pairs.
    This helper normalizes access.
    """
    if cond is None:
        raise ValueError("Conditioning is None.")
    if not isinstance(cond, (list, tuple)):
        raise TypeError(f"Expected CONDITIONING list/tuple, got: {type(cond)}")

    items = []
    for i, it in enumerate(cond):
        if not isinstance(it, (list, tuple)) or len(it) != 2:
            raise TypeError(f"Conditioning item #{i} must be [tensor, meta]. Got: {it!r}")
        t, m = it
        if not torch.is_tensor(t):
            raise TypeError(f"Conditioning tensor at #{i} is not a torch.Tensor: {type(t)}")
        if m is None:
            m = {}
        if not isinstance(m, dict):
            raise TypeError(f"Conditioning meta at #{i} must be dict, got: {type(m)}")
        items.append((t, m))
    return items


def _clone_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    # shallow copy is fine; values are mostly primitives / small tensors
    return dict(meta)


def _tensor_last_dim(t: torch.Tensor) -> int:
    if t.ndim < 1:
        raise ValueError(f"Conditioning tensor has invalid shape: {tuple(t.shape)}")
    return int(t.shape[-1])


def _pad_or_slice_last_dim(t: torch.Tensor, target_dim: int) -> torch.Tensor:
    d = _tensor_last_dim(t)
    if d == target_dim:
        return t
    if d > target_dim:
        return t[..., :target_dim]
    # pad zeros
    pad = torch.zeros(*t.shape[:-1], target_dim - d, device=t.device, dtype=t.dtype)
    return torch.cat([t, pad], dim=-1)


def _ensure_pooled_output(meta: Dict[str, Any], seq: torch.Tensor) -> Dict[str, Any]:
    """
    Some SDXL-style pipelines expect 'pooled_output' in meta.
    If missing, create a simple pooled vector (mean over token dimension).
    """
    new_meta = _clone_meta(meta)
    if "pooled_output" not in new_meta:
        # typical seq shape: [B, T, D] or [T, D]; do a conservative mean over T if possible
        if seq.ndim >= 2:
            pooled = seq.mean(dim=-2)  # mean over tokens
        else:
            pooled = seq
        new_meta["pooled_output"] = pooled
    return new_meta


def _ensure_sdxl_size_meta(meta: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    """
    SDXL conditioning often carries extra size/crop/target keys in meta.
    Different node packs may use slightly different keys; keep a minimal set.
    """
    new_meta = _clone_meta(meta)

    # Common fields seen in SDXL encoders/nodes: width/height + crop + target sizes. [web:30]
    new_meta.setdefault("width", int(width))
    new_meta.setdefault("height", int(height))

    new_meta.setdefault("crop_w", 0)
    new_meta.setdefault("crop_h", 0)
    new_meta.setdefault("target_width", int(width))
    new_meta.setdefault("target_height", int(height))

    return new_meta


# -----------------------------
# Nodes
# -----------------------------
class CR_ConditioningInspect:
    CATEGORY = "Clip Reshaper"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "inspect"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",)}}

    def inspect(self, conditioning):
        items = _conditioning_items(conditioning)

        lines = []
        lines.append(f"items: {len(items)}")
        for i, (t, m) in enumerate(items):
            shape = tuple(t.shape)
            d = _tensor_last_dim(t)
            has_pooled = "pooled_output" in m
            pooled_shape = tuple(m["pooled_output"].shape) if has_pooled and torch.is_tensor(m["pooled_output"]) else None
            size_keys = [k for k in ("width", "height", "crop_w", "crop_h", "target_width", "target_height") if k in m]
            lines.append(
                f"[{i}] tensor_shape={shape} D={d} pooled={has_pooled} pooled_shape={pooled_shape} size_keys={size_keys}"
            )

        return ("\n".join(lines),)


class CR_ConditioningAssertDim:
    CATEGORY = "Clip Reshaper"
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "assert_dim"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "expected_dim": ("INT", {"default": 768, "min": 1, "max": 65536}),
            }
        }

    def assert_dim(self, conditioning, expected_dim: int):
        items = _conditioning_items(conditioning)
        for i, (t, _) in enumerate(items):
            d = _tensor_last_dim(t)
            if d != int(expected_dim):
                raise ValueError(
                    f"Clip Reshaper: conditioning item #{i} has D={d}, expected D={expected_dim}. "
                    f"This mismatch often causes 'mat1 and mat2 shapes cannot be multiplied' later."
                )
        return (conditioning,)


class CR_ConditioningPadOrSlice:
    CATEGORY = "Clip Reshaper"
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "pad_or_slice"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "target_dim": ("INT", {"default": 1280, "min": 1, "max": 65536}),
            }
        }

    def pad_or_slice(self, conditioning, target_dim: int):
        items = _conditioning_items(conditioning)
        out = []
        for t, m in items:
            out_t = _pad_or_slice_last_dim(t, int(target_dim))
            out.append([out_t, _clone_meta(m)])
        return (out,)


class CR_ConditioningLinearProject:
    """
    Projects last-dim D_in -> D_out using a learned (or user-provided) linear layer.
    Without trained weights this is mainly for experiments / adapters.
    """
    CATEGORY = "Clip Reshaper"
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "project"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "target_dim": ("INT", {"default": 1280, "min": 1, "max": 65536}),
                "weights_path": ("STRING", {"default": ""}),
                "bias": ("BOOLEAN", {"default": True}),
            }
        }

    def project(self, conditioning, target_dim: int, weights_path: str, bias: bool):
        items = _conditioning_items(conditioning)

        # Determine input dim from first item
        din = _tensor_last_dim(items[0][0])
        dout = int(target_dim)

        layer = torch.nn.Linear(din, dout, bias=bool(bias))

        # Optional: load weights
        weights_path = (weights_path or "").strip()
        if weights_path:
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(f"weights_path not found: {weights_path}")
            sd = torch.load(weights_path, map_location="cpu")
            # Accept either full state_dict {'weight','bias'} or nested {'layer':{...}}
            if "weight" in sd:
                layer.load_state_dict(sd, strict=False)
            elif "layer" in sd and isinstance(sd["layer"], dict):
                layer.load_state_dict(sd["layer"], strict=False)
            else:
                raise ValueError("Unsupported weights format. Expected keys: weight/bias or layer{...}")

        # Move layer to tensor device/dtype per item (safe for mixed precision)
        out = []
        for t, m in items:
            l = layer.to(device=t.device, dtype=t.dtype)
            out_t = l(t)
            out.append([out_t, _clone_meta(m)])
        return (out,)


class CR_SDXLMetadataEnsure:
    CATEGORY = "Clip Reshaper"
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "ensure"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "ensure_pooled_output": ("BOOLEAN", {"default": True}),
            }
        }

    def ensure(self, conditioning, width: int, height: int, ensure_pooled_output: bool):
        items = _conditioning_items(conditioning)
        out = []
        for t, m in items:
            meta = _ensure_sdxl_size_meta(m, width=int(width), height=int(height))
            if bool(ensure_pooled_output):
                meta = _ensure_pooled_output(meta, t)
            out.append([t, meta])
        return (out,)


NODE_CLASS_MAPPINGS = {
    "CR_ConditioningInspect": CR_ConditioningInspect,
    "CR_ConditioningAssertDim": CR_ConditioningAssertDim,
    "CR_ConditioningPadOrSlice": CR_ConditioningPadOrSlice,
    "CR_ConditioningLinearProject": CR_ConditioningLinearProject,
    "CR_SDXLMetadataEnsure": CR_SDXLMetadataEnsure,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CR_ConditioningInspect": "Clip Reshaper: Inspect Conditioning",
    "CR_ConditioningAssertDim": "Clip Reshaper: Assert Dim",
    "CR_ConditioningPadOrSlice": "Clip Reshaper: Pad/Slice Dim",
    "CR_ConditioningLinearProject": "Clip Reshaper: Linear Project Dim",
    "CR_SDXLMetadataEnsure": "Clip Reshaper: Ensure SDXL Metadata",
}
