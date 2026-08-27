"""Detect checkpoint architecture from a .safetensors header (no tensor load)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

# LDM / A1111-style prefixes, most specific first.
_FLUX_MARKERS = ("double_blocks.", "model.diffusion_model.double_blocks.")
_SD3_MARKERS = ("text_encoders.t5xxl.", "joint_blocks.")
_SDXL_MARKERS = (
    "conditioner.embedders.1.",
    "conditioner.embedders.0.",
    "text_encoder_2.",
)
_SD21_MARKERS = ("cond_stage_model.model.",)
_SD15_MARKERS = ("cond_stage_model.transformer.",)


def read_safetensors_tensor_keys(path: Path) -> set[str]:
    """Return tensor names from a safetensors file header without loading weights."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        if header_len <= 0 or header_len > 100 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header size: {header_len}")
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return set(header.keys())


def _any_prefix(keys: Iterable[str], prefixes: tuple[str, ...]) -> bool:
    for key in keys:
        for prefix in prefixes:
            if key.startswith(prefix) or prefix in key:
                return True
    return False


def infer_checkpoint_architecture(path: Path) -> Optional[str]:
    """Return sd15/sd21/sdxl/sd3/flux when the header is unambiguous, else None."""
    try:
        keys = read_safetensors_tensor_keys(path)
    except Exception:
        return None
    if not keys:
        return None
    if _any_prefix(keys, _FLUX_MARKERS):
        return "flux"
    if _any_prefix(keys, _SD3_MARKERS):
        return "sd3"
    if _any_prefix(keys, _SDXL_MARKERS):
        return "sdxl"
    if _any_prefix(keys, _SD21_MARKERS):
        return "sd21"
    if _any_prefix(keys, _SD15_MARKERS):
        return "sd15"
    return None


def resolve_base_model_path(
    architecture: str,
    models_dir: Path,
    catalog: Iterable[dict],
    custom_models: Iterable[dict],
) -> Optional[Path]:
    """Pick a local checkpoint for `architecture`.

    Custom (imported/URL) files win over catalog downloads so that Import File
    is not silently ignored when the user also has the official catalog model.
    The most recently appended custom entry is tried first.
    """
    customs = list(custom_models)
    for cm in reversed(customs):
        if cm.get("architecture") != architecture:
            continue
        filename = cm.get("filename")
        if not filename:
            continue
        candidate = models_dir / filename
        if candidate.is_file():
            return candidate
    for m in catalog:
        if m.get("architecture") != architecture:
            continue
        filename = m.get("filename")
        if not filename:
            continue
        candidate = models_dir / filename
        if candidate.is_file():
            return candidate
    return None
