"""Guard Playground generate against LoRA / base-model architecture mismatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def architecture_from_result_file(result_path: Path) -> Optional[str]:
    """Read the architecture recorded when the LoRA was saved. None if unknown."""
    if not result_path.is_file():
        return None
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    arch = data.get("architecture")
    if not arch:
        return None
    return str(arch).strip().lower() or None


def architecture_mismatch(lora_arch: Optional[str], base_arch: Optional[str]) -> Optional[str]:
    """Return an error message when both architectures are known and disagree."""
    la = (lora_arch or "").strip().lower()
    ba = (base_arch or "").strip().lower()
    if not la or not ba:
        return None
    if la == ba:
        return None
    return (
        f"This LoRA was trained for {la.upper()} but the selected base model is {ba.upper()}. "
        "Choose a matching base model in Playground."
    )
