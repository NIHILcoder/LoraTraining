"""Resolve a playground/training base-model id to a local checkpoint file."""

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

AUTO_IDS = {"", "auto", "none"}


def _entry_file(models_dir: Path, entry: Mapping[str, Any]) -> Optional[Path]:
    filename = entry.get("filename")
    if not filename:
        return None
    path = models_dir / str(filename)
    return path if path.is_file() else None


def _by_id(entries: Sequence[Mapping[str, Any]], model_id: str) -> Optional[Mapping[str, Any]]:
    for entry in entries:
        if entry.get("id") == model_id:
            return entry
    return None


def _display_name(entry: Mapping[str, Any]) -> str:
    return str(entry.get("name") or entry.get("shortName") or entry.get("filename") or "Unknown")


def resolve_base_model(
    model_id: Optional[str],
    models_dir: Path,
    catalog: Sequence[Mapping[str, Any]],
    custom_models: Sequence[Mapping[str, Any]],
) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
    """Return ``(path, architecture, display_name)`` for a requested base model.

    A specific catalog or custom id is resolved only against that entry. Missing
    or not-yet-downloaded ids do **not** fall back to a different checkpoint —
    that would silently generate (or train) with the wrong weights.

    ``None`` / ``"auto"`` picks the first downloaded catalog model, then the
    first downloaded custom model.
    """
    models_dir = Path(models_dir)
    requested = (model_id or "").strip()
    combined = list(catalog) + list(custom_models)

    if requested and requested.lower() not in AUTO_IDS:
        entry = _by_id(combined, requested)
        if entry is None:
            # Older clients send architecture ("sd15") instead of a catalog id.
            for candidate in combined:
                if candidate.get("architecture") != requested:
                    continue
                path = _entry_file(models_dir, candidate)
                if path is not None:
                    return path, candidate.get("architecture"), _display_name(candidate)
            return None, None, None
        path = _entry_file(models_dir, entry)
        if path is None:
            return None, entry.get("architecture"), _display_name(entry)
        return path, entry.get("architecture"), _display_name(entry)

    for entry in combined:
        path = _entry_file(models_dir, entry)
        if path is not None:
            return path, entry.get("architecture"), _display_name(entry)
    return None, None, None
