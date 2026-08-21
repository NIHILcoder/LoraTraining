"""Filesystem helpers that must not depend on torch / FastAPI.

Kept in a separate module so unit tests can import them without loading the
training stack.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


def atomic_write_json(path: Path, data) -> None:
    """Write JSON via a temp file + os.replace so a crash cannot truncate the store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def unique_filename(desired: str, taken: Iterable[str]) -> str:
    """Return `desired` or `stem-N.suffix` if that name is already claimed."""
    taken_set = set(taken)
    if desired not in taken_set:
        return desired
    stem = Path(desired).stem
    suffix = Path(desired).suffix
    n = 1
    while n < 1000:
        candidate = f"{stem}-{n}{suffix}"
        if candidate not in taken_set:
            return candidate
        n += 1
    raise FileExistsError(f"Could not allocate a unique filename for {desired}")


def allocate_nonclobber_dest(models_dir: Path, src: Path) -> Path:
    """Choose a destination under models_dir that will not overwrite a *different* file.

    If `models_dir / src.name` is missing, use it.
    If that path is the same inode/path as `src` (file already lives in the models
    folder), return it so the caller can skip the copy.
    If another file already occupies the name, suffix -1, -2, ...
    """
    dest = models_dir / src.name
    if not dest.exists():
        return dest
    try:
        if dest.resolve() == src.resolve():
            return dest
    except OSError:
        pass
    n = 1
    while n < 1000:
        candidate = models_dir / f"{src.stem}-{n}{src.suffix}"
        if not candidate.exists():
            return candidate
        n += 1
    raise FileExistsError(f"Could not allocate a unique filename for {src.name}")


def should_unlink_model_file(filename: str, other_filenames: Iterable[str]) -> bool:
    """False when another catalog/custom entry still points at the same weight file."""
    return filename not in set(other_filenames)
