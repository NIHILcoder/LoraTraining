"""Helpers for resumable model downloads (no GPU / FastAPI imports)."""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from typing import Optional

# Real base checkpoints are multi-GB; HTML error pages and LFS pointers are far smaller.
MIN_MODEL_BYTES = 10 * 1024 * 1024


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def part_is_promotable(
    part_size: int,
    expected_sha: Optional[str] = None,
    expected_size: int = 0,
    actual_sha: Optional[str] = None,
    min_bytes: int = MIN_MODEL_BYTES,
) -> bool:
    """True when an existing .part is already the finished object and can be renamed.

    Catalog downloads carry a SHA256 and/or fileSize. Either matching digest or
    an exact size match is enough. Custom URLs with neither must not be promoted
    from size-alone heuristics (a truncated .part could match nothing we can check).
    """
    if part_size < min_bytes:
        return False
    if expected_sha:
        return bool(actual_sha) and actual_sha.lower() == expected_sha.lower()
    if expected_size > 0:
        return part_size == int(expected_size)
    return False


def should_reset_part_after_http_error(status: int, can_promote: bool) -> bool:
    """HTTP 416 on a Range-at-EOF request loops forever if the .part is kept.

    When we cannot prove the part is complete, delete it so the next attempt
    starts from byte 0 instead of requesting Range: bytes={filesize}- again.
    """
    return status == 416 and not can_promote


def promote_part_file(
    part_path: Path,
    final_path: Path,
    retries: int = 5,
    delay_s: float = 0.5,
) -> None:
    """Rename .part → final path, retrying on Windows AV/handle locks."""
    last_err: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            if final_path.exists():
                final_path.unlink()
            shutil.move(str(part_path), str(final_path))
            return
        except OSError as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(delay_s)
    if last_err:
        raise last_err
    raise OSError(f"Failed to promote {part_path} to {final_path}")
