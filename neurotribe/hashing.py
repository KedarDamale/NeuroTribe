"""Content hashing for the data registry and cache keys.

Large neuroimaging binaries (multi-GB NIfTI) are expensive to hash in full, so
above a configured size we compute a *partial* digest over the head and tail
plus the exact byte length. Partial digests are explicitly labelled so no code
path can silently treat them as a full content hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

CHUNK = 1024 * 1024


@dataclass(frozen=True)
class FileDigest:
    sha256: str
    size_bytes: int
    partial: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"{'partial' if self.partial else 'sha256'}:{self.sha256}"


def hash_file(path: Path, *, full_max_bytes: int = 256 * 1024 * 1024,
              partial_chunk_bytes: int = 8 * 1024 * 1024) -> FileDigest:
    """Hash a file, falling back to a head+tail digest for very large files."""
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.sha256()

    if size <= full_max_bytes:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(CHUNK), b""):
                digest.update(block)
        return FileDigest(digest.hexdigest(), size, partial=False)

    # Partial: size + first N bytes + last N bytes. Deterministic and cheap.
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(partial_chunk_bytes))
        handle.seek(max(0, size - partial_chunk_bytes))
        digest.update(handle.read(partial_chunk_bytes))
    return FileDigest(digest.hexdigest(), size, partial=True)


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_json(obj: Any, *, length: int | None = None) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def hash_many(digests: Iterable[str]) -> str:
    """Order-independent aggregate of several digests."""
    accumulator = hashlib.sha256()
    for item in sorted(digests):
        accumulator.update(item.encode("ascii"))
        accumulator.update(b"\x00")
    return accumulator.hexdigest()


def cache_key(kind: str, **components: Any) -> str:
    """Build a namespaced, deterministic cache key.

    Every expensive computation keys on its *inputs and configuration* so a
    changed parameter can never reuse a stale artefact.
    """
    return f"{kind}:{hash_json(components, length=32)}"
