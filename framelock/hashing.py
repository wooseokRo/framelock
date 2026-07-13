"""Content hashing utilities.

framelock identifies data by *content*, not by path or mtime. A frame that is
byte-identical across two datasets hashes the same, which is what lets us detect
moved/renamed files and reproduce a run without copying a single byte.
"""

from __future__ import annotations

import hashlib
import os

# 1 MiB read chunks keep memory flat even for multi-GB video files.
_CHUNK = 1 << 20

DEFAULT_ALGO = "sha256"


def hash_file(path: str, algo: str = DEFAULT_ALGO) -> str:
    """Return the hex content digest of a file, streamed in chunks.

    Streaming (rather than reading the whole file) is what makes this usable on
    the large media files typical of video/sequence datasets.
    """
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_bytes(data: bytes, algo: str = DEFAULT_ALGO) -> str:
    """Return the hex content digest of an in-memory buffer."""
    return hashlib.new(algo, data).hexdigest()


def merkle_root(entries, algo: str = DEFAULT_ALGO) -> str:
    """Compute a deterministic root digest over ``(relpath, filehash)`` entries.

    The root is order-independent of the caller because we sort by path first,
    so the same set of files always yields the same root regardless of how the
    filesystem enumerated them. Both path and hash feed the root, so a pure
    rename changes the root even when file contents do not.
    """
    h = hashlib.new(algo)
    for relpath, filehash in sorted(entries, key=lambda e: e[0]):
        # NUL separators avoid ambiguity between path and hash boundaries.
        h.update(relpath.encode("utf-8"))
        h.update(b"\x00")
        h.update(filehash.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def short(digest: str, length: int = 12) -> str:
    """Human-facing abbreviation of a digest (git-style short id)."""
    return digest[:length]


def normalize_relpath(path: str) -> str:
    """Normalize a relative path to POSIX separators for cross-OS determinism."""
    return path.replace(os.sep, "/")
