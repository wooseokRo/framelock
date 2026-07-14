"""Path-list ("manifest txt") datasets.

Many ML teams define a dataset not as a directory but as text files listing
absolute paths — one clip/scene per line — and concatenate several lists per
experiment. ``snapshot_from_lists`` pins that whole construct:

- every list file itself (so adding/removing/reordering lines changes the
  version), and
- every file the lists reference (so a re-encoded clip changes the version
  even though the list text did not).

Repeated paths across lists are deduped. Referenced paths that do not exist
are recorded with a ``missing:`` marker instead of aborting a 500k-line scan —
a dead path is exactly the kind of drift worth versioning.

Hashing 10^5-10^6 video files is expensive, so two accelerations are built in:

- ``HashCache`` — remembers (size, mtime_ns) -> content hash, so unchanged
  files are never re-read. First snapshot pays full price; later ones only
  hash what changed.
- ``fast=True`` — fingerprints files by (size, mtime_ns) without reading
  content. Minutes for ~500k files, but an overwrite that preserves size and
  mtime goes undetected, and identical fingerprints on different paths can
  confuse move detection. Use it for routine change tracking, and take a
  periodic full-content snapshot as the anchor of record.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

from . import hashing
from .manifest import FileEntry, Manifest


def parse_list_file(path: str) -> list:
    """Referenced paths from one list file: one per line, ``#`` comments and
    blank lines ignored, whitespace stripped."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


class HashCache:
    """(path, size, mtime_ns, algo) -> content hash, persisted as JSON."""

    def __init__(self, path: str):
        self.path = path
        self._data = {}
        self._dirty = False
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self._data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                self._data = {}

    def get(self, file_path: str, size: int, mtime_ns: int, algo: str):
        rec = self._data.get(file_path)
        if rec and rec.get("size") == size and rec.get("mtime_ns") == mtime_ns \
                and rec.get("algo") == algo:
            return rec["hash"]
        return None

    def put(self, file_path: str, size: int, mtime_ns: int, algo: str, digest: str):
        self._data[file_path] = {
            "size": size, "mtime_ns": mtime_ns, "algo": algo, "hash": digest,
        }
        self._dirty = True

    def save(self):
        if not self._dirty:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh)
        os.replace(tmp, self.path)
        self._dirty = False


def _entry_for(abspath: str, algo: str, cache: HashCache | None) -> FileEntry:
    """Content-hashed entry for a single file (used for the list files, which
    are small; referenced files go through the batched path in
    snapshot_from_lists so they can be hashed concurrently)."""
    try:
        st = os.stat(abspath)
    except OSError:
        return FileEntry(path=abspath, hash="missing:", size=0)
    if cache is not None:
        hit = cache.get(abspath, st.st_size, st.st_mtime_ns, algo)
        if hit is not None:
            return FileEntry(path=abspath, hash=hit, size=st.st_size)
    digest = hashing.hash_file(abspath, algo=algo)
    if cache is not None:
        cache.put(abspath, st.st_size, st.st_mtime_ns, algo, digest)
    return FileEntry(path=abspath, hash=digest, size=st.st_size)


def snapshot_from_lists(
    list_paths: Iterable[str],
    algo: str = hashing.DEFAULT_ALGO,
    created_at: str = "",
    cache: HashCache | None = None,
    fast: bool = False,
    jobs: int = hashing.DEFAULT_JOBS,
) -> Manifest:
    """Build a Manifest for "dataset = these list files + what they reference".

    The version id is a pure function of the list files' content plus each
    referenced file's (path, hash/fingerprint) — same construct, same id
    regardless of ``jobs``. Only cache-missing files are content-read; those
    are hashed concurrently (hashing.hash_files).
    """
    list_paths = [os.path.abspath(p) for p in list_paths]
    if not list_paths:
        raise ValueError("snapshot_from_lists: no list files given")

    entries: list = []
    total = 0

    referenced = []
    seen = set()
    for lp in list_paths:
        # list files are always content-hashed (small), never fast-fingerprinted
        entries.append(_entry_for(lp, algo, cache))
        total += entries[-1].size
        for ref in parse_list_file(lp):
            ref = os.path.abspath(ref)
            if ref not in seen:
                seen.add(ref)
                referenced.append(ref)

    # stat in the main thread; farm out only the actual content hashing
    to_hash: list = []
    stats = {}
    for ref in referenced:
        try:
            st = os.stat(ref)
        except OSError:
            entries.append(FileEntry(path=ref, hash="missing:", size=0))
            continue
        stats[ref] = st
        if fast:
            entries.append(FileEntry(
                path=ref, hash=f"stat:{st.st_size}:{st.st_mtime_ns}", size=st.st_size))
            total += st.st_size
            continue
        hit = cache.get(ref, st.st_size, st.st_mtime_ns, algo) if cache else None
        if hit is not None:
            entries.append(FileEntry(path=ref, hash=hit, size=st.st_size))
            total += st.st_size
        else:
            to_hash.append(ref)

    if to_hash:
        digests = hashing.hash_files(to_hash, algo=algo, jobs=jobs)
        for ref in to_hash:
            st = stats[ref]
            entries.append(FileEntry(path=ref, hash=digests[ref], size=st.st_size))
            total += st.st_size
            if cache is not None:
                cache.put(ref, st.st_size, st.st_mtime_ns, algo, digests[ref])

    if cache is not None:
        cache.save()

    entries.sort(key=lambda e: e.path)
    root = hashing.merkle_root(((e.path, e.hash) for e in entries), algo=algo)
    return Manifest(
        root=root,
        algo=algo,
        created_at=created_at,
        dataset_path="lists:" + ";".join(list_paths),
        file_count=len(entries),
        total_bytes=total,
        files=entries,
    )
