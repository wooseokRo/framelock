"""Dataset snapshots — the immutable "lockfile" for a directory of data.

A snapshot walks a dataset directory, content-hashes every file, and records a
deterministic manifest keyed by a Merkle root. It never copies or moves the
data: the manifest is a few KB of JSON that pins exactly which bytes were
present, so you can commit it to git and reproduce a run later.
"""

from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Iterable

from . import hashing

MANIFEST_SCHEMA = 1

# Never snapshot our own store or common VCS/OS noise by default.
DEFAULT_IGNORE = (".framepin", ".git", ".hg", ".svn", ".DS_Store", "__pycache__")


@dataclass(frozen=True)
class FileEntry:
    path: str  # POSIX relpath from the dataset root
    hash: str
    size: int


@dataclass
class Manifest:
    root: str  # Merkle root digest — the dataset version id
    algo: str
    created_at: str
    dataset_path: str
    file_count: int
    total_bytes: int
    files: list  # list[FileEntry]
    splits: dict = field(default_factory=dict)  # name -> {glob, root, file_count, total_bytes}
    schema: int = MANIFEST_SCHEMA

    @property
    def short(self) -> str:
        return hashing.short(self.root)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["files"] = [asdict(f) if not isinstance(f, dict) else f for f in self.files]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        files = [FileEntry(**f) if isinstance(f, dict) else f for f in d.get("files", [])]
        return cls(
            root=d["root"],
            algo=d.get("algo", hashing.DEFAULT_ALGO),
            created_at=d.get("created_at", ""),
            dataset_path=d.get("dataset_path", ""),
            file_count=d.get("file_count", len(files)),
            total_bytes=d.get("total_bytes", sum(f.size for f in files)),
            files=files,
            splits=d.get("splits", {}),
            schema=d.get("schema", MANIFEST_SCHEMA),
        )

    def path_map(self) -> dict:
        """path -> hash, handy for diffing."""
        return {f.path: f.hash for f in self.files}


def _iter_files(root_dir: str, ignore: Iterable[str]):
    """Yield (abspath, relpath) for every non-ignored file under root_dir."""
    ignore = tuple(ignore)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Prune ignored directories in place so os.walk does not descend them.
        dirnames[:] = [
            d for d in dirnames if not any(fnmatch.fnmatch(d, pat) for pat in ignore)
        ]
        for name in filenames:
            if any(fnmatch.fnmatch(name, pat) for pat in ignore):
                continue
            abspath = os.path.join(dirpath, name)
            if not os.path.isfile(abspath):  # skip broken symlinks etc.
                continue
            rel = os.path.relpath(abspath, root_dir)
            yield abspath, hashing.normalize_relpath(rel)


def compute_splits(entries, split_globs: dict, algo: str = hashing.DEFAULT_ALGO) -> dict:
    """Per-split version ids over a subset of ``entries`` (list[FileEntry]).

    ``split_globs`` maps a split name to a glob (``*`` matches across ``/``,
    same semantics as the rest of framepin). A glob matching zero files is
    fine — its root is just the Merkle root of the empty set.
    """
    result = {}
    for name, glob in split_globs.items():
        matching = [e for e in entries if fnmatch.fnmatch(e.path, glob)]
        result[name] = {
            "glob": glob,
            "root": hashing.merkle_root(((e.path, e.hash) for e in matching), algo=algo),
            "file_count": len(matching),
            "total_bytes": sum(e.size for e in matching),
        }
    return result


def snapshot(
    dataset_path: str,
    algo: str = hashing.DEFAULT_ALGO,
    ignore: Iterable[str] = DEFAULT_IGNORE,
    created_at: str = "",
    jobs: int = hashing.DEFAULT_JOBS,
    split_globs: dict = None,
) -> Manifest:
    """Build a :class:`Manifest` for ``dataset_path`` without copying data.

    The ``root`` digest is a pure function of the file set (paths + content
    hashes) and is independent of walk order, ``created_at`` and ``jobs``, so
    snapshotting the same bytes twice yields the same version id. Pass
    ``split_globs`` (name -> glob) to additionally record a per-split version
    id for stratified datasets (see :func:`compute_splits`); splits are pure
    metadata computed over the same entries, so the top-level ``root`` is
    unchanged whether or not splits are requested.
    """
    if not os.path.isdir(dataset_path):
        raise NotADirectoryError(f"not a directory: {dataset_path}")

    files = list(_iter_files(dataset_path, ignore))
    digests = hashing.hash_files((a for a, _ in files), algo=algo, jobs=jobs)

    entries: list = []
    total = 0
    for abspath, rel in files:
        size = os.path.getsize(abspath)
        entries.append(FileEntry(path=rel, hash=digests[abspath], size=size))
        total += size

    entries.sort(key=lambda e: e.path)  # deterministic file ordering in the JSON
    root = hashing.merkle_root(((e.path, e.hash) for e in entries), algo=algo)
    splits = compute_splits(entries, split_globs, algo=algo) if split_globs else {}

    return Manifest(
        root=root,
        algo=algo,
        created_at=created_at,
        dataset_path=os.path.abspath(dataset_path),
        file_count=len(entries),
        total_bytes=total,
        files=entries,
        splits=splits,
    )


def dumps(manifest: Manifest) -> str:
    """Serialize a manifest to stable, git-friendly JSON (sorted keys)."""
    return json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)


def loads(text: str) -> Manifest:
    return Manifest.from_dict(json.loads(text))
