"""Diffing two dataset versions — the drift detector.

Given two manifests, classify every change into added / removed / modified /
moved. ``moved`` is the interesting one: the same content at a new path (a
relabel or reorg) is reported as a move, not as a delete+add, so a directory
reshuffle does not masquerade as "half my dataset changed".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .manifest import Manifest


@dataclass
class Move:
    from_path: str
    to_path: str
    hash: str


@dataclass
class Modified:
    path: str
    old_hash: str
    new_hash: str


@dataclass
class DatasetDiff:
    added: list = field(default_factory=list)      # list[path]
    removed: list = field(default_factory=list)    # list[path]
    modified: list = field(default_factory=list)   # list[Modified]
    moved: list = field(default_factory=list)      # list[Move]
    unchanged: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed or self.modified or self.moved)

    def summary(self) -> dict:
        return {
            "added": len(self.added),
            "removed": len(self.removed),
            "modified": len(self.modified),
            "moved": len(self.moved),
            "unchanged": self.unchanged,
        }


def diff_manifests(old: Manifest, new: Manifest) -> DatasetDiff:
    """Compare two manifests and classify the delta.

    Order of classification matters: a path present in both with a different
    hash is ``modified``; paths only on one side are candidate adds/removes, and
    among those we rescue same-content pairs as ``moved`` before reporting the
    rest as pure add/remove.
    """
    old_map = old.path_map()
    new_map = new.path_map()

    result = DatasetDiff()

    old_only_paths = []
    new_only_paths = []

    for path, old_hash in old_map.items():
        if path in new_map:
            if new_map[path] == old_hash:
                result.unchanged += 1
            else:
                result.modified.append(Modified(path, old_hash, new_map[path]))
        else:
            old_only_paths.append(path)

    for path in new_map:
        if path not in old_map:
            new_only_paths.append(path)

    # Rescue moves: a removed path and an added path sharing content = a move.
    # Build hash -> paths on each side, restricted to the path-only sets.
    removed_by_hash: dict = {}
    for p in old_only_paths:
        removed_by_hash.setdefault(old_map[p], []).append(p)
    added_by_hash: dict = {}
    for p in new_only_paths:
        added_by_hash.setdefault(new_map[p], []).append(p)

    for h, from_paths in removed_by_hash.items():
        to_paths = added_by_hash.get(h)
        if not to_paths:
            continue
        # Pair deterministically; extra copies on either side fall through to
        # add/remove below.
        from_paths_sorted = sorted(from_paths)
        to_paths_sorted = sorted(to_paths)
        for frm, to in zip(from_paths_sorted, to_paths_sorted):
            result.moved.append(Move(from_path=frm, to_path=to, hash=h))
        # Consume the paired paths so they are not double-counted.
        paired = min(len(from_paths_sorted), len(to_paths_sorted))
        removed_by_hash[h] = from_paths_sorted[paired:]
        added_by_hash[h] = to_paths_sorted[paired:]

    for h, paths in removed_by_hash.items():
        for p in paths:
            result.removed.append(p)
    for h, paths in added_by_hash.items():
        for p in paths:
            result.added.append(p)

    result.added.sort()
    result.removed.sort()
    result.modified.sort(key=lambda m: m.path)
    result.moved.sort(key=lambda m: m.to_path)
    return result
