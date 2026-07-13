"""The on-disk store: a ``.framepin/`` directory of manifests and runs.

Everything framepin persists is plain JSON under ``.framepin/`` so it commits
cleanly to git and is trivially inspectable. No database, no server.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from . import hashing
from .manifest import Manifest, dumps as manifest_dumps, loads as manifest_loads

STORE_DIR = ".framepin"


class RepoError(Exception):
    pass


def find_repo(start: str = ".") -> Optional[str]:
    """Walk upward from ``start`` looking for a ``.framepin`` directory."""
    cur = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(cur, STORE_DIR)):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


class Repo:
    """A framepin store rooted at a project directory."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    @property
    def store(self) -> str:
        return os.path.join(self.root, STORE_DIR)

    @property
    def datasets_dir(self) -> str:
        return os.path.join(self.store, "datasets")

    @property
    def runs_dir(self) -> str:
        return os.path.join(self.store, "runs")

    # -- lifecycle -----------------------------------------------------------
    @classmethod
    def init(cls, root: str = ".") -> "Repo":
        repo = cls(root)
        os.makedirs(repo.datasets_dir, exist_ok=True)
        os.makedirs(repo.runs_dir, exist_ok=True)
        return repo

    @classmethod
    def discover(cls, start: str = ".") -> "Repo":
        found = find_repo(start)
        if found is None:
            raise RepoError(
                "no .framepin store found (run `framepin init` first)"
            )
        return cls(found)

    @classmethod
    def open_or_init(cls, start: str = ".") -> "Repo":
        found = find_repo(start)
        return cls(found) if found else cls.init(start)

    # -- manifests -----------------------------------------------------------
    def save_manifest(self, manifest: Manifest) -> str:
        os.makedirs(self.datasets_dir, exist_ok=True)
        path = os.path.join(self.datasets_dir, manifest.root + ".json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(manifest_dumps(manifest))
        return path

    def load_manifest(self, ref: str) -> Manifest:
        """Load a manifest by full root or unambiguous short prefix."""
        full = self._resolve(self.datasets_dir, ref)
        with open(os.path.join(self.datasets_dir, full + ".json"), encoding="utf-8") as fh:
            return manifest_loads(fh.read())

    def list_manifests(self) -> list:
        return self._list_ids(self.datasets_dir)

    # -- runs ----------------------------------------------------------------
    def save_run(self, run: dict) -> str:
        os.makedirs(self.runs_dir, exist_ok=True)
        path = os.path.join(self.runs_dir, run["id"] + ".json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(run, fh, indent=2, sort_keys=True, ensure_ascii=False)
        return path

    def load_run(self, ref: str) -> dict:
        full = self._resolve(self.runs_dir, ref)
        with open(os.path.join(self.runs_dir, full + ".json"), encoding="utf-8") as fh:
            return json.load(fh)

    def list_runs(self) -> list:
        runs = []
        for rid in self._list_ids(self.runs_dir):
            try:
                runs.append(self.load_run(rid))
            except (OSError, ValueError):
                continue
        runs.sort(key=lambda r: r.get("created_at", ""))
        return runs

    # -- helpers -------------------------------------------------------------
    def _list_ids(self, directory: str) -> list:
        if not os.path.isdir(directory):
            return []
        return sorted(
            name[:-5] for name in os.listdir(directory) if name.endswith(".json")
        )

    def _resolve(self, directory: str, ref: str) -> str:
        """Resolve a full id or short prefix to a stored id; error if ambiguous."""
        ids = self._list_ids(directory)
        if ref in ids:
            return ref
        matches = [i for i in ids if i.startswith(ref)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise RepoError(f"no object matching '{ref}'")
        raise RepoError(f"ambiguous ref '{ref}' matches {len(matches)} objects")
