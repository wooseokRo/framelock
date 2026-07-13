"""Experiment tracking that stays pinned to an exact dataset version.

The whole point of framepin is the *link* between a run and the bytes it
trained on. ``track()`` records params, metrics, the git commit, and — crucially
— the dataset version(s) the run consumed, so later you can ask "did my metric
move because of the code or because of the data?".

    import framepin

    with framepin.track(name="baseline", params={"lr": 3e-4}) as run:
        run.use_dataset("data/clips")        # snapshots + pins the version
        for epoch in range(n):
            ...
        run.log_metric("val_loss", 0.21)
"""

from __future__ import annotations

import os
import subprocess
import uuid
from datetime import datetime, timezone

from . import manifest as manifest_mod
from .repo import Repo


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit(cwd: str = ".") -> str:
    """Best-effort current git commit; empty string outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


class Run:
    """A single tracked experiment. Usually created via :func:`track`."""

    def __init__(self, repo: Repo, name: str = "", params: dict = None, run_id: str = ""):
        self.repo = repo
        self.id = run_id or uuid.uuid4().hex[:12]
        self.name = name
        self.params = dict(params or {})
        self.metrics: dict = {}
        self.datasets: list = []  # list[{root, path, file_count, total_bytes}]
        self.created_at = _utcnow_iso()
        self.git_commit = git_commit(repo.root)
        self.status = "running"

    # -- recording -----------------------------------------------------------
    def use_dataset(self, ref, label: str = "") -> str:
        """Pin a dataset version to this run.

        ``ref`` may be a directory path (snapshotted now), an existing manifest
        root/prefix, or a :class:`Manifest`. Returns the pinned root digest.
        """
        if isinstance(ref, manifest_mod.Manifest):
            man = ref
        elif isinstance(ref, str) and os.path.isdir(ref):
            man = manifest_mod.snapshot(ref, created_at=_utcnow_iso())
        else:
            # treat as an existing manifest ref (root or short prefix)
            man = self.repo.load_manifest(ref)
        self.repo.save_manifest(man)
        self.datasets.append(
            {
                "root": man.root,
                "label": label,
                "path": man.dataset_path,
                "file_count": man.file_count,
                "total_bytes": man.total_bytes,
            }
        )
        return man.root

    def log_param(self, key: str, value) -> None:
        self.params[key] = value

    def log_metric(self, key: str, value) -> None:
        self.metrics[key] = value

    def log_metrics(self, metrics: dict) -> None:
        self.metrics.update(metrics)

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "status": self.status,
            "params": self.params,
            "metrics": self.metrics,
            "datasets": self.datasets,
        }

    def save(self) -> str:
        return self.repo.save_run(self.to_dict())

    # -- context manager -----------------------------------------------------
    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.status = "failed" if exc_type else "finished"
        self.save()
        return False  # never suppress exceptions


def compare_runs(run_a: dict, run_b: dict, metric: str = "") -> dict:
    """Compare two saved run dicts: metric deltas and whether the data changed.

    This powers the "was it the code or the data?" question. If the dataset
    version(s) differ between the two runs, a metric regression cannot be
    attributed to code alone — the report flags ``data_changed`` so the caller
    is warned before chasing a code bug that is really a data change.
    """
    roots_a = sorted(d.get("root", "") for d in run_a.get("datasets", []))
    roots_b = sorted(d.get("root", "") for d in run_b.get("datasets", []))
    data_changed = roots_a != roots_b

    metrics: dict = {}
    keys = set(run_a.get("metrics", {})) | set(run_b.get("metrics", {}))
    if metric:
        keys = {metric}
    for k in keys:
        va = run_a.get("metrics", {}).get(k)
        vb = run_b.get("metrics", {}).get(k)
        delta = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
        metrics[k] = {"a": va, "b": vb, "delta": delta}

    same_commit = run_a.get("git_commit") == run_b.get("git_commit") and bool(
        run_a.get("git_commit")
    )
    return {
        "run_a": run_a.get("id"),
        "run_b": run_b.get("id"),
        "data_changed": data_changed,
        "roots_a": roots_a,
        "roots_b": roots_b,
        "code_changed": not same_commit,
        "same_commit": same_commit,
        "metrics": metrics,
    }


def track(name: str = "", params: dict = None, repo: Repo = None, start: str = ".") -> Run:
    """Start a tracked run, auto-discovering (or initializing) the store.

    Use as a context manager so the run is persisted on exit even if training
    raises (it is saved with ``status="failed"``).
    """
    repo = repo or Repo.open_or_init(start)
    # Ensure store dirs exist even when opened from an existing root.
    Repo.init(repo.root)
    return Run(repo, name=name, params=params)
