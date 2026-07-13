"""framelock — a lockfile for your video/sequence-ML datasets and experiments.

Serverless, dependency-free, git-friendly versioning + experiment tracking.
Snapshot a dataset by content (no copies), track a run pinned to that exact
version, and answer "was it the code or the data?" when a metric moves.
"""

from .manifest import Manifest, snapshot
from .diff import diff_manifests, DatasetDiff
from .tracking import track, Run, compare_runs
from .repo import Repo

__version__ = "0.1.0"

__all__ = [
    "Manifest",
    "snapshot",
    "diff_manifests",
    "DatasetDiff",
    "track",
    "Run",
    "compare_runs",
    "Repo",
    "__version__",
]
