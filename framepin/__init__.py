"""framepin — a lockfile for your video/sequence-ML datasets and experiments.

Serverless, dependency-free, git-friendly versioning + experiment tracking.
Snapshot a dataset by content (no copies), track a run pinned to that exact
version, and answer "was it the code or the data?" when a metric moves.
"""

from .manifest import Manifest, snapshot
from .listfile import snapshot_from_lists, HashCache
from .diff import diff_manifests, DatasetDiff
from .tracking import track, Run, compare_runs
from .repo import Repo
from . import integrations

__version__ = "0.3.1"

__all__ = [
    "snapshot_from_lists",
    "HashCache",
    "Manifest",
    "snapshot",
    "diff_manifests",
    "DatasetDiff",
    "track",
    "Run",
    "compare_runs",
    "Repo",
    "integrations",
    "__version__",
]
