"""Export framepin lineage to MLflow / Weights & Biases.

Send the pinned dataset version(s) for a run to whichever experiment
tracker you already use, so the exact data snapshot shows up right next
to the run in that tool's UI.

MLflow:

    with framepin.track(name="baseline") as run:
        run.use_dataset("data/clips")
        framepin.integrations.to_mlflow(run)

Weights & Biases:

    with framepin.track(name="baseline") as run:
        run.use_dataset("data/clips")
        framepin.integrations.to_wandb(run)

Neither ``mlflow`` nor ``wandb`` is a framepin dependency — they are
imported lazily inside the functions below, so importing this module
never requires either library to be installed.
"""

from __future__ import annotations


def _as_dict(run) -> dict:
    return run.to_dict() if hasattr(run, "to_dict") else dict(run)


def _lineage(prefix: str, data: dict) -> dict:
    """The prefixed key/value lineage payload shared by every exporter."""
    out = {f"{prefix}run_id": data.get("id", "")}
    if data.get("git_commit", ""):
        out[f"{prefix}git_commit"] = data["git_commit"]
    for i, d in enumerate(data.get("datasets", [])):
        key = d.get("label") or str(i)
        out[f"{prefix}dataset.{key}"] = d.get("root", "")
    return out


def to_mlflow(run, *, prefix: str = "framepin.", log_params: bool = False) -> None:
    """Tag the active MLflow run with this framepin run's lineage.

    Accepts a :class:`framepin.tracking.Run` or a plain run dict. Requires
    an active MLflow run (``mlflow.start_run()``) — tags are set on it via
    ``mlflow.set_tags``. Set ``log_params=True`` to also forward the
    framepin run's params via ``mlflow.log_params`` (off by default to
    avoid colliding with params you log yourself).
    """
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("mlflow is not installed — pip install mlflow") from exc

    data = _as_dict(run)
    mlflow.set_tags(_lineage(prefix, data))

    if log_params and data.get("params"):
        mlflow.log_params(data["params"])


def to_wandb(run, *, prefix: str = "framepin.", run_obj=None) -> None:
    """Attach this framepin run's lineage to a Weights & Biases run's config.

    Accepts a :class:`framepin.tracking.Run` or a plain run dict. Updates
    ``run_obj.config`` (or ``wandb.run.config`` if ``run_obj`` is not
    given) via ``config.update(..., allow_val_change=True)``.
    """
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed — pip install wandb") from exc

    target = run_obj if run_obj is not None else wandb.run
    if target is None:
        raise RuntimeError("no active wandb run — call wandb.init() first")

    target.config.update(_lineage(prefix, _as_dict(run)), allow_val_change=True)
