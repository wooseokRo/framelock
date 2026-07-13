#!/usr/bin/env python3
"""End-to-end framepin demo — no third-party deps, no real GPU needed.

Simulates the everyday drift problem: you train a baseline, then someone
re-labels/re-samples the dataset, you train again, the metric moves — and
framepin tells you it was the *data*, not your code.

Run it:

    cd projects/framepin
    python3 examples/quickstart_demo.py

It builds a throwaway dataset in a temp dir, so it is safe to run repeatedly and
leaves your project untouched. Great source material for an asciinema/GIF.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import framepin


def _make_clip(path: str, content: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(content)


def build_dataset_v1(root: str) -> None:
    """A tiny 'video' dataset: a few clips under train/ and val/."""
    _make_clip(os.path.join(root, "train", "clip_000.bin"), b"frames-000" * 64)
    _make_clip(os.path.join(root, "train", "clip_001.bin"), b"frames-001" * 64)
    _make_clip(os.path.join(root, "val", "clip_100.bin"), b"frames-100" * 64)


def mutate_to_v2(root: str) -> None:
    """Simulate a real dataset churn between two experiments:
    - re-label: move a clip into a new class folder (same bytes -> a *move*)
    - re-encode one clip (content change -> *modified*)
    - add a new clip (*added*)
    """
    # relabel / reorg: same content, new path -> framepin should call it a MOVE
    os.makedirs(os.path.join(root, "train", "hard"), exist_ok=True)
    shutil.move(
        os.path.join(root, "train", "clip_001.bin"),
        os.path.join(root, "train", "hard", "clip_001.bin"),
    )
    # re-encode an existing clip -> MODIFIED
    _make_clip(os.path.join(root, "val", "clip_100.bin"), b"frames-100-REENCODED" * 48)
    # collect a new clip -> ADDED
    _make_clip(os.path.join(root, "train", "clip_002.bin"), b"frames-002" * 64)


def fake_train(dataset_version: str, difficulty: float) -> float:
    """Stand-in for training. Returns a deterministic 'val_loss' so the demo is
    reproducible (no RNG). Harder data -> worse loss."""
    return round(0.20 + difficulty, 4)


def rule(title: str) -> None:
    print("\n" + "=" * 62 + f"\n{title}\n" + "=" * 62)


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="framepin-demo-")
    data = os.path.join(workdir, "data", "clips")
    try:
        os.chdir(workdir)
        repo = framepin.Repo.init(workdir)

        rule("1) Build dataset v1 and train a baseline")
        build_dataset_v1(data)
        with framepin.track(name="baseline", params={"lr": 3e-4}, repo=repo) as run1:
            v1 = run1.use_dataset(data)
            run1.log_metric("val_loss", fake_train(v1, difficulty=0.01))
        print(f"  dataset v1  = {v1[:12]}")
        print(f"  run baseline= {run1.id}   val_loss={run1.metrics['val_loss']}")

        rule("2) Someone re-labels + re-encodes + adds clips, then trains again")
        mutate_to_v2(data)
        with framepin.track(name="v2-run", params={"lr": 3e-4}, repo=repo) as run2:
            v2 = run2.use_dataset(data)
            run2.log_metric("val_loss", fake_train(v2, difficulty=0.09))
        print(f"  dataset v2  = {v2[:12]}")
        print(f"  run v2-run  = {run2.id}   val_loss={run2.metrics['val_loss']}")

        rule("3) framepin diff v1 -> v2  (drift, with move detection)")
        d = framepin.diff_manifests(repo.load_manifest(v1), repo.load_manifest(v2))
        s = d.summary()
        print(f"  +{s['added']} -{s['removed']} ~{s['modified']} "
              f"moved:{s['moved']} unchanged:{s['unchanged']}")
        for p in d.added:
            print(f"    + {p}")
        for m in d.modified:
            print(f"    ~ {m.path}")
        for mv in d.moved:
            print(f"    -> {mv.from_path}  =>  {mv.to_path}   (same bytes, relabeled)")

        rule("4) The point: was the val_loss regression code or data?")
        rep = framepin.compare_runs(run1.to_dict(), run2.to_dict(), metric="val_loss")
        m = rep["metrics"]["val_loss"]
        print(f"  val_loss: {m['a']} -> {m['b']}  (delta {m['delta']:+.4g})")
        if rep["data_changed"]:
            print("  ⚠ DATA CHANGED between these runs — the dataset version differs.")
            print("    Don't go hunting for a code bug: the data moved under you.")
        else:
            print("  ✓ same dataset version — attribute the change to code/params.")

        rule("Done")
        print("  Everything above is plain JSON under .framepin/ — commit it to git.")
        print(f"  (demo scratch dir: {workdir})")
        return 0
    finally:
        os.chdir(os.path.dirname(workdir) or "/")
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
