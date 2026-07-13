# framepin

[![PyPI](https://img.shields.io/pypi/v/framepin)](https://pypi.org/project/framepin/)

**A lockfile for your video/sequence-ML datasets and experiments — reproduce any run without copying a single frame.**

`framepin` pins the *exact* data a training run saw, links it to the run's
params/metrics/commit, and tells you — when a metric moves — whether it was the
**code** or the **data**. Zero dependencies, no server, no data copies. Just
JSON that commits cleanly to git.

```bash
pip install framepin   # zero dependencies — pure standard library
```

> Status: **v0.1 alpha.** Core (snapshot / diff / track / regress) works and is
> tested. APIs may shift before 1.0.

---

## The problem

If you train models on video or sequence data, this has happened to you:

- A run from three weeks ago had a better `val_loss`. Which dataset version was
  it? You re-sampled frames and re-labeled twice since. **Gone.**
- `val_loss` got worse after a change. Was it your code… or did someone swap out
  30% of the clips? **Nobody can say quickly.**
- `wandb`/`mlflow` track the *run* but treat the dataset as an opaque string.
  `dvc` versions the *data* but is heavy and lives apart from your metrics.
  Neither answers **"is this run reproducible from these exact bytes?"**

Video datasets are big, churn constantly, and get reorganized — so a directory
reshuffle looks like "half my data changed" in a naive diff.

## What framepin does

```python
import framepin

with framepin.track(name="baseline", params={"lr": 3e-4}) as run:
    run.use_dataset("data/clips")      # content-hashes + pins this exact version
    model = train(...)
    run.log_metric("val_loss", 0.21)   # pinned to that version forever
```

Then, three weeks later:

```bash
$ framepin regress <old_run> <new_run> -m val_loss
regress 02189c0fbd14 -> ab09cc84a35a
  val_loss: 0.21 -> 0.28  (Δ +0.07)

  ⚠ DATA CHANGED between these runs — the dataset version differs.
    A metric move here cannot be attributed to code alone.
```

That last line is the whole point. **framepin separates "the code regressed"
from "the data changed under you."**

## Before / after

**Before** — the dataset is a string you hope is still true:

```python
run.log({"dataset": "data/clips (v3?)", "val_loss": 0.21})
# 3 weeks later: what was v3? which frames? was it re-labeled? 🤷
```

**After** — the dataset is a content hash you can reproduce and diff:

```python
with framepin.track(name="baseline") as run:
    run.use_dataset("data/clips")   # -> version c29aa729c669 (Merkle root of every byte)
    run.log_metric("val_loss", 0.21)
# later: `framepin show <run>` -> exact version; `framepin diff v1 v2` -> what changed
```

## 60-second quickstart

```bash
cd your-project
framepin init                     # creates .framepin/  (commit it to git)

framepin snapshot data/clips      # -> snapshot c29aa729c669 (N files, no copies)
# ... change / relabel / resample your data ...
framepin snapshot data/clips      # -> a new version id

framepin diff c29aa729c669 <new>  # added / removed / modified / MOVED
```

### Datasets defined by path-list files (train.txt of absolute paths)

Many teams don't train on "a directory" — they train on **txt manifests**: one
absolute path per line (500k clips, several lists concatenated per experiment).
framepin pins that whole construct — the list files *and* the bytes they
point at:

```bash
framepin snapshot --from-list train_urban.txt train_highway.txt
# -> one version id covering: both txt files + every referenced clip
#    re-encode one clip, or edit one line  -> different version id
#    a listed path that no longer exists   -> "⚠ recorded as missing"
```

Repeated paths across lists are deduped. For 100k+ files, content hashes are
cached (`.framepin/hashcache.json`) so later snapshots only re-read files
whose size/mtime changed; `--fast` skips content reads entirely (size+mtime
fingerprint) for routine checks — take a periodic full snapshot as your anchor
of record.

```python
man = framepin.snapshot_from_lists(["train_urban.txt", "train_highway.txt"])
with framepin.track(name="exp-8") as run:
    run.use_dataset(man)               # pins the exact list+content version
```

Track experiments from Python:

```python
import framepin

with framepin.track(name="exp-7", params={"lr": 3e-4, "aug": "mixup"}) as run:
    run.use_dataset("data/clips")
    for epoch in range(epochs):
        ...
    run.log_metric("val_loss", best_val)
    run.log_metric("map50", 0.63)
```

Inspect lineage and compare:

```bash
framepin runs                     # every run, its metrics, and its data version
framepin show <run>               # full lineage: run -> dataset version -> files
framepin regress <a> <b> -m map50 # metric delta + "was it code or data?"
```

## Why not W&B / MLflow / DVC?

| | framepin | W&B / MLflow | DVC |
|---|---|---|---|
| Pins run ↔ **exact dataset version** | ✅ first-class | ⚠️ manual string/artifact | ⚠️ separate from metrics |
| "Code vs data" regression answer | ✅ `regress` | ❌ | ❌ |
| Detects **moved/renamed** files (not add+remove) | ✅ | — | partial |
| Copies your data | ❌ never (hashes only) | ⚠️ artifacts | ✅ cache/remote |
| Server / account required | ❌ | ⚠️ (or self-host) | ❌ |
| Runtime dependencies | **0** | many | several |
| Git-friendly plain-JSON store | ✅ | ❌ | ✅ (pointers) |

framepin is deliberately small. It is **not** a full experiment platform — it
does the one thing those tools under-serve: making the *dataset version* a
first-class, reproducible, diffable part of every run. Use it alongside W&B if
you like; framepin just owns the data-lineage question.

## How it works

- **Snapshot** walks a directory, streams a SHA-256 over each file, and records a
  deterministic manifest keyed by a **Merkle root** of `(path, content-hash)`
  pairs. Same bytes → same version id, regardless of walk order or timestamps.
  Your data is never moved or copied — the manifest is a few KB of JSON.
- **Diff** classifies changes into added / removed / modified / **moved**, so a
  reorg of identical clips reads as moves, not churn.
- **Track** records a run's params, metrics, git commit, and the dataset
  version(s) it consumed, into `.framepin/runs/<id>.json`.
- **Regress** compares two runs' metrics *and* their pinned dataset versions.

Everything lives under `.framepin/` as plain, sorted JSON. Commit it.

## Install from source

```bash
git clone https://github.com/boogy-ro/framepin
cd framepin
pip install -e .          # or just add the repo to PYTHONPATH — no deps
python -m unittest discover -s tests
```

Requires Python ≥ 3.9. No third-party packages.

See the whole workflow in ~2 seconds (builds a throwaway dataset, simulates a
re-label + re-encode, then shows the code-vs-data verdict):

```bash
python3 examples/quickstart_demo.py
```

## Roadmap

- `framepin gc` / remote manifest registry for teams
- CI gate: fail a build when the dataset regresses vs a pinned baseline
- Optional integrations (export runs to W&B / MLflow)
- Per-split / per-label manifests for stratified datasets

Feedback and issues very welcome — the niche (video/sequence ML data lineage) is
exactly where this should earn its keep or die. Tell me where it falls short.

**Using this on a team?** Comment on [the team-features issue](https://github.com/boogy-ro/framepin/issues/1)
(CI gate / shared registry / audit reports) — it decides what gets built next.

## FAQ

**How do I version an ML dataset without copying it?**
`framepin snapshot data/clips` content-hashes every file into an immutable
version id (a Merkle root). Your data never moves; the snapshot is a few KB of
JSON you commit to git.

**How do I know which dataset version an old training run used?**
If the run was wrapped in `framepin.track(...)` with `run.use_dataset(...)`,
`framepin show <run>` prints its exact pinned dataset version and file list.

**My metric regressed — was it my code or my data?**
`framepin regress <old_run> <new_run> -m val_loss` compares the two runs'
metrics *and* their pinned dataset versions, and prints either
`⚠ DATA CHANGED` or `✓ same dataset version` so you know where to look.

**My dataset is a txt file of absolute paths (500k clips), not a directory.**
`framepin snapshot --from-list train_a.txt train_b.txt` pins the list files
and every file they reference, deduped across lists. Dead paths are recorded
as missing. Use `--fast` for size+mtime fingerprints on huge datasets.

**Does framepin replace W&B, MLflow, or DVC?**
No. It runs alongside them and owns one thing they under-serve: the
dataset-version ↔ run link and the code-vs-data question. No server, no
account, zero dependencies.

**How do I detect that files were renamed/reorganized rather than changed?**
`framepin diff v1 v2` pairs identical-content files across paths and reports
them as MOVED instead of added+removed.

## License

MIT — see [LICENSE](LICENSE).
