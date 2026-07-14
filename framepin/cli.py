"""Command-line interface: init/snapshot/verify/log/gc/diff/runs/show/regress."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__, hashing
from .manifest import snapshot, dumps as manifest_dumps
from .diff import diff_manifests
from .repo import Repo, RepoError
from .tracking import compare_runs
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def cmd_init(args) -> int:
    repo = Repo.init(args.path)
    print(f"initialized framepin store at {repo.store}")
    return 0


def cmd_snapshot(args) -> int:
    repo = Repo.open_or_init(".")
    if args.from_list:
        from .listfile import HashCache, snapshot_from_lists

        cache = HashCache(os.path.join(repo.store, "hashcache.json"))
        man = snapshot_from_lists(
            args.from_list, created_at=_now(), cache=cache, fast=args.fast,
            jobs=args.jobs,
        )
    elif args.dataset:
        man = snapshot(args.dataset, created_at=_now(), jobs=args.jobs)
    else:
        print("framepin: error: give a dataset directory or --from-list", file=sys.stderr)
        return 2
    repo.save_manifest(man)
    missing = sum(1 for f in man.files if f.hash == "missing:")
    if args.json:
        print(json.dumps({
            "root": man.root, "short": man.short, "file_count": man.file_count,
            "total_bytes": man.total_bytes, "missing": missing,
        }))
        return 0
    print(f"snapshot {man.short}  ({man.file_count} files, {man.total_bytes} bytes)")
    if missing:
        print(f"  ⚠ {missing} referenced path(s) do not exist (recorded as missing)")
    if args.verbose:
        print(f"  root: {man.root}")
        for f in man.files:
            print(f"  {hashing.short(f.hash)}  {f.size:>10}  {f.path}")
    return 0


def cmd_gc(args) -> int:
    """Prune dataset manifests that no run references (dry-run by default).

    Never touches a manifest referenced by any recorded run; keeps the newest
    --keep unreferenced ones as a safety margin.
    """
    repo = Repo.discover(".")
    referenced = set()
    for run in repo.list_runs():
        for d in run.get("datasets", []):
            if d.get("root"):
                referenced.add(d["root"])

    manifests = [repo.load_manifest(mid) for mid in repo.list_manifests()]
    unreferenced = sorted(
        (m for m in manifests if m.root not in referenced),
        key=lambda m: m.created_at, reverse=True)
    victims = unreferenced[max(0, args.keep):]

    if not victims:
        print(f"nothing to prune ({len(manifests)} manifests, "
              f"{len(referenced)} referenced by runs, keep={args.keep})")
        return 0
    for m in victims:
        if args.apply:
            os.remove(os.path.join(repo.datasets_dir, m.root + ".json"))
            print(f"pruned  {m.short}  {m.created_at or '(no date)'}  {m.dataset_path}")
        else:
            print(f"would prune  {m.short}  {m.created_at or '(no date)'}  {m.dataset_path}")
    if not args.apply:
        print(f"dry-run: {len(victims)} manifest(s) — rerun with --apply to delete")
    return 0


def cmd_log(args) -> int:
    """List stored dataset versions, newest first."""
    repo = Repo.discover(".")
    manifests = [repo.load_manifest(mid) for mid in repo.list_manifests()]
    if not manifests:
        print("no dataset versions recorded")
        return 0
    manifests.sort(key=lambda m: m.created_at, reverse=True)
    if args.json:
        print(json.dumps([
            {"root": m.root, "short": m.short, "created_at": m.created_at,
             "file_count": m.file_count, "total_bytes": m.total_bytes,
             "dataset_path": m.dataset_path}
            for m in manifests
        ]))
        return 0
    for m in manifests:
        print(f"{m.short}  {m.created_at or '(no date)':25s}  "
              f"{m.file_count:>7} files  {m.total_bytes:>13} bytes  {m.dataset_path}")
    return 0


def cmd_verify(args) -> int:
    """CI gate: exit 0 if the dataset still matches a pinned version, 3 on drift."""
    repo = Repo.discover(".")
    pinned = repo.load_manifest(args.against)
    if args.from_list:
        from .listfile import HashCache, snapshot_from_lists

        cache = HashCache(os.path.join(repo.store, "hashcache.json"))
        current = snapshot_from_lists(args.from_list, algo=pinned.algo, cache=cache,
                                      jobs=args.jobs)
        target = "lists: " + " ".join(args.from_list)
    elif args.dataset:
        current = snapshot(args.dataset, algo=pinned.algo, jobs=args.jobs)
        target = args.dataset
    else:
        print("framepin: error: give a dataset directory or --from-list", file=sys.stderr)
        return 2

    match = current.root == pinned.root
    if args.json:
        s = diff_manifests(pinned, current).summary() if not match else \
            {"added": 0, "removed": 0, "modified": 0, "moved": 0}
        print(json.dumps({
            "match": match, "pinned": pinned.root, "current": current.root,
            "summary": {k: s[k] for k in ("added", "removed", "modified", "moved")},
        }))
        return 0 if match else 3

    if match:
        print(f"✓ verified: {target} matches pinned version {pinned.short}")
        return 0

    d = diff_manifests(pinned, current)
    s = d.summary()
    print(
        f"✗ DATASET DRIFT vs pinned {pinned.short}:  "
        f"+{s['added']} -{s['removed']} ~{s['modified']} →{s['moved']}"
    )
    changes = ([f"  + {p}" for p in d.added]
               + [f"  - {p}" for p in d.removed]
               + [f"  ~ {m.path}" for m in d.modified]
               + [f"  → {m.from_path} -> {m.to_path}" for m in d.moved])
    for line in changes[:10]:
        print(line)
    if len(changes) > 10:
        print(f"  … and {len(changes) - 10} more (run `framepin diff`)")
    return 3


def cmd_diff(args) -> int:
    repo = Repo.discover(".")
    old = repo.load_manifest(args.old)
    new = repo.load_manifest(args.new)
    d = diff_manifests(old, new)
    s = d.summary()
    print(
        f"{old.short}..{new.short}  "
        f"+{s['added']} -{s['removed']} ~{s['modified']} "
        f"→{s['moved']} ={s['unchanged']}"
    )
    for p in d.added:
        print(f"  + {p}")
    for p in d.removed:
        print(f"  - {p}")
    for m in d.modified:
        print(f"  ~ {m.path}  ({hashing.short(m.old_hash)} -> {hashing.short(m.new_hash)})")
    for m in d.moved:
        print(f"  → {m.from_path} -> {m.to_path}")
    return 0


def cmd_runs(args) -> int:
    repo = Repo.discover(".")
    runs = repo.list_runs()
    if not runs:
        print("no runs recorded")
        return 0
    for r in runs:
        metrics = " ".join(f"{k}={v}" for k, v in sorted(r.get("metrics", {}).items()))
        data = ",".join(hashing.short(d["root"]) for d in r.get("datasets", []))
        print(
            f"{r['id']}  {r.get('created_at','')}  "
            f"{r.get('name','') or '-':<16}  [{data}]  {metrics}"
        )
    return 0


def cmd_show(args) -> int:
    repo = Repo.discover(".")
    r = repo.load_run(args.run)
    print(f"run     {r['id']}  ({r.get('name','')})")
    print(f"created {r.get('created_at','')}   status {r.get('status','')}")
    print(f"commit  {r.get('git_commit','') or '(none)'}")
    if r.get("params"):
        print("params:")
        for k, v in sorted(r["params"].items()):
            print(f"  {k} = {v}")
    if r.get("metrics"):
        print("metrics:")
        for k, v in sorted(r["metrics"].items()):
            print(f"  {k} = {v}")
    print("datasets (lineage):")
    for d in r.get("datasets", []):
        lbl = f" {d['label']}" if d.get("label") else ""
        print(
            f"  {hashing.short(d['root'])}{lbl}  "
            f"{d.get('file_count','?')} files  {d.get('path','')}"
        )
    return 0


def cmd_regress(args) -> int:
    repo = Repo.discover(".")
    a = repo.load_run(args.run_a)
    b = repo.load_run(args.run_b)
    rep = compare_runs(a, b, metric=args.metric or "")
    print(f"regress {a['id']} -> {b['id']}")
    for k, m in sorted(rep["metrics"].items()):
        delta = m["delta"]
        darrow = ""
        if isinstance(delta, bool):
            pass
        elif isinstance(delta, int):
            darrow = f"  (Δ {'+' if delta >= 0 else ''}{delta})"
        elif isinstance(delta, float):
            darrow = f"  (Δ {'+' if delta >= 0 else ''}{delta:.6g})"
        print(f"  {k}: {m['a']} -> {m['b']}{darrow}")
    print()
    if rep["data_changed"]:
        print("  ⚠ DATA CHANGED between these runs — the dataset version differs.")
        print("    A metric move here cannot be attributed to code alone.")
    else:
        print("  ✓ same dataset version — metric changes are attributable to code/params.")
    if rep["code_changed"]:
        print("  • code/commit differs (or commit unknown) between the runs.")
    else:
        print("  • same git commit.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="framepin",
        description="A lockfile for your video/sequence-ML datasets and experiments.",
    )
    p.add_argument("--version", action="version", version=f"framepin {__version__}")
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("init", help="create a .framepin store")
    sp.add_argument("path", nargs="?", default=".")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser(
        "snapshot", help="version a dataset directory (or path-list txt files) by content"
    )
    sp.add_argument("dataset", nargs="?", default=None)
    sp.add_argument(
        "--from-list",
        nargs="+",
        metavar="LIST_TXT",
        help="dataset = these list files + every path they reference "
        "(one path per line, # comments ok; multiple lists are deduped/unioned)",
    )
    sp.add_argument(
        "--fast",
        action="store_true",
        help="fingerprint referenced files by size+mtime instead of content "
        "(fast for 100k+ files; take periodic full snapshots as anchors)",
    )
    sp.add_argument("--jobs", type=int, default=hashing.DEFAULT_JOBS,
                    help="concurrent hashing threads (default 4)")
    sp.add_argument("--json", action="store_true",
                    help="machine-readable output (for agents/CI)")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_snapshot)

    sp = sub.add_parser("gc", help="prune dataset versions no run references (dry-run by default)")
    sp.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    sp.add_argument("--keep", type=int, default=5,
                    help="always keep the newest N unreferenced versions (default 5)")
    sp.set_defaults(func=cmd_gc)

    sp = sub.add_parser("log", help="list stored dataset versions, newest first")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_log)

    sp = sub.add_parser(
        "verify",
        help="CI gate: exit 0 if a dataset still matches a pinned version, 3 on drift",
    )
    sp.add_argument("dataset", nargs="?", default=None)
    sp.add_argument("--against", required=True, metavar="VERSION",
                    help="pinned dataset version (full root or short prefix)")
    sp.add_argument("--from-list", nargs="+", metavar="LIST_TXT",
                    help="verify a path-list dataset instead of a directory")
    sp.add_argument("--jobs", type=int, default=hashing.DEFAULT_JOBS,
                    help="concurrent hashing threads (default 4)")
    sp.add_argument("--json", action="store_true",
                    help="machine-readable output (for agents/CI)")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("diff", help="show drift between two dataset versions")
    sp.add_argument("old")
    sp.add_argument("new")
    sp.set_defaults(func=cmd_diff)

    sp = sub.add_parser("runs", help="list tracked experiment runs")
    sp.set_defaults(func=cmd_runs)

    sp = sub.add_parser("show", help="show a run's full lineage")
    sp.add_argument("run")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("regress", help="compare two runs: code vs data")
    sp.add_argument("run_a")
    sp.add_argument("run_b")
    sp.add_argument("-m", "--metric", default="", help="focus a single metric")
    sp.set_defaults(func=cmd_regress)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except (RepoError, FileNotFoundError, NotADirectoryError) as e:
        print(f"framepin: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
