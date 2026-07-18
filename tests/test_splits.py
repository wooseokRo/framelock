import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

from framepin import manifest as m
from framepin.cli import main


def _write(root, rel, data: bytes):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


class TestComputeSplits(unittest.TestCase):
    def test_determinism_and_top_level_root_unaffected(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "train/a.bin", b"aaaa")
            _write(d, "train/b.bin", b"bbbbbb")
            _write(d, "val/c.bin", b"cc")
            split_globs = {"train": "train/*", "val": "val/*"}

            man1 = m.snapshot(d, split_globs=split_globs)
            man2 = m.snapshot(d, split_globs=split_globs)
            self.assertEqual(man1.splits["train"]["root"], man2.splits["train"]["root"])
            self.assertEqual(man1.splits["val"]["root"], man2.splits["val"]["root"])

            man_no_splits = m.snapshot(d)
            self.assertEqual(man1.root, man_no_splits.root)

    def test_isolation_between_splits(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "train/a.bin", b"aaaa")
            _write(d, "val/b.bin", b"bbbb")
            split_globs = {"train": "train/*", "val": "val/*"}

            man1 = m.snapshot(d, split_globs=split_globs)
            _write(d, "train/a.bin", b"aaaa-changed")
            man2 = m.snapshot(d, split_globs=split_globs)

            self.assertNotEqual(man1.splits["train"]["root"], man2.splits["train"]["root"])
            self.assertEqual(man1.splits["val"]["root"], man2.splits["val"]["root"])

    def test_split_glob_matching_zero_files(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "train/a.bin", b"aaaa")
            man = m.snapshot(d, split_globs={"test": "test/*"})
            self.assertEqual(man.splits["test"]["file_count"], 0)
            self.assertEqual(man.splits["test"]["total_bytes"], 0)

    def test_compute_splits_directly(self):
        entries = [
            m.FileEntry(path="train/a.bin", hash="h1", size=4),
            m.FileEntry(path="val/b.bin", hash="h2", size=6),
        ]
        splits = m.compute_splits(entries, {"train": "train/*", "val": "val/*"}, algo="sha256")
        self.assertEqual(splits["train"]["file_count"], 1)
        self.assertEqual(splits["train"]["total_bytes"], 4)
        self.assertEqual(splits["val"]["file_count"], 1)
        self.assertEqual(splits["val"]["total_bytes"], 6)


class TestSplitsBackCompat(unittest.TestCase):
    def test_from_dict_without_splits_key(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "a.txt", b"x")
            man = m.snapshot(d)
            d_dict = man.to_dict()
            self.assertIn("splits", d_dict)
            del d_dict["splits"]
            loaded = m.Manifest.from_dict(d_dict)
            self.assertEqual(loaded.splits, {})

    def test_splits_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "train/a.bin", b"aaaa")
            man = m.snapshot(d, split_globs={"train": "train/*"})
            back = m.loads(m.dumps(man))
            self.assertEqual(back.splits, man.splits)


class SplitsCliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="framepin-splits-")
        self.prev = os.getcwd()
        os.chdir(self.dir)
        os.makedirs("data/train")
        os.makedirs("data/val")
        for i in range(3):
            with open(f"data/train/clip_{i}.bin", "wb") as fh:
                fh.write(f"train-{i}".encode() * 16)
        for i in range(2):
            with open(f"data/val/clip_{i}.bin", "wb") as fh:
                fh.write(f"val-{i}".encode() * 16)
        main(["init"])

    def tearDown(self):
        os.chdir(self.prev)
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_snapshot_split_records_splits_in_manifest(self):
        from framepin.repo import Repo

        repo = Repo.discover(".")
        rc = main(["snapshot", "data", "--split", "train=train/*", "--split", "val=val/*"])
        self.assertEqual(rc, 0)
        vid = repo.list_manifests()[-1]
        man = repo.load_manifest(vid)
        self.assertIn("train", man.splits)
        self.assertIn("val", man.splits)
        self.assertEqual(man.splits["train"]["file_count"], 3)
        self.assertEqual(man.splits["val"]["file_count"], 2)

    def test_snapshot_split_json_output(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main([
                "snapshot", "data", "--split", "train=train/*", "--split", "val=val/*",
                "--json",
            ])
        self.assertEqual(rc, 0)
        payload = json.loads(out.getvalue())
        self.assertIn("splits", payload)
        self.assertEqual(payload["splits"]["train"]["file_count"], 3)
        self.assertEqual(payload["splits"]["val"]["file_count"], 2)

    def test_snapshot_split_malformed_returns_2(self):
        rc = main(["snapshot", "data", "--split", "foo"])
        self.assertEqual(rc, 2)

    def test_verify_drift_names_changed_split(self):
        from framepin.repo import Repo

        repo = Repo.discover(".")
        main(["snapshot", "data", "--split", "train=train/*", "--split", "val=val/*"])
        vid = repo.list_manifests()[-1]

        with open("data/train/clip_0.bin", "wb") as fh:
            fh.write(b"REENCODED" * 16)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main(["verify", "data", "--against", vid[:12]])
        self.assertEqual(rc, 3)
        output = out.getvalue()
        self.assertIn("splits changed: train", output)
        self.assertIn("splits unchanged: val", output)

    def test_verify_drift_json_splits_changed(self):
        from framepin.repo import Repo

        repo = Repo.discover(".")
        main(["snapshot", "data", "--split", "train=train/*", "--split", "val=val/*"])
        vid = repo.list_manifests()[-1]

        with open("data/train/clip_0.bin", "wb") as fh:
            fh.write(b"REENCODED" * 16)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main(["verify", "data", "--against", vid[:12], "--json"])
        self.assertEqual(rc, 3)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["splits_changed"], ["train"])


if __name__ == "__main__":
    unittest.main()
