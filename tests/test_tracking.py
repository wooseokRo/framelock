import os
import tempfile
import unittest

import framepin
from framepin.repo import Repo
from framepin.tracking import compare_runs


def _write(root, rel, data: bytes):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


class TestTracking(unittest.TestCase):
    def test_run_persists_and_links_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "data/a.mp4", b"aaa")
            repo = Repo.init(d)
            with framepin.track(name="exp1", params={"lr": 0.01}, repo=repo) as run:
                root = run.use_dataset(os.path.join(d, "data"))
                run.log_metric("val_loss", 0.5)
            # run file exists and links the dataset version
            loaded = repo.load_run(run.id)
            self.assertEqual(loaded["name"], "exp1")
            self.assertEqual(loaded["metrics"]["val_loss"], 0.5)
            self.assertEqual(loaded["params"]["lr"], 0.01)
            self.assertEqual(loaded["datasets"][0]["root"], root)
            self.assertEqual(loaded["status"], "finished")
            # the manifest was saved too
            self.assertIn(root, repo.list_manifests())

    def test_status_failed_on_exception(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Repo.init(d)
            try:
                with framepin.track(name="boom", repo=repo) as run:
                    rid = run.id
                    raise ValueError("training exploded")
            except ValueError:
                pass
            loaded = repo.load_run(rid)
            self.assertEqual(loaded["status"], "failed")

    def test_git_commit_graceful_without_repo(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Repo.init(d)
            with framepin.track(repo=repo) as run:
                run.log_metric("x", 1)
            loaded = repo.load_run(run.id)
            # No git repo at a bare temp dir -> empty string, no crash.
            self.assertIn("git_commit", loaded)
            self.assertIsInstance(loaded["git_commit"], str)

    def test_use_dataset_by_existing_ref(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "data/a.mp4", b"aaa")
            repo = Repo.init(d)
            man = framepin.snapshot(os.path.join(d, "data"))
            repo.save_manifest(man)
            with framepin.track(repo=repo) as run:
                root = run.use_dataset(man.short)  # short prefix ref
            self.assertEqual(root, man.root)

    def test_compare_runs_flags_data_change(self):
        a = {"id": "a", "git_commit": "c1", "metrics": {"val_loss": 0.4},
             "datasets": [{"root": "D1"}]}
        b = {"id": "b", "git_commit": "c1", "metrics": {"val_loss": 0.6},
             "datasets": [{"root": "D2"}]}
        rep = compare_runs(a, b, metric="val_loss")
        self.assertTrue(rep["data_changed"])
        self.assertAlmostEqual(rep["metrics"]["val_loss"]["delta"], 0.2)
        self.assertFalse(rep["code_changed"])  # same commit

    def test_compare_runs_same_data(self):
        a = {"id": "a", "git_commit": "c1", "metrics": {"m": 1.0},
             "datasets": [{"root": "D1"}]}
        b = {"id": "b", "git_commit": "c2", "metrics": {"m": 2.0},
             "datasets": [{"root": "D1"}]}
        rep = compare_runs(a, b)
        self.assertFalse(rep["data_changed"])
        self.assertTrue(rep["code_changed"])


if __name__ == "__main__":
    unittest.main()
