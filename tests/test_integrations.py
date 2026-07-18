import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import framepin
from framepin.repo import Repo
from framepin import integrations


def _write(root, rel, data: bytes):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def _make_run(tmp_dir, label=""):
    _write(tmp_dir, "data/a.mp4", b"aaa")
    repo = Repo.init(tmp_dir)
    with framepin.track(name="exp1", params={"lr": 0.01}, repo=repo) as run:
        run.use_dataset(os.path.join(tmp_dir, "data"), label=label)
        run.log_metric("val_loss", 0.5)
    return run


class TestToMlflow(unittest.TestCase):
    def setUp(self):
        self.set_tags_calls = []
        self.log_params_calls = []

        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.set_tags = lambda tags: self.set_tags_calls.append(tags)
        fake_mlflow.log_params = lambda params: self.log_params_calls.append(params)

        self._patcher = patch.dict(sys.modules, {"mlflow": fake_mlflow})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_forwards_run_id_and_dataset_tags_by_label(self):
        with tempfile.TemporaryDirectory() as d:
            run = _make_run(d, label="clips")
            integrations.to_mlflow(run)

        self.assertEqual(len(self.set_tags_calls), 1)
        tags = self.set_tags_calls[0]
        self.assertEqual(tags["framepin.run_id"], run.id)
        self.assertEqual(tags["framepin.dataset.clips"], run.datasets[0]["root"])
        if run.git_commit:
            self.assertEqual(tags["framepin.git_commit"], run.git_commit)
        else:
            self.assertNotIn("framepin.git_commit", tags)

    def test_dataset_tag_falls_back_to_index_when_no_label(self):
        with tempfile.TemporaryDirectory() as d:
            run = _make_run(d, label="")
            integrations.to_mlflow(run)

        tags = self.set_tags_calls[0]
        self.assertEqual(tags["framepin.dataset.0"], run.datasets[0]["root"])

    def test_skips_empty_git_commit(self):
        run_dict = {
            "id": "r1",
            "git_commit": "",
            "params": {},
            "metrics": {},
            "datasets": [],
        }
        integrations.to_mlflow(run_dict)

        tags = self.set_tags_calls[0]
        self.assertNotIn("framepin.git_commit", tags)

    def test_includes_git_commit_when_present(self):
        run_dict = {
            "id": "r1",
            "git_commit": "deadbeef",
            "params": {},
            "metrics": {},
            "datasets": [],
        }
        integrations.to_mlflow(run_dict)

        tags = self.set_tags_calls[0]
        self.assertEqual(tags["framepin.git_commit"], "deadbeef")

    def test_log_params_only_when_flag_set(self):
        run_dict = {
            "id": "r1",
            "git_commit": "",
            "params": {"lr": 0.01},
            "metrics": {},
            "datasets": [],
        }
        integrations.to_mlflow(run_dict)
        self.assertEqual(self.log_params_calls, [])

        integrations.to_mlflow(run_dict, log_params=True)
        self.assertEqual(self.log_params_calls, [{"lr": 0.01}])

    def test_accepts_plain_dict(self):
        run_dict = {
            "id": "abc",
            "git_commit": "",
            "params": {},
            "metrics": {},
            "datasets": [{"root": "R1", "label": "train"}],
        }
        integrations.to_mlflow(run_dict)

        tags = self.set_tags_calls[0]
        self.assertEqual(tags["framepin.run_id"], "abc")
        self.assertEqual(tags["framepin.dataset.train"], "R1")

    def test_custom_prefix(self):
        run_dict = {
            "id": "abc",
            "git_commit": "",
            "params": {},
            "metrics": {},
            "datasets": [],
        }
        integrations.to_mlflow(run_dict, prefix="mypkg.")
        tags = self.set_tags_calls[0]
        self.assertEqual(tags["mypkg.run_id"], "abc")


class TestToMlflowMissing(unittest.TestCase):
    def test_raises_runtime_error_when_not_installed(self):
        with patch.dict(sys.modules, {"mlflow": None}):
            with self.assertRaises(RuntimeError) as ctx:
                integrations.to_mlflow({"id": "r1"})
        self.assertIn("pip install mlflow", str(ctx.exception))


class _FakeConfig(dict):
    def update(self, other, allow_val_change=False):
        self.last_allow_val_change = allow_val_change
        dict.update(self, other)


class _FakeWandbRun:
    def __init__(self):
        self.config = _FakeConfig()


class TestToWandb(unittest.TestCase):
    def setUp(self):
        self.fake_run = _FakeWandbRun()

        fake_wandb = types.ModuleType("wandb")
        fake_wandb.run = self.fake_run

        self._patcher = patch.dict(sys.modules, {"wandb": fake_wandb})
        self._patcher.start()
        self._fake_wandb_module = fake_wandb

    def tearDown(self):
        self._patcher.stop()

    def test_updates_config_with_run_id_and_dataset_roots(self):
        with tempfile.TemporaryDirectory() as d:
            run = _make_run(d, label="clips")
            integrations.to_wandb(run)

        self.assertEqual(self.fake_run.config["framepin.run_id"], run.id)
        self.assertEqual(
            self.fake_run.config["framepin.dataset.clips"], run.datasets[0]["root"]
        )
        self.assertTrue(self.fake_run.config.last_allow_val_change)

    def test_dataset_tag_falls_back_to_index_when_no_label(self):
        with tempfile.TemporaryDirectory() as d:
            run = _make_run(d, label="")
            integrations.to_wandb(run)

        self.assertEqual(
            self.fake_run.config["framepin.dataset.0"], run.datasets[0]["root"]
        )

    def test_uses_explicit_run_obj_over_wandb_run(self):
        other_run = _FakeWandbRun()
        run_dict = {
            "id": "abc",
            "git_commit": "",
            "params": {},
            "metrics": {},
            "datasets": [],
        }
        integrations.to_wandb(run_dict, run_obj=other_run)

        self.assertEqual(other_run.config["framepin.run_id"], "abc")
        self.assertNotIn("framepin.run_id", self.fake_run.config)

    def test_raises_runtime_error_when_no_active_run(self):
        self._fake_wandb_module.run = None
        run_dict = {
            "id": "abc",
            "git_commit": "",
            "params": {},
            "metrics": {},
            "datasets": [],
        }
        with self.assertRaises(RuntimeError) as ctx:
            integrations.to_wandb(run_dict)
        self.assertIn("wandb.init()", str(ctx.exception))

    def test_skips_empty_git_commit(self):
        run_dict = {
            "id": "abc",
            "git_commit": "",
            "params": {},
            "metrics": {},
            "datasets": [],
        }
        integrations.to_wandb(run_dict)
        self.assertNotIn("framepin.git_commit", self.fake_run.config)

    def test_includes_git_commit_when_present(self):
        run_dict = {
            "id": "abc",
            "git_commit": "deadbeef",
            "params": {},
            "metrics": {},
            "datasets": [],
        }
        integrations.to_wandb(run_dict)
        self.assertEqual(self.fake_run.config["framepin.git_commit"], "deadbeef")


class TestToWandbMissing(unittest.TestCase):
    def test_raises_runtime_error_when_not_installed(self):
        with patch.dict(sys.modules, {"wandb": None}):
            with self.assertRaises(RuntimeError) as ctx:
                integrations.to_wandb({"id": "r1"})
        self.assertIn("pip install wandb", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
