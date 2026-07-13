import contextlib
import io
import os
import tempfile
import unittest

import framepin
from framepin.cli import main
from framepin.repo import Repo


def _write(root, rel, data: bytes):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


@contextlib.contextmanager
def _chdir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


class TestCLI(unittest.TestCase):
    def test_end_to_end_smoke(self):
        with tempfile.TemporaryDirectory() as d, _chdir(d):
            _write(d, "data/a.mp4", b"aaa")
            _write(d, "data/b.mp4", b"bbb")

            code, out = _run(["init"])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.isdir(os.path.join(d, ".framepin")))

            code, out = _run(["snapshot", "data"])
            self.assertEqual(code, 0)
            self.assertIn("snapshot", out)

            # mutate the dataset and snapshot again
            _write(d, "data/b.mp4", b"BBBB")  # modify
            _write(d, "data/c.mp4", b"ccc")   # add
            code, out2 = _run(["snapshot", "data"])
            self.assertEqual(code, 0)

            repo = Repo.discover(d)
            versions = repo.list_manifests()
            self.assertEqual(len(versions), 2)
            v_old, v_new = versions[0][:12], versions[1][:12]

            # diff should show one modified and one added (order of versions may
            # vary, so just assert the command works and reports changes)
            code, dout = _run(["diff", v_old, v_new])
            self.assertEqual(code, 0)
            self.assertTrue(("~" in dout) or ("+" in dout))

    def test_run_lineage_and_regress(self):
        with tempfile.TemporaryDirectory() as d, _chdir(d):
            _write(d, "data/a.mp4", b"aaa")
            repo = Repo.init(d)

            # run 1 on dataset v1
            with framepin.track(name="r1", repo=repo) as run1:
                run1.use_dataset(os.path.join(d, "data"))
                run1.log_metric("val_loss", 0.5)

            # change data, run 2 on dataset v2
            _write(d, "data/a.mp4", b"aaaa-changed")
            with framepin.track(name="r2", repo=repo) as run2:
                run2.use_dataset(os.path.join(d, "data"))
                run2.log_metric("val_loss", 0.7)

            code, out = _run(["runs"])
            self.assertEqual(code, 0)
            self.assertIn(run1.id, out)
            self.assertIn(run2.id, out)

            code, out = _run(["show", run1.id])
            self.assertEqual(code, 0)
            self.assertIn("lineage", out)

            code, out = _run(["regress", run1.id, run2.id, "-m", "val_loss"])
            self.assertEqual(code, 0)
            self.assertIn("DATA CHANGED", out)  # dataset differed between runs

    def test_no_command_prints_help(self):
        code, _ = _run([])
        self.assertEqual(code, 1)

    def test_diff_without_store_errors(self):
        with tempfile.TemporaryDirectory() as d, _chdir(d):
            code = main(["diff", "aa", "bb"])
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
