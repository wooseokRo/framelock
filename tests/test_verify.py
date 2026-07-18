import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

from framepin.cli import main


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="framepin-verify-")
        self.prev = os.getcwd()
        os.chdir(self.dir)
        os.makedirs("data")
        for i in range(3):
            with open(f"data/clip_{i}.bin", "wb") as fh:
                fh.write(f"frames-{i}".encode() * 16)
        main(["init"])

    def tearDown(self):
        os.chdir(self.prev)
        shutil.rmtree(self.dir, ignore_errors=True)

    def snapshot_id(self):
        from framepin.repo import Repo
        repo = Repo.discover(".")
        main(["snapshot", "data"])
        return repo.list_manifests()[-1]

    def test_verify_passes_when_unchanged(self):
        vid = self.snapshot_id()
        self.assertEqual(main(["verify", "data", "--against", vid[:12]]), 0)

    def test_verify_fails_on_drift(self):
        vid = self.snapshot_id()
        with open("data/clip_1.bin", "wb") as fh:
            fh.write(b"REENCODED" * 16)
        self.assertEqual(main(["verify", "data", "--against", vid[:12]]), 3)

    def test_verify_from_list(self):
        paths = [os.path.abspath(f"data/clip_{i}.bin") for i in range(3)]
        with open("train.txt", "w") as fh:
            fh.write("\n".join(paths) + "\n")
        from framepin.repo import Repo
        repo = Repo.discover(".")
        main(["snapshot", "--from-list", "train.txt"])
        vid = repo.list_manifests()[-1]
        self.assertEqual(main(["verify", "--from-list", "train.txt", "--against", vid[:12]]), 0)
        os.remove(paths[2])  # a listed file disappears -> drift
        self.assertEqual(main(["verify", "--from-list", "train.txt", "--against", vid[:12]]), 3)

    def test_verify_allow_passes_when_drift_fully_allowed(self):
        # manifest paths are relative to the dataset dir itself, e.g. "clip_1.bin"
        vid = self.snapshot_id()
        with open("data/clip_1.bin", "wb") as fh:
            fh.write(b"REENCODED" * 16)
        self.assertEqual(
            main(["verify", "data", "--against", vid[:12], "--allow", "clip_1.bin"]), 0
        )

    def test_verify_allow_fails_on_mixed_drift(self):
        vid = self.snapshot_id()
        with open("data/clip_0.bin", "wb") as fh:
            fh.write(b"REENCODED0" * 16)
        with open("data/clip_1.bin", "wb") as fh:
            fh.write(b"REENCODED1" * 16)
        self.assertEqual(
            main(["verify", "data", "--against", vid[:12], "--allow", "clip_1.bin"]), 3
        )

    def test_verify_allow_json_fields(self):
        vid = self.snapshot_id()
        with open("data/clip_1.bin", "wb") as fh:
            fh.write(b"REENCODED" * 16)

        # fully allowed -> gate pass, exit 0
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main([
                "verify", "data", "--against", vid[:12],
                "--allow", "clip_1.bin", "--json",
            ])
        self.assertEqual(rc, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["gate"], "pass")
        self.assertEqual(payload["allowed"], 1)
        self.assertFalse(payload["match"])

        # not allowed at all -> gate fail, exit 3
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main([
                "verify", "data", "--against", vid[:12],
                "--allow", "nope_*.bin", "--json",
            ])
        self.assertEqual(rc, 3)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["gate"], "fail")
        self.assertEqual(payload["allowed"], 0)

    def test_verify_allow_removed_and_moved_paths(self):
        vid = self.snapshot_id()
        os.remove("data/clip_2.bin")  # removed
        os.rename("data/clip_1.bin", "data/clip_1_renamed.bin")  # moved (same content)

        # both sides of the move and the removed path are allowed -> pass
        self.assertEqual(
            main([
                "verify", "data", "--against", vid[:12],
                "--allow", "clip_2.bin", "--allow", "clip_1*",
            ]),
            0,
        )

        # move only partially covered (destination not allowed) -> still fails
        self.assertEqual(
            main([
                "verify", "data", "--against", vid[:12],
                "--allow", "clip_2.bin", "--allow", "clip_1.bin",
            ]),
            3,
        )


if __name__ == "__main__":
    unittest.main()
