import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

from framepin.cli import main
from framepin.repo import Repo


class PinTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="framepin-pin-")
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

    def read_pinfile(self, path="framepin.pin"):
        with open(path, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh.readlines()]
        non_comment = [ln for ln in lines if ln and not ln.startswith("#")]
        return lines, non_comment

    # -- pin DATASET -----------------------------------------------------

    def test_pin_dataset_writes_pinfile_with_full_root(self):
        rc = main(["pin", "data"])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists("framepin.pin"))

        repo = Repo.discover(".")
        vid = repo.list_manifests()[-1]
        man = repo.load_manifest(vid)

        lines, non_comment = self.read_pinfile()
        self.assertEqual(len(non_comment), 1)
        self.assertEqual(non_comment[0], man.root)
        self.assertEqual(len(man.root), 64)  # full hex root, not short

    def test_pin_output_mentions_pinfile(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main(["pin", "data"])
        self.assertEqual(rc, 0)
        output = out.getvalue()
        self.assertIn("pinned", output)
        self.assertIn("framepin.pin", output)

    def test_pin_json_fields(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main(["pin", "data", "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(out.getvalue())

        repo = Repo.discover(".")
        vid = repo.list_manifests()[-1]
        man = repo.load_manifest(vid)

        self.assertEqual(payload["root"], man.root)
        self.assertEqual(payload["short"], man.short)
        self.assertEqual(payload["pinfile"], "framepin.pin")

    # -- pin --version -----------------------------------------------------

    def test_pin_version_resolves_existing_without_new_manifest(self):
        main(["snapshot", "data"])
        repo = Repo.discover(".")
        vid = repo.list_manifests()[-1]
        man = repo.load_manifest(vid)

        before = len(os.listdir(os.path.join(repo.store, "datasets")))
        rc = main(["pin", "--version", vid[:12]])
        after = len(os.listdir(os.path.join(repo.store, "datasets")))
        self.assertEqual(rc, 0)
        self.assertEqual(before, after)  # no new manifest created

        lines, non_comment = self.read_pinfile()
        self.assertEqual(non_comment[0], man.root)

    # -- pin --file ----------------------------------------------------------

    def test_pin_file_custom_path(self):
        custom = "ci/dataset.pin"
        os.makedirs("ci")
        rc = main(["pin", "data", "--file", custom])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(custom))
        self.assertFalse(os.path.exists("framepin.pin"))

    # -- mutual exclusivity ---------------------------------------------------

    def test_pin_neither_dataset_nor_version_nor_from_list_exits_2(self):
        rc = main(["pin"])
        self.assertEqual(rc, 2)

    def test_pin_both_dataset_and_version_exits_2(self):
        main(["snapshot", "data"])
        repo = Repo.discover(".")
        vid = repo.list_manifests()[-1]
        rc = main(["pin", "data", "--version", vid[:12]])
        self.assertEqual(rc, 2)

    # -- pin --split -----------------------------------------------------

    def test_pin_split_records_splits_in_manifest(self):
        os.makedirs("data2/train")
        with open("data2/train/a.bin", "wb") as fh:
            fh.write(b"aaaa")
        rc = main(["pin", "data2", "--split", "train=train/*"])
        self.assertEqual(rc, 0)
        repo = Repo.discover(".")
        vid = repo.list_manifests()[-1]
        man = repo.load_manifest(vid)
        self.assertIn("train", man.splits)


class VerifyAgainstFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="framepin-verify-file-")
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

    def test_verify_against_file_passes_when_unchanged(self):
        main(["pin", "data"])
        rc = main(["verify", "data", "--against-file", "framepin.pin"])
        self.assertEqual(rc, 0)

    def test_verify_against_file_fails_on_drift(self):
        main(["pin", "data"])
        with open("data/clip_1.bin", "wb") as fh:
            fh.write(b"REENCODED" * 16)
        rc = main(["verify", "data", "--against-file", "framepin.pin"])
        self.assertEqual(rc, 3)

    def test_verify_against_file_combined_with_allow(self):
        main(["pin", "data"])
        with open("data/clip_1.bin", "wb") as fh:
            fh.write(b"REENCODED" * 16)
        rc = main([
            "verify", "data", "--against-file", "framepin.pin",
            "--allow", "clip_1.bin",
        ])
        self.assertEqual(rc, 0)

    def test_against_and_against_file_together_is_argparse_error(self):
        main(["pin", "data"])
        with self.assertRaises(SystemExit) as cm:
            main([
                "verify", "data", "--against", "deadbeef",
                "--against-file", "framepin.pin",
            ])
        self.assertEqual(cm.exception.code, 2)

    def test_pinfile_with_comments_and_blank_lines_parses(self):
        main(["pin", "data"])
        with open("framepin.pin", encoding="utf-8") as fh:
            root_line = [
                ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")
            ][0]
        with open("framepin.pin", "w", encoding="utf-8") as fh:
            fh.write("# a leading comment\n")
            fh.write("\n")
            fh.write("# another comment\n")
            fh.write(root_line + "\n")
            fh.write("\n")
        rc = main(["verify", "data", "--against-file", "framepin.pin"])
        self.assertEqual(rc, 0)

    def test_empty_pinfile_exits_2(self):
        with open("framepin.pin", "w", encoding="utf-8") as fh:
            fh.write("")
        rc = main(["verify", "data", "--against-file", "framepin.pin"])
        self.assertEqual(rc, 2)

    def test_missing_pinfile_exits_2(self):
        rc = main(["verify", "data", "--against-file", "does-not-exist.pin"])
        self.assertEqual(rc, 2)

    # -- baseline-update round trip -------------------------------------------

    def test_round_trip_intentional_change_then_repin(self):
        main(["pin", "data"])
        with open("data/clip_1.bin", "wb") as fh:
            fh.write(b"REENCODED" * 16)
        rc = main(["verify", "data", "--against-file", "framepin.pin"])
        self.assertEqual(rc, 3)

        # intentional change -> update the baseline
        rc = main(["pin", "data"])
        self.assertEqual(rc, 0)

        rc = main(["verify", "data", "--against-file", "framepin.pin"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
