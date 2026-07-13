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


if __name__ == "__main__":
    unittest.main()
