import os
import tempfile
import unittest

from framelock import manifest as m
from framelock import hashing


def _write(root, rel, data: bytes):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


class TestManifest(unittest.TestCase):
    def test_snapshot_records_files_without_copying(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "clips/a.mp4", b"aaaa")
            _write(d, "clips/b.mp4", b"bbbbbb")
            man = m.snapshot(d)
            self.assertEqual(man.file_count, 2)
            self.assertEqual(man.total_bytes, 4 + 6)
            paths = sorted(f.path for f in man.files)
            self.assertEqual(paths, ["clips/a.mp4", "clips/b.mp4"])
            # No data was copied anywhere under the store; manifest is metadata only.
            self.assertNotIn(".framelock", os.listdir(d))

    def test_root_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "x/1.txt", b"one")
            _write(d, "x/2.txt", b"two")
            r1 = m.snapshot(d, created_at="2020-01-01T00:00:00+00:00").root
            r2 = m.snapshot(d, created_at="2099-12-31T23:59:59+00:00").root
            # created_at must not affect the version id.
            self.assertEqual(r1, r2)

    def test_root_changes_on_content_change(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "a.txt", b"hello")
            r1 = m.snapshot(d).root
            _write(d, "a.txt", b"hello!")
            r2 = m.snapshot(d).root
            self.assertNotEqual(r1, r2)

    def test_root_changes_on_rename(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "a.txt", b"hello")
            r1 = m.snapshot(d).root
            os.rename(os.path.join(d, "a.txt"), os.path.join(d, "b.txt"))
            r2 = m.snapshot(d).root
            self.assertNotEqual(r1, r2)

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as d:
            man = m.snapshot(d)
            self.assertEqual(man.file_count, 0)
            self.assertEqual(man.total_bytes, 0)
            # Two empty snapshots share a stable root.
            self.assertEqual(man.root, m.snapshot(d).root)

    def test_ignores_framelock_and_git(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "real.txt", b"data")
            _write(d, ".framelock/datasets/x.json", b"{}")
            _write(d, ".git/config", b"[core]")
            man = m.snapshot(d)
            self.assertEqual([f.path for f in man.files], ["real.txt"])

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "a.txt", b"x")
            man = m.snapshot(d)
            back = m.loads(m.dumps(man))
            self.assertEqual(back.root, man.root)
            self.assertEqual(back.path_map(), man.path_map())

    def test_snapshot_on_missing_dir_raises(self):
        with self.assertRaises(NotADirectoryError):
            m.snapshot("/nonexistent/path/xyz")

    def test_merkle_root_order_independent(self):
        a = hashing.merkle_root([("b", "h2"), ("a", "h1")])
        b = hashing.merkle_root([("a", "h1"), ("b", "h2")])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
