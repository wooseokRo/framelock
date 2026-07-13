import unittest

from framelock.manifest import Manifest, FileEntry
from framelock.diff import diff_manifests


def _man(files):
    entries = [FileEntry(path=p, hash=h, size=len(h)) for p, h in files]
    return Manifest(
        root="r" + "".join(h for _, h in files),
        algo="sha256",
        created_at="",
        dataset_path="/tmp/x",
        file_count=len(entries),
        total_bytes=sum(e.size for e in entries),
        files=entries,
    )


class TestDiff(unittest.TestCase):
    def test_added(self):
        old = _man([("a", "h1")])
        new = _man([("a", "h1"), ("b", "h2")])
        d = diff_manifests(old, new)
        self.assertEqual(d.added, ["b"])
        self.assertEqual(d.removed, [])
        self.assertEqual(d.unchanged, 1)

    def test_removed(self):
        old = _man([("a", "h1"), ("b", "h2")])
        new = _man([("a", "h1")])
        d = diff_manifests(old, new)
        self.assertEqual(d.removed, ["b"])
        self.assertEqual(d.added, [])

    def test_modified(self):
        old = _man([("a", "h1")])
        new = _man([("a", "h2")])
        d = diff_manifests(old, new)
        self.assertEqual(len(d.modified), 1)
        self.assertEqual(d.modified[0].path, "a")
        self.assertEqual(d.modified[0].old_hash, "h1")
        self.assertEqual(d.modified[0].new_hash, "h2")
        self.assertEqual(d.added, [])
        self.assertEqual(d.removed, [])

    def test_moved_not_reported_as_add_remove(self):
        # same content, different path -> move, not remove+add
        old = _man([("old/name.mp4", "hX")])
        new = _man([("new/name.mp4", "hX")])
        d = diff_manifests(old, new)
        self.assertEqual(d.added, [])
        self.assertEqual(d.removed, [])
        self.assertEqual(len(d.moved), 1)
        self.assertEqual(d.moved[0].from_path, "old/name.mp4")
        self.assertEqual(d.moved[0].to_path, "new/name.mp4")

    def test_move_plus_extra_copy_falls_through(self):
        # one source hash, two destinations: one pairs as move, extra is add
        old = _man([("a", "hX")])
        new = _man([("b", "hX"), ("c", "hX")])
        d = diff_manifests(old, new)
        self.assertEqual(len(d.moved), 1)
        self.assertEqual(len(d.added), 1)
        self.assertEqual(d.removed, [])

    def test_no_change(self):
        old = _man([("a", "h1"), ("b", "h2")])
        new = _man([("a", "h1"), ("b", "h2")])
        d = diff_manifests(old, new)
        self.assertFalse(d.changed)
        self.assertEqual(d.unchanged, 2)

    def test_summary_counts(self):
        old = _man([("a", "h1"), ("gone", "h9"), ("mv", "hM")])
        new = _man([("a", "h2"), ("new", "h3"), ("moved/mv", "hM")])
        d = diff_manifests(old, new)
        s = d.summary()
        self.assertEqual(s["modified"], 1)
        self.assertEqual(s["added"], 1)
        self.assertEqual(s["removed"], 1)
        self.assertEqual(s["moved"], 1)


if __name__ == "__main__":
    unittest.main()
