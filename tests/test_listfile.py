import os
import shutil
import tempfile
import unittest

from framepin.listfile import HashCache, parse_list_file, snapshot_from_lists


class ListfileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="framepin-list-")
        self.data = os.path.join(self.dir, "clips")
        os.makedirs(self.data)
        self.files = []
        for i in range(4):
            p = os.path.join(self.data, f"clip_{i}.bin")
            with open(p, "wb") as fh:
                fh.write(f"frames-{i}".encode() * 32)
            self.files.append(p)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write_list(self, name, paths, extra_lines=()):
        lp = os.path.join(self.dir, name)
        with open(lp, "w", encoding="utf-8") as fh:
            for line in extra_lines:
                fh.write(line + "\n")
            for p in paths:
                fh.write(p + "\n")
        return lp

    def test_parse_skips_comments_and_blanks(self):
        lp = self.write_list("a.txt", self.files[:2], extra_lines=["# header", "", "  "])
        self.assertEqual(parse_list_file(lp), self.files[:2])

    def test_deterministic_and_content_sensitive(self):
        lp = self.write_list("a.txt", self.files)
        m1 = snapshot_from_lists([lp])
        m2 = snapshot_from_lists([lp])
        self.assertEqual(m1.root, m2.root)  # same construct -> same id
        # touch content of a referenced clip -> id changes even though list text didn't
        with open(self.files[0], "wb") as fh:
            fh.write(b"REENCODED" * 32)
        m3 = snapshot_from_lists([lp])
        self.assertNotEqual(m1.root, m3.root)

    def test_list_text_change_changes_id(self):
        lp = self.write_list("a.txt", self.files)
        m1 = snapshot_from_lists([lp])
        # reorder lines: same referenced set, different list definition
        self.write_list("a.txt", list(reversed(self.files)))
        m2 = snapshot_from_lists([lp])
        self.assertNotEqual(m1.root, m2.root)

    def test_concat_lists_dedupe(self):
        a = self.write_list("a.txt", self.files[:3])
        b = self.write_list("b.txt", self.files[1:])  # overlaps 1,2
        m = snapshot_from_lists([a, b])
        # 2 list files + 4 unique refs
        self.assertEqual(m.file_count, 6)

    def test_missing_path_recorded_not_fatal(self):
        ghost = os.path.join(self.data, "deleted.bin")
        lp = self.write_list("a.txt", self.files[:1] + [ghost])
        m = snapshot_from_lists([lp])
        marks = [f for f in m.files if f.hash == "missing:"]
        self.assertEqual(len(marks), 1)
        self.assertEqual(marks[0].path, ghost)

    def test_cache_skips_unchanged_and_detects_change(self):
        lp = self.write_list("a.txt", self.files)
        cpath = os.path.join(self.dir, "hashcache.json")
        m1 = snapshot_from_lists([lp], cache=HashCache(cpath))
        self.assertTrue(os.path.isfile(cpath))
        # cached second pass must yield the identical root
        m2 = snapshot_from_lists([lp], cache=HashCache(cpath))
        self.assertEqual(m1.root, m2.root)
        # change content (mtime moves) -> cache must NOT serve the stale hash
        with open(self.files[1], "wb") as fh:
            fh.write(b"CHANGED" * 32)
        m3 = snapshot_from_lists([lp], cache=HashCache(cpath))
        self.assertNotEqual(m1.root, m3.root)

    def test_fast_mode_fingerprints(self):
        lp = self.write_list("a.txt", self.files)
        m = snapshot_from_lists([lp], fast=True)
        ref_entries = [f for f in m.files if f.path != lp]
        self.assertTrue(all(f.hash.startswith("stat:") for f in ref_entries))
        # list files themselves stay content-hashed even in fast mode
        lst = [f for f in m.files if f.path == lp][0]
        self.assertFalse(lst.hash.startswith("stat:"))


if __name__ == "__main__":
    unittest.main()
