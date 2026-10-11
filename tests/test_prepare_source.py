"""Exercise failure boundaries without CUDA, Torch or network access."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_source import DEFAULT_PROFILE, apply_patch, extract_archive, load_profile, prepare


class PreparationBoundaries(unittest.TestCase):
    def test_patch_replay_does_not_depend_on_callers_git_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patch = root / 'new-file.patch'
            patch.write_bytes(b'diff --git a/new.py b/new.py\nnew file mode 100644\n'
                              b'--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+answer = 42\n')
            for setting in ('true', 'false'):
                source = root / setting
                source.mkdir()
                subprocess.run(['git', 'init', '--quiet'], cwd=source, check=True)
                subprocess.run(['git', 'config', 'core.autocrlf', setting], cwd=source, check=True)
                apply_patch(source, patch, root / f'{setting}.log')
                self.assertEqual((source / 'new.py').read_bytes(), b'answer = 42\r\n')

    def test_changed_input_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
            profile["input_sha256"][profile["source_archive"]] = "0" * 64
            candidate = root / "profile.json"
            candidate.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen build input changed"):
                prepare(candidate, root / "output")
            self.assertFalse((root / "output").exists())

    def test_unfrozen_declared_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
            del profile["input_sha256"][profile["upstream_lock"]]
            candidate = Path(directory) / "profile.json"
            candidate.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "freeze every declared"):
                load_profile(candidate)

    def test_archive_traversal_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escaped.txt", "must not escape")
            with self.assertRaisesRegex(ValueError, "escapes output"):
                extract_archive(archive, root / "destination")
            self.assertFalse((root / "escaped.txt").exists())
            self.assertFalse((root / "destination").exists())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "keep.txt"
            marker.write_bytes(b"existing output")
            with self.assertRaises(FileExistsError):
                prepare(DEFAULT_PROFILE, root)
            self.assertEqual(marker.read_bytes(), b"existing output")

    def test_failed_patch_stops_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            target = source / "file.txt"
            target.write_bytes(b"original\n")
            patch = root / "bad.patch"
            patch.write_bytes(b"diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-wrong\n+replacement\n")
            with self.assertRaises(subprocess.CalledProcessError):
                apply_patch(source, patch, root / "failure.log")
            self.assertEqual(target.read_bytes(), b"original\n")
            self.assertIn("patch failed", (root / "failure.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
