"""Resume / finalize behavior for model downloads (no GPU required)."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from download_utils import (
    MIN_MODEL_BYTES,
    part_is_promotable,
    promote_part_file,
    sha256_file,
    should_reset_part_after_http_error,
)


class PartIsPromotableTests(unittest.TestCase):
    def test_rejects_tiny_files_even_with_matching_size(self):
        self.assertFalse(
            part_is_promotable(part_size=1024, expected_sha=None, expected_size=1024, actual_sha=None)
        )

    def test_sha_match_promotes(self):
        self.assertTrue(
            part_is_promotable(
                part_size=MIN_MODEL_BYTES,
                expected_sha="ABC",
                expected_size=0,
                actual_sha="abc",
            )
        )

    def test_sha_mismatch_does_not_promote(self):
        self.assertFalse(
            part_is_promotable(
                part_size=MIN_MODEL_BYTES,
                expected_sha="aaa",
                expected_size=MIN_MODEL_BYTES,
                actual_sha="bbb",
            )
        )

    def test_exact_catalog_size_promotes_without_sha(self):
        self.assertTrue(
            part_is_promotable(
                part_size=MIN_MODEL_BYTES,
                expected_sha=None,
                expected_size=MIN_MODEL_BYTES,
                actual_sha=None,
            )
        )

    def test_size_mismatch_without_sha_does_not_promote(self):
        self.assertFalse(
            part_is_promotable(
                part_size=MIN_MODEL_BYTES + 1,
                expected_sha=None,
                expected_size=MIN_MODEL_BYTES,
                actual_sha=None,
            )
        )

    def test_custom_url_without_sha_or_size_is_never_promoted(self):
        # A 416 on a custom URL must not treat an arbitrary .part as complete.
        self.assertFalse(
            part_is_promotable(
                part_size=MIN_MODEL_BYTES * 2,
                expected_sha=None,
                expected_size=0,
                actual_sha=None,
            )
        )


class Http416ResetTests(unittest.TestCase):
    def test_416_with_unverified_part_must_reset(self):
        self.assertTrue(should_reset_part_after_http_error(416, can_promote=False))

    def test_416_with_verified_part_keeps_file(self):
        self.assertFalse(should_reset_part_after_http_error(416, can_promote=True))

    def test_other_errors_do_not_imply_reset(self):
        self.assertFalse(should_reset_part_after_http_error(403, can_promote=False))
        self.assertFalse(should_reset_part_after_http_error(200, can_promote=False))


class PromotePartFileTests(unittest.TestCase):
    def test_renames_part_to_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            part = tmp_path / "model.safetensors.part"
            final = tmp_path / "model.safetensors"
            payload = b"complete-weights"
            part.write_bytes(payload)
            promote_part_file(part, final, retries=1, delay_s=0)
            self.assertFalse(part.exists())
            self.assertEqual(final.read_bytes(), payload)

    def test_replaces_existing_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            part = tmp_path / "model.safetensors.part"
            final = tmp_path / "model.safetensors"
            part.write_bytes(b"new")
            final.write_bytes(b"old")
            promote_part_file(part, final, retries=1, delay_s=0)
            self.assertEqual(final.read_bytes(), b"new")

    def test_retries_then_succeeds_on_windows_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            part = tmp_path / "model.safetensors.part"
            final = tmp_path / "model.safetensors"
            part.write_bytes(b"ok")
            calls = {"n": 0}

            real_move = promote_part_file.__globals__["shutil"].move

            def flaky_move(src, dst):
                calls["n"] += 1
                if calls["n"] < 3:
                    raise OSError(13, "Permission denied")
                return real_move(src, dst)

            with mock.patch("download_utils.shutil.move", side_effect=flaky_move):
                promote_part_file(part, final, retries=5, delay_s=0)

            self.assertEqual(calls["n"], 3)
            self.assertEqual(final.read_bytes(), b"ok")

    def test_sha256_file_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "blob.bin"
            data = b"x" * 10000
            p.write_bytes(data)
            self.assertEqual(sha256_file(p), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    unittest.main()
