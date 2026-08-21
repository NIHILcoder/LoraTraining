import json
import os
import tempfile
import unittest
from pathlib import Path

from storage_util import (
    allocate_nonclobber_dest,
    atomic_write_json,
    should_unlink_model_file,
    unique_filename,
)


class UniqueFilenameTests(unittest.TestCase):
    def test_returns_desired_when_free(self):
        self.assertEqual(unique_filename("model.safetensors", []), "model.safetensors")

    def test_suffixes_until_free(self):
        taken = {"model.safetensors", "model-1.safetensors"}
        self.assertEqual(unique_filename("model.safetensors", taken), "model-2.safetensors")


class AllocateDestTests(unittest.TestCase):
    def test_uses_original_name_when_free(self):
        with tempfile.TemporaryDirectory() as td:
            models = Path(td)
            src = Path(td) / "outside" / "sd_xl_base_1.0.safetensors"
            src.parent.mkdir()
            src.write_bytes(b"abc")
            dest = allocate_nonclobber_dest(models, src)
            self.assertEqual(dest, models / "sd_xl_base_1.0.safetensors")

    def test_does_not_clobber_existing_catalog_file(self):
        with tempfile.TemporaryDirectory() as td:
            models = Path(td)
            catalog = models / "sd_xl_base_1.0.safetensors"
            catalog.write_bytes(b"catalog-weights")
            src = Path(td) / "downloads" / "sd_xl_base_1.0.safetensors"
            src.parent.mkdir()
            src.write_bytes(b"other-weights")
            dest = allocate_nonclobber_dest(models, src)
            self.assertEqual(dest, models / "sd_xl_base_1.0-1.safetensors")
            self.assertEqual(catalog.read_bytes(), b"catalog-weights")

    def test_same_file_already_in_models_dir_keeps_path(self):
        with tempfile.TemporaryDirectory() as td:
            models = Path(td)
            src = models / "v1-5-pruned-emaonly.safetensors"
            src.write_bytes(b"weights")
            dest = allocate_nonclobber_dest(models, src)
            self.assertEqual(dest.resolve(), src.resolve())


class UnlinkGuardTests(unittest.TestCase):
    def test_refuses_when_catalog_still_holds_filename(self):
        self.assertFalse(
            should_unlink_model_file("sd_xl_base_1.0.safetensors", ["sd_xl_base_1.0.safetensors"])
        )

    def test_allows_when_nothing_else_references_file(self):
        self.assertTrue(should_unlink_model_file("custom.safetensors", ["sd_xl_base_1.0.safetensors"]))


class AtomicWriteTests(unittest.TestCase):
    def test_replaces_file_and_leaves_no_tmp(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "datasets.json"
            atomic_write_json(path, {"a": 1})
            atomic_write_json(path, {"a": 2, "b": 3})
            self.assertEqual(json.loads(path.read_text()), {"a": 2, "b": 3})
            self.assertFalse(path.with_name("datasets.json.tmp").exists())
            self.assertEqual(len(os.listdir(td)), 1)


if __name__ == "__main__":
    unittest.main()
