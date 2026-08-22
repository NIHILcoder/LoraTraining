import tempfile
import unittest
from pathlib import Path

from model_resolve import resolve_base_model

CATALOG = [
    {
        "id": "sd15",
        "name": "Stable Diffusion 1.5",
        "architecture": "sd15",
        "filename": "v1-5-pruned-emaonly.safetensors",
    },
    {
        "id": "sdxl10",
        "name": "Stable Diffusion XL 1.0",
        "architecture": "sdxl",
        "filename": "sd_xl_base_1.0.safetensors",
    },
]


class ResolveBaseModelTests(unittest.TestCase):
    def _touch(self, directory: Path, name: str) -> Path:
        path = directory / name
        path.write_bytes(b"weights")
        return path

    def test_specific_custom_id_uses_imported_file_not_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            catalog_file = self._touch(models_dir, "v1-5-pruned-emaonly.safetensors")
            custom_file = self._touch(models_dir, "my-finetune.safetensors")
            custom = [
                {
                    "id": "custom-abcd1234",
                    "name": "My Finetune",
                    "architecture": "sd15",
                    "filename": "my-finetune.safetensors",
                }
            ]

            path, arch, name = resolve_base_model(
                "custom-abcd1234", models_dir, CATALOG, custom
            )

            self.assertEqual(path, custom_file)
            self.assertEqual(arch, "sd15")
            self.assertEqual(name, "My Finetune")
            self.assertNotEqual(path, catalog_file)

    def test_unknown_custom_id_does_not_silently_use_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            self._touch(models_dir, "v1-5-pruned-emaonly.safetensors")

            path, arch, name = resolve_base_model(
                "custom-missing", models_dir, CATALOG, []
            )

            self.assertIsNone(path)
            self.assertIsNone(arch)
            self.assertIsNone(name)

    def test_catalog_id_still_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            expected = self._touch(models_dir, "sd_xl_base_1.0.safetensors")

            path, arch, name = resolve_base_model("sdxl10", models_dir, CATALOG, [])

            self.assertEqual(path, expected)
            self.assertEqual(arch, "sdxl")
            self.assertEqual(name, "Stable Diffusion XL 1.0")

    def test_auto_falls_back_to_custom_when_catalog_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            custom_file = self._touch(models_dir, "only-custom.safetensors")
            custom = [
                {
                    "id": "custom-only",
                    "name": "Only Custom",
                    "architecture": "sdxl",
                    "filename": "only-custom.safetensors",
                }
            ]

            path, arch, name = resolve_base_model(None, models_dir, CATALOG, custom)

            self.assertEqual(path, custom_file)
            self.assertEqual(arch, "sdxl")
            self.assertEqual(name, "Only Custom")

    def test_requested_id_missing_file_does_not_substitute(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            self._touch(models_dir, "v1-5-pruned-emaonly.safetensors")
            custom = [
                {
                    "id": "custom-abcd1234",
                    "name": "My Finetune",
                    "architecture": "sdxl",
                    "filename": "my-finetune.safetensors",
                }
            ]

            path, arch, name = resolve_base_model(
                "custom-abcd1234", models_dir, CATALOG, custom
            )

            self.assertIsNone(path)
            self.assertEqual(arch, "sdxl")
            self.assertEqual(name, "My Finetune")


if __name__ == "__main__":
    unittest.main()
