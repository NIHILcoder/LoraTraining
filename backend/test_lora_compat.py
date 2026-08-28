import json
import tempfile
import unittest
from pathlib import Path

from lora_compat import architecture_from_result_file, architecture_mismatch


class ArchitectureMismatchTests(unittest.TestCase):
    def test_matching_arches_ok(self):
        self.assertIsNone(architecture_mismatch("sdxl", "sdxl"))
        self.assertIsNone(architecture_mismatch("SDXL", "sdxl"))

    def test_mismatch_message(self):
        msg = architecture_mismatch("sdxl", "sd15")
        self.assertIsNotNone(msg)
        self.assertIn("SDXL", msg)
        self.assertIn("SD15", msg)

    def test_unknown_arch_does_not_block(self):
        self.assertIsNone(architecture_mismatch(None, "sd15"))
        self.assertIsNone(architecture_mismatch("sdxl", None))
        self.assertIsNone(architecture_mismatch("", "sd15"))

    def test_reads_training_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_result.json"
            path.write_text(json.dumps({"architecture": "sdxl", "final_loss": 0.1}), encoding="utf-8")
            self.assertEqual(architecture_from_result_file(path), "sdxl")

    def test_missing_or_corrupt_result_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "training_result.json"
            self.assertIsNone(architecture_from_result_file(missing))
            missing.write_text("{not json", encoding="utf-8")
            self.assertIsNone(architecture_from_result_file(missing))


if __name__ == "__main__":
    unittest.main()
