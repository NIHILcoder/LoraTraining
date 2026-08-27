"""Imported checkpoints must be labeled and selected by their real architecture."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

from checkpoint_arch import (
    infer_checkpoint_architecture,
    read_safetensors_tensor_keys,
    resolve_base_model_path,
)


def write_fake_safetensors(path: Path, tensor_names: list[str]) -> None:
    header = {
        name: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
        for name in tensor_names
    }
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00\x00")


class TestInferCheckpointArchitecture(unittest.TestCase):
    def test_reads_header_keys_without_payload(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.safetensors"
            write_fake_safetensors(p, ["cond_stage_model.transformer.text_model.a"])
            keys = read_safetensors_tensor_keys(p)
            self.assertIn("cond_stage_model.transformer.text_model.a", keys)

    def test_sdxl_conditioner_is_not_sd15(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "pony.safetensors"
            write_fake_safetensors(
                p,
                [
                    "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
                    "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
                    "model.diffusion_model.input_blocks.0.0.weight",
                ],
            )
            self.assertEqual(infer_checkpoint_architecture(p), "sdxl")

    def test_sd15_cond_stage_transformer(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sd15.safetensors"
            write_fake_safetensors(
                p,
                [
                    "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
                    "model.diffusion_model.input_blocks.0.0.weight",
                ],
            )
            self.assertEqual(infer_checkpoint_architecture(p), "sd15")

    def test_sd21_openclip_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sd21.safetensors"
            write_fake_safetensors(
                p,
                [
                    "cond_stage_model.model.token_embedding.weight",
                    "model.diffusion_model.input_blocks.0.0.weight",
                ],
            )
            self.assertEqual(infer_checkpoint_architecture(p), "sd21")

    def test_unknown_header_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mystery.safetensors"
            write_fake_safetensors(p, ["some.unrelated.tensor"])
            self.assertIsNone(infer_checkpoint_architecture(p))


class TestResolveBaseModelPath(unittest.TestCase):
    def test_custom_checkpoint_wins_over_catalog_same_arch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            catalog_file = root / "sd_xl_base_1.0.safetensors"
            custom_file = root / "pony.safetensors"
            catalog_file.write_bytes(b"x")
            custom_file.write_bytes(b"y")
            chosen = resolve_base_model_path(
                "sdxl",
                root,
                [{"architecture": "sdxl", "filename": catalog_file.name}],
                [{"architecture": "sdxl", "filename": custom_file.name}],
            )
            self.assertEqual(chosen, custom_file)

    def test_falls_back_to_catalog_when_no_custom_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            catalog_file = root / "v1-5-pruned-emaonly.safetensors"
            catalog_file.write_bytes(b"x")
            chosen = resolve_base_model_path(
                "sd15",
                root,
                [{"architecture": "sd15", "filename": catalog_file.name}],
                [{"architecture": "sdxl", "filename": "pony.safetensors"}],
            )
            self.assertEqual(chosen, catalog_file)

    def test_latest_custom_entry_wins(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "a.safetensors"
            second = root / "b.safetensors"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            chosen = resolve_base_model_path(
                "sdxl",
                root,
                [],
                [
                    {"architecture": "sdxl", "filename": first.name},
                    {"architecture": "sdxl", "filename": second.name},
                ],
            )
            self.assertEqual(chosen, second)


class TestMainWiring(unittest.TestCase):
    def test_import_and_training_use_header_detection_and_custom_priority(self):
        src = Path(__file__).with_name("main.py").read_text()
        self.assertIn("infer_checkpoint_architecture", src)
        self.assertIn("resolve_base_model_path", src)
        self.assertIn("detected or req.architecture", src)


if __name__ == "__main__":
    unittest.main()
