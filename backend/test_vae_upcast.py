"""SDXL VAE encoding must upcast to fp32 in the fp16 training path."""
import re
import unittest
from pathlib import Path

TRAINER_SRC = Path(__file__).with_name("trainer.py").read_text()


def _load_fn():
    match = re.search(
        r"def vae_encode_needs_fp32_upcast\(.*?(?=\ndef )",
        TRAINER_SRC,
        re.S,
    )
    assert match, "vae_encode_needs_fp32_upcast not found in trainer.py"
    ns = {}
    exec(match.group(0), ns)
    return ns["vae_encode_needs_fp32_upcast"]


class TestVaeUpcast(unittest.TestCase):
    def test_fp16_force_upcast(self):
        fn = _load_fn()
        self.assertTrue(fn("float16", True, "sdxl"))
        self.assertTrue(fn("torch.float16", True, "sd15"))
        self.assertTrue(fn("fp16", True, "sdxl"))
        self.assertTrue(fn("half", True, "sdxl"))

    def test_sdxl_fp16_even_without_flag(self):
        fn = _load_fn()
        self.assertTrue(fn("float16", False, "sdxl"))
        self.assertTrue(fn("float16", False, "kolors"))

    def test_sd15_fp16_without_flag_stays_fp16(self):
        fn = _load_fn()
        self.assertFalse(fn("float16", False, "sd15"))

    def test_bf16_and_fp32_never_upcast(self):
        fn = _load_fn()
        self.assertFalse(fn("bfloat16", True, "sdxl"))
        self.assertFalse(fn("torch.bfloat16", True, "sdxl"))
        self.assertFalse(fn("float32", True, "sdxl"))
        self.assertFalse(fn("fp32", True, "sdxl"))

    def test_cache_loop_uses_upcast_helper_and_rejects_nan(self):
        self.assertIn("vae_encode_needs_fp32_upcast", TRAINER_SRC)
        self.assertIn("vae_encode_dtype", TRAINER_SRC)
        self.assertIn("torch.isfinite(latents)", TRAINER_SRC)
        self.assertRegex(
            TRAINER_SRC,
            r"pixel_values.*to\(device, dtype=vae_encode_dtype\)",
        )


if __name__ == "__main__":
    unittest.main()
