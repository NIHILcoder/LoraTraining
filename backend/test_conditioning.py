"""SDXL training must condition on the same CLIP setup Playground inference uses."""
import re
import unittest
from pathlib import Path

TRAINER_SRC = Path(__file__).with_name("trainer.py").read_text()


def _load_prompt_hidden_state_index():
    match = re.search(
        r"def prompt_hidden_state_index\(architecture: str, clip_skip: int\) -> int:.*?(?=\ndef )",
        TRAINER_SRC,
        re.S,
    )
    assert match, "prompt_hidden_state_index not found in trainer.py"
    ns = {}
    exec(match.group(0), ns)
    return ns["prompt_hidden_state_index"]


class TestSdxlConditioning(unittest.TestCase):
    def test_sdxl_always_penultimate_regardless_of_clip_skip(self):
        fn = _load_prompt_hidden_state_index()
        self.assertEqual(fn("sdxl", 1), -2)
        self.assertEqual(fn("sdxl", 2), -2)
        self.assertEqual(fn("sdxl", 4), -2)

    def test_sd15_uses_a1111_clip_skip(self):
        fn = _load_prompt_hidden_state_index()
        self.assertEqual(fn("sd15", 1), -1)
        self.assertEqual(fn("sd15", 2), -2)
        self.assertEqual(fn("sd15", 0), -1)
        self.assertEqual(fn("sd15", "nope"), -1)

    def test_does_not_load_raw_openclip_bigg(self):
        self.assertNotIn("laion/CLIP-ViT-bigG", TRAINER_SRC)
        self.assertNotRegex(TRAINER_SRC, r'from_pretrained\(\s*"laion/')

    def test_fallback_uses_official_sdxl_diffusers_repo(self):
        self.assertIn("stabilityai/stable-diffusion-xl-base-1.0", TRAINER_SRC)
        self.assertIn('subfolder="text_encoder_2"', TRAINER_SRC)
        self.assertIn('subfolder="tokenizer_2"', TRAINER_SRC)
        self.assertIn("_load_sdxl_text_encoders", TRAINER_SRC)


if __name__ == "__main__":
    unittest.main()
