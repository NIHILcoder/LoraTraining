import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it, expect } from 'vitest';

const trainer = readFileSync(resolve(__dirname, '../backend/trainer.py'), 'utf8');

describe('SDXL training text encoders', () => {
  it('does not load the raw OpenCLIP bigG repo (wrong tokenizer pad token)', () => {
    expect(trainer).not.toContain('laion/CLIP-ViT-bigG');
    expect(trainer).not.toMatch(/from_pretrained\(\s*"laion\//);
  });

  it('falls back to the official SDXL diffusers text_encoder_2', () => {
    expect(trainer).toContain('stabilityai/stable-diffusion-xl-base-1.0');
    expect(trainer).toContain('subfolder="text_encoder_2"');
    expect(trainer).toContain('subfolder="tokenizer_2"');
  });

  it('uses the penultimate CLIP layer for SDXL (matches diffusers encode_prompt)', () => {
    expect(trainer).toContain('def prompt_hidden_state_index');
    expect(trainer).toMatch(/if architecture == ["']sdxl["']:\s*\n\s*return -2/);
  });
});
