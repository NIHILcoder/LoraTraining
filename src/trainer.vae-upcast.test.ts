import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it, expect } from 'vitest';

const trainer = readFileSync(resolve(__dirname, '../backend/trainer.py'), 'utf8');

describe('SDXL VAE latent cache precision', () => {
  it('upcasts VAE encode to fp32 when training dtype is float16', () => {
    expect(trainer).toContain('def vae_encode_needs_fp32_upcast');
    expect(trainer).toContain('vae_encode_dtype');
    expect(trainer).toMatch(/pixel_values.*to\(device, dtype=vae_encode_dtype\)/);
  });

  it('rejects non-finite latents instead of training on NaNs', () => {
    expect(trainer).toContain('torch.isfinite(latents)');
  });
});
