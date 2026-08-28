import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';
import { pickBaseIdForLora } from './loraArchMatch';

const bases = [
  { id: 'sd15', architecture: 'sd15' },
  { id: 'sdxl10', architecture: 'sdxl' },
];

describe('pickBaseIdForLora', () => {
  it('prefers a downloaded checkpoint with the same architecture as the LoRA', () => {
    expect(pickBaseIdForLora(bases, 'sdxl')).toBe('sdxl10');
    expect(pickBaseIdForLora(bases, 'SD15')).toBe('sd15');
  });

  it('falls back to the first downloaded model when the LoRA arch is unknown', () => {
    expect(pickBaseIdForLora(bases, '')).toBe('sd15');
    expect(pickBaseIdForLora(bases, undefined)).toBe('sd15');
  });

  it('returns undefined when nothing is downloaded', () => {
    expect(pickBaseIdForLora([], 'sdxl')).toBeUndefined();
  });
});

describe('playground / generate wiring', () => {
  it('Playground uses pickBaseIdForLora when Gallery Test supplies a LoRA', () => {
    const src = readFileSync(join(__dirname, 'pages/PlaygroundPage.tsx'), 'utf8');
    expect(src).toContain('pickBaseIdForLora');
    expect(src).toContain('loraModel.architecture');
  });

  it('generate_image rejects LoRA/base architecture mismatch', () => {
    const src = readFileSync(join(__dirname, '..', 'backend/main.py'), 'utf8');
    expect(src).toContain('architecture_mismatch');
    expect(src).toContain('architecture_from_result_file');
  });
});
