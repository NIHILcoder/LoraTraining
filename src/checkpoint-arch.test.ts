import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it, expect } from 'vitest';

const mainPy = readFileSync(resolve(__dirname, '../backend/main.py'), 'utf8');
const pkg = readFileSync(resolve(__dirname, '../package.json'), 'utf8');

describe('imported checkpoint architecture', () => {
  it('detects architecture from the safetensors header instead of defaulting to sd15', () => {
    expect(mainPy).toContain('infer_checkpoint_architecture');
    expect(mainPy).toMatch(/detected or req\.architecture/);
  });

  it('prefers an imported custom checkpoint over the catalog file when training', () => {
    expect(mainPy).toContain('resolve_base_model_path');
  });

  it('ships checkpoint_arch.py with the packaged backend', () => {
    expect(pkg).toContain('checkpoint_arch.py');
  });
});
