/** Pick a downloaded base checkpoint that matches a LoRA's training architecture. */

export function pickBaseIdForLora(
  bases: { id: string; architecture?: string }[],
  loraArchitecture?: string | null,
): string | undefined {
  if (!bases.length) return undefined;
  const arch = (loraArchitecture || '').trim().toLowerCase();
  if (arch) {
    const match = bases.find((b) => (b.architecture || '').toLowerCase() === arch);
    if (match) return match.id;
  }
  return bases[0].id;
}
