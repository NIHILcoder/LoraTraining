# Changelog

All notable changes to **LoRA Studio** will be documented in this file.

## [1.0.0-beta.2] - 2026-07-05

### Fixed (correctness / data-loss)
- **SD 2.1 training** gated off — it was silently training with the wrong text encoder and epsilon loss. The trainer now also respects v-prediction targets.
- **Auto-caption-all** no longer overwrites earlier captions (stale-closure bug).
- Advanced training settings (aspect-ratio bucketing, caption dropout, noise offset) now actually reach the trainer — they were dropped by the request model.
- LoRA weight is no longer applied twice during inference.
- Concurrent generations are serialized (inference pipeline cache race).
- Uploaded image dimensions are read for real instead of hardcoded 1024×1024.
- Prompt/exception text is escaped in the placeholder SVG; the captioner returns proper HTTP errors instead of sentinel strings.
- Generated-image URLs are relative and the auth gate accepts a query token, so gallery/generated images load under the session token.

### Added
- **Downloads**: resumable/segmented downloads (survive network drops), SHA256 verification hook, disk-space precheck.
- **Models**: import an existing local `.safetensors` via a file picker.
- **Dataset**: bulk caption tools (prepend/trigger word, append, find-replace), server-side batch captioning, atomic per-image delete, dataset validation hints.
- **Playground**: batch generation (N images), visible generation errors, served-URL output with embedded A1111/Civitai PNG metadata (no more multi-MB base64 payloads).
- **Config**: saved user presets (localStorage).
- Vitest test harness with reducer coverage.

### Changed
- Training: no-replacement per-bucket batch sampling, grad-accumulation-aware LR schedule, seeded RNGs.
- Python dependencies pinned to tested versions; dependency versions logged at startup; custom `.ckpt/.bin/.pt` (pickle) models are rejected for safety.
- Training loss/log history is now capped to bound memory on long runs.

---

## [1.0.0-beta.1] - 2026-04-27

### Added
- **Flux.1 (Dev) Training**: Support for the latest transformer-based architecture.
- **Image Gallery**: A dedicated section in the Gallery to view, manage, and delete generated images.
- **Metadata Persistence**: Generation parameters (prompt, seed, model) are now saved alongside images.
- **Hardware Profiles**: Real-time VRAM/RAM monitoring and architecture feasibility checking.
- **Dynamic Output Management**: Ability to change the save directory for models and open folders directly from the UI.
- **Playground Enhancements**: Seed reuse, LoRA weight slider, and navigation from images back to playground.

### Fixed
- **Meta Tensor Error**: Fixed a critical issue where components were stuck on the `meta` device during inference.
- **VRAM Leak**: Implemented manual garbage collection and cache clearing after model extraction to save ~2-4GB VRAM.
- **OS Error 3**: Resolved pathing issues on Windows by ensuring eager directory creation.
- **Naming Conflicts**: Models saved with default settings now get unique suffixes to prevent overwriting.

### Optimized
- **Inference Speed**: Improved pipeline loading strategy by materializing weights in RAM before moving to GPU.
- **UI Performance**: Implemented canvas-based thumbnails for dataset uploads to prevent UI freezing.

---

## [0.9.0-alpha] - 2026-04-15

### Added
- **Initial Dashboard**: Basic layout with Sidebar and Header.
- **Dataset Tab**: Local file uploading and preview.
- **Training Config**: Basic hyperparameters (LR, steps, rank).
- **Backend Bridge**: WebSocket integration for real-time logs and progress.
- **Models Hub**: Basic catalog for downloading base models.
