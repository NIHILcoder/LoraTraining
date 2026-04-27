# Changelog

All notable changes to **LoRA Studio** will be documented in this file.

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
