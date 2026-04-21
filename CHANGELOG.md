# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to cargo's version of [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Per Keep a Changelog there are 6 main categories of changes:
- Added
- Changed
- Deprecated
- Removed
- Fixed
- Security

#### Table of Contents

- [Unreleased](#unreleased)
- [v0.1.0](#v010)

## Unreleased

## v0.1.0

Released 2026-04-21

Initial release. `wgpu-ffx` is a Rust port of AMD's FidelityFX Super
Resolution 3 (FSR3) temporal upscaler targeting the [wgpu](https://wgpu.rs/)
graphics library. It can be used to render at a lower resolution and
reconstruct display-resolution output, or at a 1:1 ratio as a
temporal anti-aliasing (TAA) solution.

### Added

- `FsrContext` — compiles FSR3 compute pipelines against a `wgpu::Device`
  and a set of `FsrContextFlags`.
- `FsrView` — owns per-resolution GPU textures and temporal-accumulation
  state; created from a context via `FsrContext::create_view` and resizable
  via `FsrView::resize` without rebuilding the parent context's pipelines.
- `FsrContext::dispatch` — records the FSR3 passes into a caller-owned
  `wgpu::CommandEncoder` and validates `FsrDispatchInfo` parameters up front,
  returning `FsrDispatchError` on invalid inputs.
- `FsrContextFlags` for high dynamic range input, display-resolution motion
  vectors, jitter-cancelled motion vectors, inverted depth, infinite depth,
  auto exposure, and dynamic resolution.
- Optional Robust Contrast Adaptive Sharpening (RCAS) pass, selected per
  dispatch via `FsrDispatchInfo::enable_sharpening` and `sharpness`.
- Halton-sequence jitter helpers `get_jitter_phase_count` and
  `get_jitter_offset`.
- `FsrView::estimated_memory_usage` for reporting a lower-bound estimate of
  internal GPU memory use.
- Flag-driven shader permutation selection for HDR input, inverted depth,
  jittered motion vectors, and low-resolution motion vectors.
- `xtask vendor` / `xtask compile-shaders` commands for fetching and
  compiling the upstream FidelityFX shader sources to SPIR-V.

### Known Limitations

- The `GenerateReactive` pass is not wired up.
- Lanczos-LUT, half-precision, and wave64 shader permutations are always
  compiled with their reference/full-precision variants — device-capability
  driven selection is not yet implemented.

## Diffs

- [Unreleased](https://github.com/cwfitzgerald/wgpu-ffx/compare/v0.1.0...HEAD)
- [v0.1.0](https://github.com/cwfitzgerald/wgpu-ffx/releases/tag/v0.1.0)
