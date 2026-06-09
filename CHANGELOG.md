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

FSR3 no longer requires `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`. Devices
are classified into a `FormatProfile` (`Core` | `Tier2` | `Native`),
auto-detected at context creation, and the upscaler selects storage formats
and shader variants the device actually supports:

- **`Core`** — baseline WebGPU (`core-features-and-limits`). Works in the
  browser; the crate now builds for `wasm32-unknown-unknown` (CI-verified).
- **`Tier2`** — `texture-formats-tier2` (e.g. Metal). Not yet auto-detected
  (wgpu does not surface the feature); can be forced via
  `FsrContextInfo::format_profile`.
- **`Native`** — adapter-specific format features; unchanged behavior.

### Added

- `FormatProfile` with `FormatProfile::from_device` detection.
- `FsrFormats` — the formats the caller must use for each texture it
  provides. Obtainable before a context exists via `FormatProfile::formats()`,
  or from a live context via `FsrContext::formats()`.
- `FsrContext::format_profile()` — the profile the context was created with.

### Changed

- Breaking: `FsrContextInfo` gained a `format_profile: Option<FormatProfile>`
  field. `None` auto-detects; `Some` forces a profile (e.g. to exercise
  `Core` on a desktop adapter).

  ```diff
   FsrContextInfo {
       device,
       flags,
  +    format_profile: None,
   }
  ```

- Breaking: caller texture formats are profile-dependent and validated.
  `dilated_motion_vectors` is `Rg32Float` on `Core` (vs `Rg16Float`
  elsewhere). Allocate your textures from `FsrFormats` instead of
  hardcoding — it covers all six caller-provided textures (`color`, `depth`,
  `motion_vectors`, `output`, `dilated_depth`, `dilated_motion_vectors`).
  `FsrContext::dispatch` now validates every caller-provided texture against
  the active profile before recording any GPU work.

  ```diff
  +let formats = FormatProfile::from_device(&device).formats();
  +// or, with a live context: context.formats()
   device.create_texture(&wgpu::TextureDescriptor {
  -    format: wgpu::TextureFormat::Rg16Float,
  +    format: formats.dilated_motion_vectors,
       ..
   })
  ```

- Breaking: `FsrDispatchError` is now `#[non_exhaustive]` and gained a
  `TextureFormatMismatch { texture, expected, actual }` variant. Exhaustive
  matches need a wildcard arm.

  ```diff
   match err {
  +    FsrDispatchError::TextureFormatMismatch { texture, expected, actual } => ..,
  +    _ => ..,
       ..
   }
  ```

- Internal (no action needed):
  - Per-frame metadata moved from a 1×1 read-write storage texture to a
    storage buffer.
  - On `Core`, the luma / shading-change pyramids run as a write-only
    multi-pass chain instead of single-pass SPD; scene-average luma and
    auto-exposure match Native closely (covered by comparison tests).
  - Store-only storage images are now declared `writeonly` (a correctness
    improvement on all profiles).

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
