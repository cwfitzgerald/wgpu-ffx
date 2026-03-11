# wgpu-ffx

AMD FidelityFX FSR 3 Upscaler for [wgpu](https://wgpu.rs/).

A Rust implementation of AMD's FidelityFX Super Resolution 3 (FSR3) upscaler
for the wgpu graphics library. Renders at lower resolution and reconstructs
high-quality output at display resolution using temporal accumulation. Can also
be used at a 1:1 ratio for temporal anti-aliasing (TAA) without upscaling.

## Status

**Early / experimental.** Known limitations:

- RCAS sharpening pass is not yet wired up
- Shader permutation selection is not yet driven by flags
- `GenerateReactive` pass is unimplemented

## Requirements

- **wgpu 28**

## Usage

```rust
use wgpu_ffx::{FsrContext, FsrContextInfo, FsrContextFlags, FsrDispatchInfo};

// Create the upscaler context once.
let ctx = FsrContext::new(FsrContextInfo {
    device,
    queue,
    max_render_size: [1920, 1080],
    max_upscale_size: [3840, 2160],
    flags: FsrContextFlags::HIGH_DYNAMIC_RANGE
         | FsrContextFlags::DEPTH_INVERTED
         | FsrContextFlags::DEPTH_INFINITE,
});

// Per frame:
ctx.dispatch(&mut FsrDispatchInfo { /* ... */ })?;
```

## Texture format requirements

| Field | Type | Format | Size | Required usage flags |
|---|---|---|---|---|
| `color` | Texture | `Rgba16Float` | render_size | `TEXTURE_BINDING` |
| `depth` | Texture | `Depth32Float` | render_size | `TEXTURE_BINDING` |
| `motion_vectors` | Texture | `Rg16Float` | render_size (or upscale_size with `DISPLAY_RESOLUTION_MOTION_VECTORS`) | `TEXTURE_BINDING` |
| `output` | Texture | `Rgba16Float` | upscale_size | `STORAGE_BINDING + TEXTURE_BINDING` |
| `dilated_depth` | Texture | `R32Float` | render_size | `STORAGE_BINDING + TEXTURE_BINDING` |
| `dilated_motion_vectors` | Texture | `Rg16Float` | render_size | `STORAGE_BINDING + TEXTURE_BINDING` |
| `reconstructed_previous_depth` | Buffer | — | render_width × render_height × 4 bytes | `STORAGE + COPY_DST` |
| `exposure` | Texture (optional) | `R32Float` | 1×1 | `TEXTURE_BINDING` |
| `reactive_mask` | Texture (optional) | `R8Unorm` | render_size | `TEXTURE_BINDING` |
| `transparency_and_composition` | Texture (optional) | `R8Unorm` | render_size | `TEXTURE_BINDING` |

## Context flags

`FsrContextFlags` configure the upscaler behavior:

| Flag | Meaning |
|---|---|
| `HIGH_DYNAMIC_RANGE` | Input color data uses a high-dynamic range |
| `DISPLAY_RESOLUTION_MOTION_VECTORS` | Motion vectors are at display resolution instead of render resolution |
| `MOTION_VECTORS_JITTER_CANCELLATION` | Motion vectors already have the jitter pattern applied; FSR will cancel it |
| `DEPTH_INVERTED` | Depth buffer is inverted (1 = near, 0 = far) |
| `DEPTH_INFINITE` | Depth buffer uses an infinite far plane |
| `AUTO_EXPOSURE` | FSR computes exposure automatically; do not provide an `exposure` texture |
| `DYNAMIC_RESOLUTION` | Application uses dynamic resolution scaling |

## Jitter

Use `get_jitter_phase_count` and `get_jitter_offset` to compute per-frame
sub-pixel jitter offsets. Apply the jitter to your projection matrix **and**
pass the same offsets to `FsrDispatchInfo::jitter_offset`.

```rust
use wgpu_ffx::{get_jitter_phase_count, get_jitter_offset};

let phase_count = get_jitter_phase_count(render_width, display_width);
let [jx, jy] = get_jitter_offset(frame_index, phase_count);
```

## Building shaders

Requires glslc from the VulkanSDK.

```sh
cargo xtask vendor           # fetch upstream FidelityFX shader sources
cargo xtask compile-shaders  # compile to SPIR-V (requires glslc)
```

## Minimum Supported Rust Version (MSRV)

The MSRV is **1.92**. MSRV bumps are considered breaking changes and will be
accompanied by a minor version bump.

## License

[MIT](LICENSE.MIT)
