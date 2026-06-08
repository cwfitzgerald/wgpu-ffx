# Texture format handling

FSR3 runs entirely in compute and reads/writes a number of internal and
caller-provided textures through `STORAGE_BINDING`. WebGPU restricts which
texture formats may be used as storage textures, and the restriction depends on
the access mode (write-only, read-only, read-write) and on optional device
features. This document describes how `wgpu-ffx` selects formats so that the
upscaler runs across the full range of supported devices, from a baseline
WebGPU device up to a fully featured native desktop adapter.

## Why this is needed

The format restrictions apply specifically to the `STORAGE_BINDING` usage. A
texture that is only ever **sampled** (`TEXTURE_BINDING`) is unaffected: a
device can sample `r8unorm`, `r16float`, `rg16float`, etc. without any optional
feature. The limits only matter when a format is bound as a storage texture in
a shader (`image2D` in GLSL / `texture_storage_2d` in WGSL).

The relevant facts about baseline WebGPU storage textures are:

- The only formats usable with **read-write** storage access are `r32uint`,
  `r32sint`, and `r32float`. No multi-channel or sub-32-bit format supports
  read-write access without an optional feature, and `rg16float` read-write is
  not supported at *any* feature level.
- **Write-only** and **read-only** storage access is available for the 32-bit
  RGBA formats (`rgba16float`, `rgba8unorm`, `rgba32float`, the `rgba*uint/sint`
  family), the 32-bit R/RG formats (`r32*`, `rg32*`), but **not** for any
  1- or 2-channel format narrower than 32 bits (`r8`, `r16f`, `rg8`, `rg16f`,
  …). Those require `texture-formats-tier1`.
- Small-format storage access (`r8unorm`, `r16float`, … with write/read, and
  read-write for the wider formats) becomes available with
  `texture-formats-tier1` / `texture-formats-tier2`.

Baseline WebGPU here means a device with `core-features-and-limits`. WebGPU
*compatibility mode* (which lacks `core-features-and-limits`, and with it the
`rg32float`/`rgba16float` storage guarantees) is **not** supported.

## Format profiles

A device is classified into one of three `FormatProfile` levels, from most
restrictive to least:

| Profile  | Capability source | Read-write storage | Small-format storage |
|----------|-------------------|--------------------|----------------------|
| `Core`   | `core-features-and-limits` only | `r32uint/sint/float` only | none (must widen or pack) |
| `Tier2`  | `texture-formats-tier2` | wide formats gain read-write (e.g. `rgba16float`); `rg16float` read-write still impossible | `r8`/`r16f`/`rg16f` available |
| `Native` | `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` | adapter-reported, effectively unrestricted | adapter-reported |

`FormatProfile::from_device` inspects a `wgpu::Device` and returns the richest
profile the device supports. The profile can also be set explicitly when
constructing a context (see [Public API](#public-api)); forcing `Core` on a
capable device is the supported way to exercise the baseline path.

`Tier2` corresponds to running on an adapter (notably Metal) through the native
tier feature. It differs from `Native` in exactly one place — the SPD mip chain
(see [Special cases](#special-cases)) — because `rg16float` read-write is
unavailable even at tier 2.

## Internal storage textures

Each internal storage texture is classified by its true access pattern
(write-only vs. read-modify-write — determined by whether the shader ever
`imageLoad`s from the storage binding) and by whether it is linearly sampled
anywhere. Those two properties decide its fallback on `Core`:

- **Linearly sampled** ⇒ the fallback must preserve hardware filtering, so the
  format is **widened to an RGBA format** (`rgba8unorm` for `[0,1]` data,
  `rgba16float` for HDR/float data). RGBA formats support filtered sampling and
  write-only/read-only storage on `Core`.
- **Point-sampled only** ⇒ no filtering is required, so the format is **packed
  into a 32-bit format** (`r32float`, `rg32float`, `r32uint`), which is cheaper
  than widening to RGBA and is the only option for read-write data.

| Texture | Access | Sampling | `Native` | `Tier2` | `Core` |
|---|---|---|---|---|---|
| internal upscaled / luma history / output color | write-only | linear | `Rgba16Float` | `Rgba16Float` | `Rgba16Float` |
| dilated reactive masks | write-only | linear | `Rgba8Unorm` | `Rgba8Unorm` | `Rgba8Unorm` |
| dilated depth | write-only | point | `R32Float` | `R32Float` | `R32Float` |
| accumulation | write-only | linear | `R8Unorm` | `R8Unorm` | `Rgba8Unorm` |
| shading change | write-only | linear | `R8Unorm` | `R8Unorm` | `Rgba8Unorm` |
| luma / intermediate fp16 / farthest depth mip1 | write-only | linear | `R16Float` | `R16Float` | `Rgba16Float` |
| dilated motion vectors | write-only | point | `Rg16Float` | `Rg16Float` | `Rg32Float` |
| new locks | read-write | point | `R8Unorm` | `R8Unorm` | `R32Float` |
| frame info (1×1) | read-write | point | storage buffer | storage buffer | storage buffer |
| SPD mips | read-write | linear | `Rg16Float` | `Rgba16Float` | `Rgba16Float` (write-only chain) |

A texture that the shader only writes is bound with `WriteOnly` storage access
(not `ReadWrite`), which is what makes the wide formats valid on `Core`. The
sampled-texture (`Srv`) binding's `sample_type` must match the chosen format:
the point-sampled 32-bit fallbacks (`new locks` → `R32Float`, `dilated motion
vectors` → `Rg32Float`) are bound as unfilterable-float so they do not require
the `float32-filterable` feature.

## Special cases

**Frame info** is a 1×1 read-modify-write texel holding per-frame metadata. It
is stored as a small `STORAGE` **buffer** rather than a texture at every
profile. Buffers have unrestricted read-write access regardless of feature
level, so this sidesteps the format restriction entirely and keeps the binding
uniform across profiles.

**SPD mips** is the hardest case: the single-pass downsampler reads back a
previously written mip through a globally-coherent read-write storage binding,
and the mips are linearly sampled. `rg16float` read-write is impossible at every
profile, so the format is widened to `Rgba16Float`. On `Tier2` and `Native`,
`Rgba16Float` (resp. `Rg16Float`) read-write is available and the fast
single-pass algorithm is retained. On `Core`, read-write multi-channel storage
does not exist, so the pyramid is generated by a chain of **write-only**
dispatches — each level reads the previous level as a sampled texture and writes
the next level as a write-only `Rgba16Float` storage texture.

## Caller-provided textures

Format restrictions only affect the caller-provided textures that are bound as
storage. The textures the upscaler only samples (`color`, `depth`,
`motion_vectors`, the reactive/transparency masks, `exposure`) may keep their
natural formats on every profile.

| Caller texture | Storage use | `Native` / `Tier2` | `Core` |
|---|---|---|---|
| `output` | write-only | `Rgba16Float` | `Rgba16Float` |
| `dilated_depth` | write-only | `R32Float` | `R32Float` |
| `dilated_motion_vectors` | write-only, point-sampled | `Rg16Float` | `Rg32Float` |
| `reconstructed_previous_depth` | buffer | — | — |

`dilated_motion_vectors` is the only caller-provided texture whose required
format changes with the profile. Callers that allocate these textures should
query the expected formats from the context rather than hard-coding them; the
dispatch path validates each provided texture against the active profile and
reports a clear error naming the expected format.

## Shader variants

A storage texture's format is baked into its compiled SPIR-V via the GLSL
`layout(..., <format>)` qualifier and the surrounding load/store/sample helper
code, so a format change requires a recompiled shader variant. Variant
selection is driven by a single permutation define,
`FFX_WGPU_FORMAT_PROFILE` (`0` = core, `1` = tier2, `2` = native), declared in
`shaders/.../perm.toml` and consumed in the FSR3 GLSL callbacks header. The
define gates:

- the `layout(..., <format>)` qualifier and the `writeonly` / read-write
  qualifier on each storage image,
- channel swizzles on the load/store helpers where a single-channel value is
  stored in a widened RGBA texture,
- the SPD reduction path (single-pass vs. write-only mip chain).

The shader build (`xtask compile-shaders`) takes the cartesian product of all
permutation defines, so adding the profile define generates the variant set
automatically. Most passes compile to identical SPIR-V across `Tier2` and
`Native`; content-hash deduplication collapses those, so the practical cost is a
full `Core` variant set plus an SPD-specific `Tier2` variant.

## Public API

```rust
pub enum FormatProfile { Core, Tier2, Native }

impl FormatProfile {
    /// The richest profile a device supports.
    pub fn from_device(device: &wgpu::Device) -> FormatProfile;
    /// The formats a caller must use for the textures it provides.
    pub fn formats(self) -> FsrFormats;
}

/// Formats for the caller-provided textures under a given profile.
#[derive(Clone, Copy)]
pub struct FsrFormats {
    pub color: wgpu::TextureFormat,
    pub depth: wgpu::TextureFormat,
    pub motion_vectors: wgpu::TextureFormat,
    pub output: wgpu::TextureFormat,
    pub dilated_depth: wgpu::TextureFormat,
    pub dilated_motion_vectors: wgpu::TextureFormat,
}

pub struct FsrContextInfo {
    pub device: wgpu::Device,
    pub flags: FsrContextFlags,
    /// `None` auto-detects via `FormatProfile::from_device`; `Some` forces a
    /// profile (the device must support it).
    pub format_profile: Option<FormatProfile>,
}

impl FsrContext {
    pub fn format_profile(&self) -> FormatProfile;
    pub fn formats(&self) -> FsrFormats;
}
```

`FsrFormats` includes the fixed-format textures as well as the variable one so
that callers have a single authoritative source for every texture they
allocate. It is obtainable from a `FormatProfile` before a context exists, which
is required when allocating the textures that will be passed into the context.
