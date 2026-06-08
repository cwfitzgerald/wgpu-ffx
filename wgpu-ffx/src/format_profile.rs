//! Storage-texture capability profiles and the caller-facing formats they
//! imply.
//!
//! WebGPU restricts which texture formats may be bound as storage textures, and
//! the restriction depends on the access mode and on optional device features.
//! A device is classified into a [`FormatProfile`] so the upscaler can select
//! formats (and matching shader variants) that the device actually supports.
//!
//! See `docs/texture-formats.md` for the full rationale and the per-texture
//! format mapping.

/// The storage-texture capability level a device is driven at, from most
/// restrictive to least.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatProfile {
    /// Baseline WebGPU (`core-features-and-limits`). Read-write storage is
    /// limited to the `r32{uint,sint,float}` formats, and 1-/2-channel formats
    /// narrower than 32 bits are unavailable as storage textures.
    Core,
    /// `texture-formats-tier2` — e.g. Metal through the native tier feature.
    /// Wide formats gain read-write access; `rg16float` read-write remains
    /// unavailable.
    Tier2,
    /// Native desktop, relying on `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`.
    /// Storage support is reported per adapter and is effectively unrestricted.
    Native,
}

impl FormatProfile {
    /// The richest profile `device` supports.
    pub fn from_device(device: &wgpu::Device) -> FormatProfile {
        let features = device.features();

        if features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
            FormatProfile::Native
        } else {
            // `texture-formats-tier2` is not yet surfaced by wgpu. Once it is,
            // detect it here and return `FormatProfile::Tier2` before falling
            // back to `Core`.
            FormatProfile::Core
        }
    }

    /// The formats a caller must use for the textures it provides to a context
    /// running at this profile.
    pub fn formats(self) -> FsrFormats {
        FsrFormats {
            color: wgpu::TextureFormat::Rgba16Float,
            depth: wgpu::TextureFormat::Depth32Float,
            motion_vectors: wgpu::TextureFormat::Rg16Float,
            output: wgpu::TextureFormat::Rgba16Float,
            dilated_depth: wgpu::TextureFormat::R32Float,
            dilated_motion_vectors: match self {
                // `rg16float` is unavailable as a storage texture on `Core`, so
                // the dilated motion vectors widen to the 32-bit two-channel
                // format. They are only ever point-sampled, so this needs no
                // filtering feature.
                FormatProfile::Core => wgpu::TextureFormat::Rg32Float,
                FormatProfile::Tier2 | FormatProfile::Native => wgpu::TextureFormat::Rg16Float,
            },
        }
    }
}

/// The formats the caller must use for each texture it provides, for a given
/// [`FormatProfile`].
///
/// Obtainable from a [`FormatProfile`] before a context exists (via
/// [`FormatProfile::formats`]), which is required when allocating the textures
/// that will be passed into the context. Includes the fixed-format textures as
/// well as the variable ones so callers have a single authoritative source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsrFormats {
    /// Input color (current frame), at render resolution.
    pub color: wgpu::TextureFormat,
    /// Input depth, at render resolution.
    pub depth: wgpu::TextureFormat,
    /// Input motion vectors.
    pub motion_vectors: wgpu::TextureFormat,
    /// Output color, at presentation resolution.
    pub output: wgpu::TextureFormat,
    /// Shared dilated depth output, at render resolution.
    pub dilated_depth: wgpu::TextureFormat,
    /// Shared dilated motion vectors output, at render resolution. This is the
    /// only caller-provided format that changes with the profile.
    pub dilated_motion_vectors: wgpu::TextureFormat,
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::TextureFormat;

    #[test]
    fn dilated_motion_vectors_widen_on_core() {
        // The only caller-provided format that varies: `rg16float` is not a
        // valid storage texture on `Core`, so it widens to the 32-bit format.
        assert_eq!(
            FormatProfile::Core.formats().dilated_motion_vectors,
            TextureFormat::Rg32Float
        );
        assert_eq!(
            FormatProfile::Tier2.formats().dilated_motion_vectors,
            TextureFormat::Rg16Float
        );
        assert_eq!(
            FormatProfile::Native.formats().dilated_motion_vectors,
            TextureFormat::Rg16Float
        );
    }

    #[test]
    fn fixed_formats_are_profile_independent() {
        for profile in [
            FormatProfile::Core,
            FormatProfile::Tier2,
            FormatProfile::Native,
        ] {
            let f = profile.formats();
            assert_eq!(f.color, TextureFormat::Rgba16Float);
            assert_eq!(f.depth, TextureFormat::Depth32Float);
            assert_eq!(f.motion_vectors, TextureFormat::Rg16Float);
            assert_eq!(f.output, TextureFormat::Rgba16Float);
            assert_eq!(f.dilated_depth, TextureFormat::R32Float);
        }
    }
}
