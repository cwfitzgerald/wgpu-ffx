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
    /// The `texture-formats-tier2` capability set. Wide formats gain
    /// read-write access; `rg16float` read-write remains unavailable. Reached
    /// today through `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` on adapters
    /// whose reported capabilities cover the tier-2 set but not `rg16float`
    /// read-write — notably Metal.
    Tier2,
    /// Native desktop, relying on `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`
    /// with the adapter reporting every capability the upscaler uses,
    /// including `rg16float` read-write storage.
    Native,
}

impl FormatProfile {
    /// The richest profile supported by `device`, created from `adapter`.
    ///
    /// `Tier2` and `Native` require the device feature
    /// `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` (wgpu does not yet surface
    /// `texture-formats-tier1/2` directly). With that feature enabled, wgpu
    /// validates storage bindings against the capabilities the adapter
    /// reports per format, so each profile's required capabilities are
    /// verified through [`wgpu::Adapter::get_texture_format_features`] rather
    /// than assumed from the feature bit. An adapter that reports everything
    /// except `rg16float` read-write storage (e.g. Metal) lands on `Tier2`.
    pub fn from_adapter(adapter: &wgpu::Adapter, device: &wgpu::Device) -> FormatProfile {
        Self::classify(device.features(), |format| {
            adapter.get_texture_format_features(format)
        })
    }

    /// Whether `device`, created from `adapter`, can run at this profile.
    pub fn supported_by(self, adapter: &wgpu::Adapter, device: &wgpu::Device) -> bool {
        self.check_support(device.features(), |format| {
            adapter.get_texture_format_features(format)
        })
        .is_ok()
    }

    /// Pure core of [`FormatProfile::from_adapter`], decoupled from live wgpu
    /// objects so it can be tested against synthetic adapter capabilities.
    fn classify(
        device_features: wgpu::Features,
        adapter_format_features: impl Fn(wgpu::TextureFormat) -> wgpu::TextureFormatFeatures,
    ) -> FormatProfile {
        for profile in [FormatProfile::Native, FormatProfile::Tier2] {
            if profile
                .check_support(device_features, &adapter_format_features)
                .is_ok()
            {
                return profile;
            }
        }
        FormatProfile::Core
    }

    /// Check every capability this profile requires, describing the first
    /// missing one. `Core` needs only baseline WebGPU and always passes.
    pub(crate) fn check_support(
        self,
        device_features: wgpu::Features,
        adapter_format_features: impl Fn(wgpu::TextureFormat) -> wgpu::TextureFormatFeatures,
    ) -> Result<(), String> {
        if matches!(self, FormatProfile::Core) {
            return Ok(());
        }

        // Without adapter-specific format features, wgpu validates storage
        // bindings against the WebGPU spec guarantees, where every capability
        // below is unavailable. Once wgpu surfaces `texture-formats-tier1/2`,
        // those features become an alternative gate here.
        if !device_features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
            return Err(format!(
                "FormatProfile::{self:?} requires the \
                 TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES device feature"
            ));
        }

        for (format, required_flags) in self.format_requirements() {
            let reported = adapter_format_features(format);

            let missing = required_flags.difference(reported.flags);
            if !missing.is_empty() {
                return Err(format!(
                    "FormatProfile::{self:?} requires {format:?} to support {missing:?}, \
                     which the adapter does not report"
                ));
            }

            let required_usages =
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
            if !reported.allowed_usages.contains(required_usages) {
                return Err(format!(
                    "FormatProfile::{self:?} requires {format:?} to allow {:?}, \
                     which the adapter does not report",
                    required_usages.difference(reported.allowed_usages)
                ));
            }
        }

        Ok(())
    }

    /// The per-format capabilities this profile requires beyond baseline
    /// WebGPU, checked against the adapter-reported
    /// [`wgpu::TextureFormatFeatures`]. Mirrors the internal-texture table in
    /// `docs/texture-formats.md`.
    fn format_requirements(self) -> Vec<(wgpu::TextureFormat, wgpu::TextureFormatFeatureFlags)> {
        use wgpu::TextureFormat as Tf;
        use wgpu::TextureFormatFeatureFlags as Flags;

        // Shared by Tier2 and Native: accumulation / shading change (r8unorm
        // write-only, linearly sampled), new locks (r8unorm read-write), luma
        // and the fp16 intermediates (r16float write-only, linearly sampled),
        // dilated motion vectors (rg16float write-only, point-sampled).
        let common = [
            (
                Tf::R8Unorm,
                Flags::FILTERABLE | Flags::STORAGE_WRITE_ONLY | Flags::STORAGE_READ_WRITE,
            ),
            (Tf::R16Float, Flags::FILTERABLE | Flags::STORAGE_WRITE_ONLY),
            (Tf::Rg16Float, Flags::STORAGE_WRITE_ONLY),
        ];

        // The profiles differ only in the SPD mip chain, which is read-write
        // and linearly sampled: rg16float on Native, widened to rgba16float
        // on Tier2.
        let spd = match self {
            FormatProfile::Core => return Vec::new(),
            FormatProfile::Tier2 => (
                Tf::Rgba16Float,
                Flags::FILTERABLE | Flags::STORAGE_READ_WRITE,
            ),
            FormatProfile::Native => (Tf::Rg16Float, Flags::FILTERABLE | Flags::STORAGE_READ_WRITE),
        };

        common.into_iter().chain([spd]).collect()
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
    use wgpu::{TextureFormat, TextureFormatFeatureFlags as Flags};

    /// A synthetic adapter that reports `flags` for the formats listed in
    /// `caps` (and nothing for the rest), with usages that always allow
    /// sampled + storage binding.
    fn adapter_caps(
        caps: &[(TextureFormat, Flags)],
    ) -> impl Fn(TextureFormat) -> wgpu::TextureFormatFeatures + '_ {
        move |format| wgpu::TextureFormatFeatures {
            allowed_usages: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            flags: caps
                .iter()
                .find(|(f, _)| *f == format)
                .map(|(_, flags)| *flags)
                .unwrap_or_else(Flags::empty),
        }
    }

    /// Everything the upscaler uses, including `rg16float` read-write — a
    /// desktop Vulkan/D3D12-style adapter.
    const NATIVE_CAPS: &[(TextureFormat, Flags)] = &[
        (
            TextureFormat::R8Unorm,
            Flags::FILTERABLE
                .union(Flags::STORAGE_WRITE_ONLY)
                .union(Flags::STORAGE_READ_WRITE),
        ),
        (
            TextureFormat::R16Float,
            Flags::FILTERABLE.union(Flags::STORAGE_WRITE_ONLY),
        ),
        (
            TextureFormat::Rg16Float,
            Flags::FILTERABLE
                .union(Flags::STORAGE_WRITE_ONLY)
                .union(Flags::STORAGE_READ_WRITE),
        ),
        (
            TextureFormat::Rgba16Float,
            Flags::FILTERABLE
                .union(Flags::STORAGE_WRITE_ONLY)
                .union(Flags::STORAGE_READ_WRITE),
        ),
    ];

    /// Metal-style: tier-2 capabilities, but no `rg16float` read-write.
    const METAL_CAPS: &[(TextureFormat, Flags)] = &[
        (
            TextureFormat::R8Unorm,
            Flags::FILTERABLE
                .union(Flags::STORAGE_WRITE_ONLY)
                .union(Flags::STORAGE_READ_WRITE),
        ),
        (
            TextureFormat::R16Float,
            Flags::FILTERABLE.union(Flags::STORAGE_WRITE_ONLY),
        ),
        (
            TextureFormat::Rg16Float,
            Flags::FILTERABLE.union(Flags::STORAGE_WRITE_ONLY),
        ),
        (
            TextureFormat::Rgba16Float,
            Flags::FILTERABLE
                .union(Flags::STORAGE_WRITE_ONLY)
                .union(Flags::STORAGE_READ_WRITE),
        ),
    ];

    const ADAPTER_SPECIFIC: wgpu::Features =
        wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;

    #[test]
    fn core_without_adapter_specific_feature() {
        // Capabilities alone are not enough: without the device feature, wgpu
        // validates against spec guarantees, so detection must stay on Core.
        assert_eq!(
            FormatProfile::classify(wgpu::Features::empty(), adapter_caps(NATIVE_CAPS)),
            FormatProfile::Core
        );
    }

    #[test]
    fn native_with_full_capabilities() {
        assert_eq!(
            FormatProfile::classify(ADAPTER_SPECIFIC, adapter_caps(NATIVE_CAPS)),
            FormatProfile::Native
        );
    }

    #[test]
    fn tier2_when_rg16float_read_write_missing() {
        assert_eq!(
            FormatProfile::classify(ADAPTER_SPECIFIC, adapter_caps(METAL_CAPS)),
            FormatProfile::Tier2
        );
    }

    #[test]
    fn core_when_small_format_storage_missing() {
        // The feature bit alone must not imply Native (or Tier2): an adapter
        // reporting no extra per-format capabilities falls back to Core.
        assert_eq!(
            FormatProfile::classify(ADAPTER_SPECIFIC, adapter_caps(&[])),
            FormatProfile::Core
        );
    }

    #[test]
    fn check_support_names_missing_capability() {
        let err = FormatProfile::Native
            .check_support(ADAPTER_SPECIFIC, adapter_caps(METAL_CAPS))
            .unwrap_err();
        assert!(err.contains("Rg16Float"), "{err}");
        assert!(err.contains("STORAGE_READ_WRITE"), "{err}");
    }

    #[test]
    fn check_support_requires_storage_binding_usage() {
        let no_storage_usage = |format: TextureFormat| wgpu::TextureFormatFeatures {
            allowed_usages: wgpu::TextureUsages::TEXTURE_BINDING,
            ..adapter_caps(NATIVE_CAPS)(format)
        };
        let err = FormatProfile::Native
            .check_support(ADAPTER_SPECIFIC, no_storage_usage)
            .unwrap_err();
        assert!(err.contains("STORAGE_BINDING"), "{err}");
    }

    #[test]
    fn core_always_supported() {
        assert!(
            FormatProfile::Core
                .check_support(wgpu::Features::empty(), adapter_caps(&[]))
                .is_ok()
        );
    }

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
