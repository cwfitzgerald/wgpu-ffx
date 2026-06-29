use wgpu::util::DeviceExt as _;

use crate::{FormatProfile, FrameKind, pass::ResourceAccess};

use super::FsrDispatchInfo;

/// A pair of textures used for temporal double-buffering.
///
/// One texture holds the current frame's data (write target), while the other
/// holds the previous frame's data (read source). Which physical texture is
/// "current" flips each frame based on [`FrameKind`].
pub(crate) struct DoubleBuffered {
    current_on_even: wgpu::Texture,
    current_on_odd: wgpu::Texture,
}

impl DoubleBuffered {
    fn new(current_on_even: wgpu::Texture, current_on_odd: wgpu::Texture) -> Self {
        Self {
            current_on_even,
            current_on_odd,
        }
    }

    /// The texture being written to this frame.
    pub fn current(&self, kind: FrameKind) -> &wgpu::Texture {
        match kind {
            FrameKind::Even => &self.current_on_even,
            FrameKind::Odd => &self.current_on_odd,
        }
    }

    /// The texture that was written to last frame (read source).
    pub fn previous(&self, kind: FrameKind) -> &wgpu::Texture {
        match kind {
            FrameKind::Even => &self.current_on_odd,
            FrameKind::Odd => &self.current_on_even,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AccessType {
    Srv,
    Uav,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum FsrResourceName {
    InputColor,
    InputDepth,
    InputMotionVectors,
    InputExposure,
    InputReactiveMask,
    InputTransparencyAndComposition,

    OutputColor,
    OutputDilatedDepth,
    OutputDilatedMotionVectors,
    OutputReconstructedPreviousDepth,

    Constants,

    /// Current frame's accumulation (write target).
    AccumulationCurrent,
    /// Previous frame's accumulation (read source).
    AccumulationPrevious,
    /// Current frame's luma.
    Luma,
    /// Previous frame's luma.
    PreviousLuma,
    LumaInstability,
    ShadingChange,
    NewLocks,
    /// Current frame's internal upscaled color (write target for Accumulate,
    /// read source for RCAS and DebugView).
    InternalUpscaledCurrent,
    /// Previous frame's internal upscaled color (read source for Accumulate).
    InternalUpscaledPrevious,
    SpdMips,
    FarthestDepth,
    FarthestDepthMip1,
    /// Current frame's luma history (write target).
    LumaHistoryCurrent,
    /// Previous frame's luma history (read source).
    LumaHistoryPrevious,
    SpdAtomicCount,
    DilatedReactiveMasks,
    Lanczos2Lut,
    DefaultReactivityMask,
    DefaultExposure,
    FrameInfo,

    SamplerPointClamp,
    SamplerLinearClamp,
}

impl FsrResourceName {
    pub(crate) fn format(&self, profile: FormatProfile) -> wgpu::TextureFormat {
        use wgpu::TextureFormat as Tf;

        // On the Core profile, storage formats that baseline WebGPU cannot bind
        // are remapped: single-channel formats that are linearly sampled widen
        // to RGBA (keeping hardware filtering), while point-sampled / read-write
        // ones pack into a 32-bit format. These must agree with the FSR3_FMT_*
        // macros in the GLSL callbacks header. Tier2 and Native keep the native
        // formats; the SPD mips are the only Tier2 difference and are handled by
        // the SPD restructure. See docs/texture-formats.md.
        let core = matches!(profile, FormatProfile::Core);

        match self {
            FsrResourceName::InputColor
            | FsrResourceName::InputDepth
            | FsrResourceName::InputMotionVectors
            | FsrResourceName::InputExposure
            | FsrResourceName::InputReactiveMask
            | FsrResourceName::InputTransparencyAndComposition => {
                panic!("Input resources do not have a fixed format")
            }
            FsrResourceName::OutputColor => Tf::Rgba16Float,
            FsrResourceName::OutputDilatedDepth => Tf::R32Float,
            FsrResourceName::OutputDilatedMotionVectors => {
                if core {
                    Tf::Rg32Float
                } else {
                    Tf::Rg16Float
                }
            }
            FsrResourceName::OutputReconstructedPreviousDepth => {
                panic!("ReconstructedPreviousDepth is a buffer")
            }

            FsrResourceName::Constants => {
                panic!("Constants is a buffer")
            }

            FsrResourceName::AccumulationCurrent
            | FsrResourceName::AccumulationPrevious
            | FsrResourceName::ShadingChange => {
                if core {
                    Tf::Rgba8Unorm
                } else {
                    Tf::R8Unorm
                }
            }
            FsrResourceName::Luma
            | FsrResourceName::PreviousLuma
            | FsrResourceName::LumaInstability
            | FsrResourceName::FarthestDepth
            | FsrResourceName::FarthestDepthMip1 => {
                if core {
                    Tf::Rgba16Float
                } else {
                    Tf::R16Float
                }
            }
            FsrResourceName::NewLocks => {
                if core {
                    Tf::R32Float
                } else {
                    Tf::R8Unorm
                }
            }
            FsrResourceName::InternalUpscaledCurrent
            | FsrResourceName::InternalUpscaledPrevious => Tf::Rgba16Float,
            // `rg16float` read-write storage is impossible at every tier, so the
            // SPD mips only stay `rg16float` on Native. Core and Tier2 both widen
            // to `rgba16float` (Tier2 keeps the single-pass read-write SPD; Core
            // uses the write-only mip chain). Matches FSR3_FMT_SPD in the GLSL.
            FsrResourceName::SpdMips => {
                if matches!(profile, FormatProfile::Native) {
                    Tf::Rg16Float
                } else {
                    Tf::Rgba16Float
                }
            }
            FsrResourceName::LumaHistoryCurrent | FsrResourceName::LumaHistoryPrevious => {
                Tf::Rgba16Float
            }
            FsrResourceName::SpdAtomicCount => {
                panic!("SpdAtomicCount is a buffer")
            }
            FsrResourceName::DilatedReactiveMasks => Tf::Rgba8Unorm,
            FsrResourceName::Lanczos2Lut => {
                panic!("Lanczos2Lut format depends on device features, not the profile")
            }
            FsrResourceName::DefaultReactivityMask => Tf::R8Unorm,
            FsrResourceName::DefaultExposure => Tf::Rg32Float,
            FsrResourceName::FrameInfo => {
                panic!("FrameInfo is a buffer")
            }

            FsrResourceName::SamplerPointClamp | FsrResourceName::SamplerLinearClamp => {
                panic!("Samplers are Samplers")
            }
        }
    }

    pub(crate) fn to_bgl_entry(
        self,
        binding: u32,
        access_type: AccessType,
        profile: FormatProfile,
    ) -> wgpu::BindGroupLayoutEntry {
        match (self, access_type) {
            (
                FsrResourceName::InputColor
                | FsrResourceName::InputDepth
                | FsrResourceName::InputMotionVectors
                | FsrResourceName::InputExposure
                | FsrResourceName::InputReactiveMask
                | FsrResourceName::InputTransparencyAndComposition,
                AccessType::Uav,
            ) => {
                panic!("Input resources cannot be UAVs")
            }
            (FsrResourceName::Constants, AccessType::Uav) => {
                panic!("Constants cannot be UAVs")
            }
            (FsrResourceName::Constants, AccessType::Srv) => wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            (
                FsrResourceName::OutputReconstructedPreviousDepth | FsrResourceName::SpdAtomicCount,
                AccessType::Srv,
            ) => wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            (
                FsrResourceName::OutputReconstructedPreviousDepth | FsrResourceName::SpdAtomicCount,
                AccessType::Uav,
            ) => wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Frame info is a storage buffer: read-only when sampled (Srv),
            // read-write when produced (Uav).
            (FsrResourceName::FrameInfo, access) => wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: matches!(access, AccessType::Srv),
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            (FsrResourceName::SamplerPointClamp, _) => wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            (FsrResourceName::SamplerLinearClamp, _) => wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },

            (_, AccessType::Uav) => {
                let access = match self {
                    // Store-only storage textures: the shader only ever
                    // `imageStore`s to these, so they are bound write-only.
                    // (The genuinely read-modify-write storage textures —
                    // NewLocks, FrameInfo, SpdMips — fall through to ReadWrite.)
                    FsrResourceName::InternalUpscaledCurrent
                    | FsrResourceName::OutputColor
                    | FsrResourceName::OutputDilatedDepth
                    | FsrResourceName::OutputDilatedMotionVectors
                    | FsrResourceName::DilatedReactiveMasks
                    | FsrResourceName::AccumulationCurrent
                    | FsrResourceName::Luma
                    | FsrResourceName::LumaInstability
                    | FsrResourceName::FarthestDepth
                    | FsrResourceName::ShadingChange
                    | FsrResourceName::FarthestDepthMip1
                    | FsrResourceName::LumaHistoryCurrent => wgpu::StorageTextureAccess::WriteOnly,
                    _ => wgpu::StorageTextureAccess::ReadWrite,
                };
                wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access,
                        format: self.format(profile),
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }
            }
            (_, AccessType::Srv) => {
                // A `Float` SRV must be bound `filterable: false` whenever its
                // texture is a 32-bit float format the device cannot filter
                // without `float32-filterable` (R32Float / Rg32Float). wgpu
                // checks this at bind-group creation against the bound view's
                // format, independent of whether the device happens to have the
                // feature, so the layout must match the worst case.
                //
                // These three are 32-bit float at *every* profile and are only
                // ever point-loaded (`texelFetch` / indexing — no sampler), so
                // they are unfilterable everywhere:
                //   - InputDepth (caller depth, point-sampled),
                //   - InputExposure (Rg32Float; `r_input_exposure` is read only
                //     via `texelFetch`),
                //   - OutputDilatedDepth (R32Float; read only via
                //     `LoadDilatedDepth`/`texelFetch` — the linear-sampling
                //     `SampleDilatedDepth` helper is dead code, never called).
                //
                // On Core, the storage formats that additionally pack into a
                // 32-bit format (NewLocks → R32Float, dilated motion vectors →
                // Rg32Float) become unfilterable too; on Tier2/Native those stay
                // narrow (R8Unorm / Rg16Float) and remain filterable.
                let unfilterable = matches!(
                    self,
                    FsrResourceName::InputDepth
                        | FsrResourceName::InputExposure
                        | FsrResourceName::OutputDilatedDepth
                ) || (matches!(profile, FormatProfile::Core)
                    && matches!(
                        self,
                        FsrResourceName::NewLocks | FsrResourceName::OutputDilatedMotionVectors
                    ));
                let filterable = !unfilterable;
                wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }
            }
        }
    }
}

pub(crate) struct FsrResources {
    pub(crate) constant_buffer: wgpu::Buffer,

    pub(crate) accumulation: DoubleBuffered,
    pub(crate) luma: DoubleBuffered,
    pub(crate) intermediate_fp16x1: wgpu::Texture,
    pub(crate) shading_change: wgpu::Texture,
    pub(crate) new_locks: wgpu::Texture,
    pub(crate) internal_upscaled: DoubleBuffered,
    pub(crate) spd_mips: wgpu::Texture,
    /// Per-level source-index uniform for the Core write-only SPD chain. One
    /// 256-byte-aligned slot per mip level, slot `k` holding the `u32` value `k`.
    /// Static (resolution-independent); the downsample shader derives the
    /// per-level texel dimensions from the level index and `RenderSize()`.
    pub(crate) spd_level_buffer: wgpu::Buffer,
    /// Byte stride between [`Self::spd_level_buffer`] slots.
    pub(crate) spd_level_slot_size: u64,
    pub(crate) farthest_depth_mip1: wgpu::Texture,
    pub(crate) luma_history: DoubleBuffered,
    pub(crate) spd_atomic_counter: wgpu::Buffer,
    pub(crate) dilated_reactive_masks: wgpu::Texture,
    pub(crate) lanczos2_lut: wgpu::Texture,
    pub(crate) default_reactivity_mask: wgpu::Texture,
    pub(crate) default_exposure: wgpu::Texture,
    pub(crate) frame_info: wgpu::Buffer,

    pub(crate) sampler_point_clamp: wgpu::Sampler,
    pub(crate) sampler_linear_clamp: wgpu::Sampler,
}

impl FsrResources {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format_profile: FormatProfile,
        max_render_size_array: [u32; 2],
        max_upscale_size_array: [u32; 2],
    ) -> Self {
        let lanczos2_lut_data = crate::lanczos2::generate_lanczos2_lut();

        let max_render_size = wgpu::Extent3d {
            width: max_render_size_array[0],
            height: max_render_size_array[1],
            depth_or_array_layers: 1,
        };

        let half_max_render_size = wgpu::Extent3d {
            width: max_render_size_array[0] / 2,
            height: max_render_size_array[1] / 2,
            depth_or_array_layers: 1,
        };

        let max_upscale_size = wgpu::Extent3d {
            width: max_upscale_size_array[0],
            height: max_upscale_size_array[1],
            depth_or_array_layers: 1,
        };

        let constant_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FSR3 Constants"),
            size: std::mem::size_of::<crate::constants::Constants>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let render_tex = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: max_render_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };

        let upscale_tex = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: max_upscale_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };

        // Double-buffered resources: current_on_even written on even frames,
        // current_on_odd written on odd frames.
        // Internal-texture formats are profile-dependent; route them through
        // `format` so they always match the bind-group-layout formats.
        let fmt_accumulation = FsrResourceName::AccumulationCurrent.format(format_profile);
        let fmt_luma = FsrResourceName::Luma.format(format_profile);

        let accumulation = DoubleBuffered::new(
            render_tex("FSR3 Accumulation (even)", fmt_accumulation),
            render_tex("FSR3 Accumulation (odd)", fmt_accumulation),
        );

        let luma = DoubleBuffered::new(
            render_tex("FSR3 Luma (even)", fmt_luma),
            render_tex("FSR3 Luma (odd)", fmt_luma),
        );

        let internal_upscaled = DoubleBuffered::new(
            upscale_tex(
                "FSR3 Internal Upscaled (even)",
                wgpu::TextureFormat::Rgba16Float,
            ),
            upscale_tex(
                "FSR3 Internal Upscaled (odd)",
                wgpu::TextureFormat::Rgba16Float,
            ),
        );

        let luma_history = DoubleBuffered::new(
            render_tex("FSR3 Luma History (even)", wgpu::TextureFormat::Rgba16Float),
            render_tex("FSR3 Luma History (odd)", wgpu::TextureFormat::Rgba16Float),
        );

        let intermediate_fp16x1 = render_tex("FSR3 Intermediate FP16x1", fmt_luma);

        let shading_change = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Shading Change"),
            size: half_max_render_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FsrResourceName::ShadingChange.format(format_profile),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let new_locks = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 New Locks"),
            size: max_upscale_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FsrResourceName::NewLocks.format(format_profile),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let spd_mips = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 SPD Mips"),
            size: half_max_render_size,
            mip_level_count: half_max_render_size.max_mips(wgpu::TextureDimension::D2),
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FsrResourceName::SpdMips.format(format_profile),
            // RENDER_ATTACHMENT so each mip can be zeroed with a render-pass clear
            // (`CommandEncoder::clear_texture` is unimplemented on the webgpu
            // backend). `Rg16Float`/`Rgba16Float` are both color-renderable in
            // baseline WebGPU. See `FsrContext::clear_color_texture`.
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        // Per-level source-index uniform for the Core write-only SPD chain. One
        // 256-byte-aligned slot per physical mip level, each holding its level
        // index; bound at a per-dispatch offset. Resolution-independent, so it is
        // built once here.
        let spd_level_slot_size =
            (device.limits().min_uniform_buffer_offset_alignment as u64).max(4);
        let spd_level_count = half_max_render_size.max_mips(wgpu::TextureDimension::D2) as u64;
        let mut spd_level_data = vec![0u8; (spd_level_slot_size * spd_level_count) as usize];
        for level in 0..spd_level_count {
            let slot = (level * spd_level_slot_size) as usize;
            spd_level_data[slot..slot + 4].copy_from_slice(&(level as u32).to_le_bytes());
        }
        let spd_level_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FSR3 SPD Core Level Indices"),
            contents: &spd_level_data,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let farthest_depth_mip1 = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Farthest Depth Mip1"),
            size: half_max_render_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FsrResourceName::FarthestDepthMip1.format(format_profile),
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // This needs to be initialized to zero, but wgpu does this for us.
        let spd_atomic_counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FSR3 SPD Atomic Counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dilated_reactive_masks = render_tex(
            "FSR3 Dilated Reactive Masks",
            wgpu::TextureFormat::Rgba8Unorm,
        );

        // `r16snorm` requires the `TEXTURE_FORMAT_16BIT_NORM` device feature,
        // independent of the format profile (the LUT is sampled-only, so no
        // storage capability is involved). Without it the LUT widens to
        // filterable half floats and the upload converts the snorm values.
        let lanczos2_lut_format = if device
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
        {
            wgpu::TextureFormat::R16Snorm
        } else {
            wgpu::TextureFormat::R16Float
        };
        let lanczos2_lut_f16_data: Vec<half::f16>;
        let lanczos2_lut_bytes: &[u8] = match lanczos2_lut_format {
            wgpu::TextureFormat::R16Float => {
                lanczos2_lut_f16_data = lanczos2_lut_data
                    .iter()
                    .map(|&v| half::f16::from_f32(f32::from(v) / 32767.0))
                    .collect();
                bytemuck::cast_slice(&lanczos2_lut_f16_data)
            }
            _ => bytemuck::cast_slice(&lanczos2_lut_data),
        };
        let lanczos2_lut = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("FSR3 Lanczos2 LUT"),
                size: wgpu::Extent3d {
                    width: lanczos2_lut_data.len() as u32,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: lanczos2_lut_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::default(),
            lanczos2_lut_bytes,
        );

        // This needs to be initialized to zero, but wgpu does this for us.
        let default_reactivity_mask = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Default Reactivity Mask"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let default_exposure = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Default Exposure"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // A single `vec4<f32>` of per-frame metadata, read-modify-written by the
        // compute passes. Backed by a storage buffer so the read-write access
        // works on every format profile.
        let frame_info = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FSR3 Frame Info"),
            size: 4 * std::mem::size_of::<f32>() as u64,
            // COPY_SRC so tests can read back the auto-exposure / scene-average
            // luma to compare the SPD pyramids across format profiles.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sampler_linear_clamp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FSR3 Sampler Linear Clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let sampler_point_clamp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FSR3 Sampler Point Clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            constant_buffer,
            accumulation,
            luma,
            intermediate_fp16x1,
            shading_change,
            new_locks,
            internal_upscaled,
            spd_mips,
            spd_level_buffer,
            spd_level_slot_size,
            farthest_depth_mip1,
            luma_history,
            spd_atomic_counter,
            dilated_reactive_masks,
            lanczos2_lut,
            default_reactivity_mask,
            default_exposure,
            frame_info,
            sampler_point_clamp,
            sampler_linear_clamp,
        }
    }

    pub(crate) fn to_view(
        &self,
        dispatch: &FsrDispatchInfo,
        access: ResourceAccess,
        kind: FrameKind,
    ) -> OwnedBindingResource {
        let label;
        let descriptor = match access.desc {
            Some(desc) => desc,
            None => {
                label = format!("FSR3 Resource View: {:?}", access.name);
                wgpu::TextureViewDescriptor {
                    label: Some(&label),
                    ..Default::default()
                }
            }
        };

        match access.name {
            // --- External (dispatch-provided) resources ---
            FsrResourceName::InputColor => {
                OwnedBindingResource::View(dispatch.color.create_view(&descriptor))
            }
            FsrResourceName::InputDepth => {
                OwnedBindingResource::View(dispatch.depth.create_view(&descriptor))
            }
            FsrResourceName::InputMotionVectors => {
                OwnedBindingResource::View(dispatch.motion_vectors.create_view(&descriptor))
            }
            FsrResourceName::InputExposure => {
                if let Some(exposure) = &dispatch.exposure {
                    OwnedBindingResource::View(exposure.create_view(&descriptor))
                } else {
                    OwnedBindingResource::View(self.default_exposure.create_view(&descriptor))
                }
            }
            FsrResourceName::InputReactiveMask => {
                if let Some(reactive_mask) = &dispatch.reactive_mask {
                    OwnedBindingResource::View(reactive_mask.create_view(&descriptor))
                } else {
                    OwnedBindingResource::View(
                        self.default_reactivity_mask.create_view(&descriptor),
                    )
                }
            }
            FsrResourceName::InputTransparencyAndComposition => {
                if let Some(transparency_and_composition) = &dispatch.transparency_and_composition {
                    OwnedBindingResource::View(
                        transparency_and_composition.create_view(&descriptor),
                    )
                } else {
                    // Note: We use the default reactivity mask here.
                    OwnedBindingResource::View(
                        self.default_reactivity_mask.create_view(&descriptor),
                    )
                }
            }
            FsrResourceName::OutputColor => {
                OwnedBindingResource::View(dispatch.output.create_view(&descriptor))
            }
            FsrResourceName::OutputDilatedDepth => {
                OwnedBindingResource::View(dispatch.dilated_depth.create_view(&descriptor))
            }
            FsrResourceName::OutputDilatedMotionVectors => {
                OwnedBindingResource::View(dispatch.dilated_motion_vectors.create_view(&descriptor))
            }
            FsrResourceName::OutputReconstructedPreviousDepth => {
                OwnedBindingResource::Buffer(dispatch.reconstructed_previous_depth.clone())
            }

            // --- Uniform buffer ---
            FsrResourceName::Constants => {
                OwnedBindingResource::Buffer(self.constant_buffer.clone())
            }

            // --- Double-buffered resources (temporal side is in the name) ---
            FsrResourceName::AccumulationCurrent => {
                OwnedBindingResource::View(self.accumulation.current(kind).create_view(&descriptor))
            }
            FsrResourceName::AccumulationPrevious => OwnedBindingResource::View(
                self.accumulation.previous(kind).create_view(&descriptor),
            ),
            FsrResourceName::Luma => {
                OwnedBindingResource::View(self.luma.current(kind).create_view(&descriptor))
            }
            FsrResourceName::PreviousLuma => {
                OwnedBindingResource::View(self.luma.previous(kind).create_view(&descriptor))
            }
            FsrResourceName::InternalUpscaledCurrent => OwnedBindingResource::View(
                self.internal_upscaled
                    .current(kind)
                    .create_view(&descriptor),
            ),
            FsrResourceName::InternalUpscaledPrevious => OwnedBindingResource::View(
                self.internal_upscaled
                    .previous(kind)
                    .create_view(&descriptor),
            ),
            FsrResourceName::LumaHistoryCurrent => {
                OwnedBindingResource::View(self.luma_history.current(kind).create_view(&descriptor))
            }
            FsrResourceName::LumaHistoryPrevious => OwnedBindingResource::View(
                self.luma_history.previous(kind).create_view(&descriptor),
            ),

            // --- Single-instance internal resources ---
            FsrResourceName::LumaInstability | FsrResourceName::FarthestDepth => {
                OwnedBindingResource::View(self.intermediate_fp16x1.create_view(&descriptor))
            }
            FsrResourceName::ShadingChange => {
                OwnedBindingResource::View(self.shading_change.create_view(&descriptor))
            }
            FsrResourceName::NewLocks => {
                OwnedBindingResource::View(self.new_locks.create_view(&descriptor))
            }
            FsrResourceName::SpdMips => {
                OwnedBindingResource::View(self.spd_mips.create_view(&descriptor))
            }
            FsrResourceName::FarthestDepthMip1 => {
                OwnedBindingResource::View(self.farthest_depth_mip1.create_view(&descriptor))
            }
            FsrResourceName::SpdAtomicCount => {
                OwnedBindingResource::Buffer(self.spd_atomic_counter.clone())
            }
            FsrResourceName::DilatedReactiveMasks => {
                OwnedBindingResource::View(self.dilated_reactive_masks.create_view(&descriptor))
            }
            FsrResourceName::Lanczos2Lut => {
                OwnedBindingResource::View(self.lanczos2_lut.create_view(&descriptor))
            }
            FsrResourceName::DefaultReactivityMask => {
                OwnedBindingResource::View(self.default_reactivity_mask.create_view(&descriptor))
            }
            FsrResourceName::DefaultExposure => {
                OwnedBindingResource::View(self.default_exposure.create_view(&descriptor))
            }
            FsrResourceName::FrameInfo => OwnedBindingResource::Buffer(self.frame_info.clone()),

            // --- Samplers ---
            FsrResourceName::SamplerPointClamp => {
                OwnedBindingResource::Sampler(self.sampler_point_clamp.clone())
            }
            FsrResourceName::SamplerLinearClamp => {
                OwnedBindingResource::Sampler(self.sampler_linear_clamp.clone())
            }
        }
    }
}

pub(crate) enum OwnedBindingResource {
    View(wgpu::TextureView),
    Buffer(wgpu::Buffer),
    Sampler(wgpu::Sampler),
}

impl<'a> From<&'a OwnedBindingResource> for wgpu::BindingResource<'a> {
    fn from(value: &'a OwnedBindingResource) -> Self {
        match value {
            OwnedBindingResource::View(v) => wgpu::BindingResource::TextureView(v),
            OwnedBindingResource::Buffer(b) => b.as_entire_binding(),
            OwnedBindingResource::Sampler(s) => wgpu::BindingResource::Sampler(s),
        }
    }
}
