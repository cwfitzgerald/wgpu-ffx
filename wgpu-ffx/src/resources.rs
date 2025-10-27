use wgpu::util::DeviceExt as _;

use crate::FrameKind;

use super::FsrDispatchInfo;

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

    Accumulation,
    Luma,
    PreviousLuma,
    LumaInstability,
    ShadingChange,
    NewLocks,
    InternalUpscaled,
    SpdMips,
    FarthestDepth,
    FarthestDepthMip1,
    LumaHistory,
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
    pub(crate) fn format(&self) -> wgpu::TextureFormat {
        match self {
            FsrResourceName::InputColor
            | FsrResourceName::InputDepth
            | FsrResourceName::InputMotionVectors
            | FsrResourceName::InputExposure
            | FsrResourceName::InputReactiveMask
            | FsrResourceName::InputTransparencyAndComposition => {
                panic!("Input resources do not have a fixed format")
            }
            FsrResourceName::OutputColor => wgpu::TextureFormat::Rgba16Float,
            FsrResourceName::OutputDilatedDepth => wgpu::TextureFormat::R32Float,
            FsrResourceName::OutputDilatedMotionVectors => wgpu::TextureFormat::Rg16Float,
            FsrResourceName::OutputReconstructedPreviousDepth => {
                panic!("ReconstructedPreviousDepth is a buffer")
            }

            FsrResourceName::Constants => {
                panic!("Constants is a buffer")
            }

            FsrResourceName::Accumulation => wgpu::TextureFormat::R8Unorm,
            FsrResourceName::Luma | FsrResourceName::PreviousLuma => wgpu::TextureFormat::R16Float,
            FsrResourceName::LumaInstability | FsrResourceName::FarthestDepth => {
                wgpu::TextureFormat::R16Float
            }
            FsrResourceName::ShadingChange => wgpu::TextureFormat::R8Unorm,
            FsrResourceName::NewLocks => wgpu::TextureFormat::R8Unorm,
            FsrResourceName::InternalUpscaled => wgpu::TextureFormat::Rgba16Float,
            FsrResourceName::SpdMips => wgpu::TextureFormat::Rg16Float,
            FsrResourceName::FarthestDepthMip1 => wgpu::TextureFormat::R16Float,
            FsrResourceName::LumaHistory => wgpu::TextureFormat::Rgba8Unorm,
            FsrResourceName::SpdAtomicCount => {
                panic!("SpdAtomicCount is a buffer")
            }
            FsrResourceName::DilatedReactiveMasks => wgpu::TextureFormat::Rgba8Unorm,
            FsrResourceName::Lanczos2Lut => wgpu::TextureFormat::R16Snorm,
            FsrResourceName::DefaultReactivityMask => wgpu::TextureFormat::R8Unorm,
            FsrResourceName::DefaultExposure => wgpu::TextureFormat::Rg32Float,
            FsrResourceName::FrameInfo => wgpu::TextureFormat::Rgba32Float,

            FsrResourceName::SamplerPointClamp | FsrResourceName::SamplerLinearClamp => {
                panic!("Samplers are Samplers")
            }
        }
    }

    pub(crate) fn to_bgl_entry(
        self,
        binding: u32,
        access_type: AccessType,
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
                    FsrResourceName::InternalUpscaled
                    | FsrResourceName::OutputColor
                    | FsrResourceName::OutputDilatedDepth
                    | FsrResourceName::OutputDilatedMotionVectors
                    | FsrResourceName::DilatedReactiveMasks => {
                        wgpu::StorageTextureAccess::WriteOnly
                    }
                    _ => wgpu::StorageTextureAccess::ReadWrite,
                };
                wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: access,
                        format: self.format(),
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }
            }
            (_, AccessType::Srv) => wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        }
    }
}

pub(crate) struct FsrResources {
    pub(crate) constant_buffer: wgpu::Buffer,

    pub(crate) accumulation_1: wgpu::Texture,
    pub(crate) accumulation_2: wgpu::Texture,
    pub(crate) luma_1: wgpu::Texture,
    pub(crate) luma_2: wgpu::Texture,
    pub(crate) intermediate_fp16x1: wgpu::Texture,
    pub(crate) shading_change: wgpu::Texture,
    pub(crate) new_locks: wgpu::Texture,
    pub(crate) internal_upscaled_1: wgpu::Texture,
    pub(crate) internal_upscaled_2: wgpu::Texture,
    pub(crate) spd_mips: wgpu::Texture,
    pub(crate) farthest_depth_mip1: wgpu::Texture,
    pub(crate) luma_history1: wgpu::Texture,
    pub(crate) luma_history2: wgpu::Texture,
    pub(crate) spd_atomic_counter: wgpu::Buffer,
    pub(crate) dilated_reactive_masks: wgpu::Texture,
    pub(crate) lanczos2_lut: wgpu::Texture,
    pub(crate) default_reactivity_mask: wgpu::Texture,
    pub(crate) default_exposure: wgpu::Texture,
    pub(crate) frame_info: wgpu::Texture,

    pub(crate) sampler_point_clamp: wgpu::Sampler,
    pub(crate) sampler_linear_clamp: wgpu::Sampler,
}

impl FsrResources {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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

        let accumulation_1 = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Accumulation 1"),
            size: max_render_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let accumulation_2 = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Accumulation 2"),
            size: max_render_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let luma_1 = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Luma 1"),
            size: max_render_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let luma_2 = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Luma 2"),
            size: max_render_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let intermediate_fp16x1 = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Intermediate FP16x1"),
            size: max_render_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let shading_change = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Shading Change"),
            size: half_max_render_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let new_locks = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 New Locks"),
            size: max_upscale_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let internal_upscaled_1 = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Internal Upscaled 1"),
            size: max_upscale_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let internal_upscaled_2 = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Internal Upscaled 2"),
            size: max_upscale_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let spd_mips = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 SPD Mips"),
            size: half_max_render_size,
            mip_level_count: half_max_render_size.max_mips(wgpu::TextureDimension::D2),
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let farthest_depth_mip1 = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Farthest Depth Mip1"),
            size: half_max_render_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let luma_history1 = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Luma History1"),
            size: max_render_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let luma_history2 = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Luma History2"),
            size: max_render_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        // This needs to be initialized to zero, but wgpu does this for us.
        let spd_atomic_counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FSR3 SPD Atomic Counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dilated_reactive_masks = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Dilated Reactive Masks"),
            size: max_render_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

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
                format: wgpu::TextureFormat::R16Snorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::default(),
            bytemuck::cast_slice(&lanczos2_lut_data),
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

        let frame_info = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 Frame Info"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let sampler_linear_clamp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FSR3 Sampler Linear Clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let sampler_point_clamp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FSR3 Sampler Point Clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            constant_buffer,
            accumulation_1,
            accumulation_2,
            luma_1,
            luma_2,
            intermediate_fp16x1,
            shading_change,
            new_locks,
            internal_upscaled_1,
            internal_upscaled_2,
            spd_mips,
            farthest_depth_mip1,
            luma_history1,
            luma_history2,
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
        name: FsrResourceName,
        kind: FrameKind,
        descriptor: Option<wgpu::TextureViewDescriptor>,
    ) -> OwnedBindingResource {
        let descriptor = descriptor.unwrap_or_default();

        match name {
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

            FsrResourceName::Constants => {
                OwnedBindingResource::Buffer(self.constant_buffer.clone())
            }

            FsrResourceName::Accumulation => {
                if kind == FrameKind::Odd {
                    OwnedBindingResource::View(self.accumulation_1.create_view(&descriptor))
                } else {
                    OwnedBindingResource::View(self.accumulation_2.create_view(&descriptor))
                }
            }
            FsrResourceName::Luma => {
                if kind == FrameKind::Odd {
                    OwnedBindingResource::View(self.luma_1.create_view(&descriptor))
                } else {
                    OwnedBindingResource::View(self.luma_2.create_view(&descriptor))
                }
            }
            FsrResourceName::PreviousLuma => {
                if kind == FrameKind::Odd {
                    OwnedBindingResource::View(self.luma_2.create_view(&descriptor))
                } else {
                    OwnedBindingResource::View(self.luma_1.create_view(&descriptor))
                }
            }
            FsrResourceName::LumaInstability | FsrResourceName::FarthestDepth => {
                OwnedBindingResource::View(self.intermediate_fp16x1.create_view(&descriptor))
            }
            FsrResourceName::ShadingChange => {
                OwnedBindingResource::View(self.shading_change.create_view(&descriptor))
            }
            FsrResourceName::NewLocks => {
                OwnedBindingResource::View(self.new_locks.create_view(&descriptor))
            }
            FsrResourceName::InternalUpscaled => {
                if kind == FrameKind::Odd {
                    OwnedBindingResource::View(self.internal_upscaled_1.create_view(&descriptor))
                } else {
                    OwnedBindingResource::View(self.internal_upscaled_2.create_view(&descriptor))
                }
            }
            FsrResourceName::SpdMips => {
                OwnedBindingResource::View(self.spd_mips.create_view(&descriptor))
            }
            FsrResourceName::FarthestDepthMip1 => {
                OwnedBindingResource::View(self.farthest_depth_mip1.create_view(&descriptor))
            }
            FsrResourceName::LumaHistory => {
                if kind == FrameKind::Odd {
                    OwnedBindingResource::View(self.luma_history1.create_view(&descriptor))
                } else {
                    OwnedBindingResource::View(self.luma_history2.create_view(&descriptor))
                }
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
            FsrResourceName::FrameInfo => {
                OwnedBindingResource::View(self.frame_info.create_view(&descriptor))
            }
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
