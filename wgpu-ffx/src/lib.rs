use wgpu::util::DeviceExt;

mod lanczos2;

pub struct FsrContext {
    device: wgpu::Device,

    constants: Constants,
    constant_buffer: wgpu::Buffer,

    resources: FsrResources,

    first_execution: bool,
    previous_jitter_offset: [f32; 2],
    pre_exposure: f32,
    previous_frame_pre_exposure: f32,
}

impl FsrContext {
    pub fn new(info: FsrContextInfo) -> Self {
        let constants = Constants {
            max_upscale_size: info.max_upscale_size,
            velocity_factor: 1.0,
            reactiveness_scale: 1.0,
            shading_change_scale: 1.0,
            accumulation_added_per_frame: 1.0 / 3.0,
            min_disocclusion_accumulation: -1.0 / 3.0,
            ..Default::default()
        };

        todo!()
    }
}

pub struct FsrContextInfo {
    /// The wgpu device to use for GPU operations.
    pub device: wgpu::Device,
    /// The maximum resolution in pixels that the application will render will at.
    pub max_render_size: [u32; 2],
    /// The maximum resolution in pixels that FSR will upscale to.
    pub max_upscale_size: [u32; 2],
    /// Configuration options for the FSR context.
    pub flags: FsrContextFlags,
}

bitflags::bitflags! {
    /// Configuration options for the FSR context.
    pub struct FsrContextFlags: u32 {
        /// A bit indicating if the input color data provided is using a high-dynamic range.
        const HIGH_DYNAMIC_RANGE = 1 << 0;
        /// A bit indicating if the motion vectors are rendered at display resolution.
        const DISPLAY_RESOLUTION_MOTION_VECTORS = 1 << 1;
        /// A bit indicating that the motion vectors have the jittering pattern applied to them.
        const MOTION_VECTORS_JITTER_CANCELLATION = 1 << 2;
        /// A bit indicating that the input depth buffer data provided is inverted [1..0].
        const DEPTH_INVERTED = 1 << 3;
        /// A bit indicating that the input depth buffer data provided is using an infinite far plane.
        const DEPTH_INFINITE = 1 << 4;
        /// A bit indicating if automatic exposure should be applied to input color data.
        const AUTO_EXPOSURE = 1 << 5;
        /// A bit indicating that the application uses dynamic resolution scaling.
        const DYNAMIC_RESOLUTION = 1 << 6;
    }
}

#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Constants {
    render_size: [u32; 2],
    previous_frame_render_size: [u32; 2],

    upscale_size: [u32; 2],
    previous_frame_upscale_size: [u32; 2],

    max_render_size: [u32; 2],
    max_upscale_size: [u32; 2],

    device_to_view_depth: [f32; 4],

    jitter_offset: [f32; 2],
    previous_frame_jitter_offset: [f32; 2],

    motion_vector_scale: [f32; 2],
    downscale_factor: [f32; 2],

    motion_vector_jitter_cancellation: [f32; 2],
    tan_half_fov: f32,
    jitter_phase_count: f32,

    delta_time: f32,
    delta_pre_exposure: f32,
    view_space_to_meters_factor: f32,
    frame_index: f32,

    velocity_factor: f32,
    reactiveness_scale: f32,
    shading_change_scale: f32,
    accumulation_added_per_frame: f32,
    min_disocclusion_accumulation: f32,
}

struct FsrResources {
    accumulation_1: wgpu::Texture,
    accumulation_2: wgpu::Texture,
    luma_1: wgpu::Texture,
    luma_2: wgpu::Texture,
    intermediate_fp16x1: wgpu::Texture,
    shading_change: wgpu::Texture,
    new_locks: wgpu::Texture,
    internal_upscaled_1: wgpu::Texture,
    internal_upscaled_2: wgpu::Texture,
    spd_mips: wgpu::Texture,
    farthest_depth_mip1: wgpu::Texture,
    luma_history1: wgpu::Texture,
    luma_history2: wgpu::Texture,
    spd_atomic_count: wgpu::Texture,
    dilated_reactive_masks: wgpu::Texture,
    lanczos2_lut: wgpu::Texture,
    default_reactivity_mask: wgpu::Texture,
    default_exposure: wgpu::Texture,
    frame_info: wgpu::Texture,
}

impl FsrResources {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_render_size_array: [u32; 2],
        max_upscale_size_array: [u32; 2],
    ) -> Self {
        let lanczos2_lut_data = lanczos2::generate_lanczos2_lut();

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
            size: half_max_render_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
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
        let spd_atomic_counter = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FSR3 SPD Atomic Counter"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
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

        Self {
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
            spd_atomic_count: spd_atomic_counter,
            dilated_reactive_masks,
            lanczos2_lut,
            default_reactivity_mask,
            default_exposure,
            frame_info,
        }
    }
}
