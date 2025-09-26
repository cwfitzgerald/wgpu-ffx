mod lanczos2;

struct FsrContext {
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

        let lanczos2_lut = lanczos2::generate_lanczos2_lut();

        let half_max_render_size = [info.max_render_size[0] / 2, info.max_render_size[1] / 2];

        todo!()
    }
}

pub struct FsrContextInfo {
    /// The wgpu device to use for GPU operations.
    device: wgpu::Device,
    /// The maximum resolution in pixels that the application will render will at.
    max_render_size: [u32; 2],
    /// The maximum resolution in pixels that FSR will upscale to.
    max_upscale_size: [u32; 2],
    /// Configuration options for the FSR context.
    flags: FsrContextFlags,
}

bitflags::bitflags! {
    /// Configuration options for the FSR context.
    struct FsrContextFlags: u32 {
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
    internal_upscaled_color_1: wgpu::Texture,
    internal_upscaled_color_2: wgpu::Texture,
    spd_mips: wgpu::Texture,
    farthest_depth_mip1: wgpu::Texture,
    luma_history1: wgpu::Texture,
    luma_history2: wgpu::Texture,
    spd_atomic_count: wgpu::Buffer,
    dilated_reactive_masks: wgpu::Texture,
    lanczos2_lut: wgpu::Buffer,
    internal_default_reactivity_mask: wgpu::Texture,
    default_exposure: wgpu::Texture,
    frame_info: wgpu::Texture,
}
