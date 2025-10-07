#![allow(dead_code)]

mod constants;
mod lanczos2;
mod pass;
mod resources;

pub struct FsrContext {
    device: wgpu::Device,

    constants: constants::Constants,
    resources: resources::FsrResources,

    first_execution: bool,
    previous_jitter_offset: [f32; 2],
    pre_exposure: f32,
    previous_frame_pre_exposure: f32,
}

impl FsrContext {
    pub fn new(info: FsrContextInfo) -> Self {
        let constants = constants::FsrConstants {
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

pub struct FsrDispatchInfo {
    /// The <c><i>FfxCommandList</i></c> to record FSR3 rendering commands into.
    pub encoder: wgpu::CommandEncoder,

    /// A Texture containing the color buffer for the current frame (at render resolution).
    pub color: wgpu::Texture,
    /// A Texture containing 32bit depth values for the current frame (at render resolution).
    pub depth: wgpu::Texture,
    /// A Texture containing 2-dimensional motion vectors (at render resolution if <c><i>FFX_FSR3UPSCALER_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS</i></c> is not set).
    pub motion_vectors: wgpu::Texture,
    /// An optional Texture containing a 1x1 exposure value.
    pub exposure: Option<wgpu::Texture>,
    /// An optional Texture containing alpha value of reactive objects in the scene.
    pub reactive_mask: Option<wgpu::Texture>,
    /// An optional Texture containing alpha value of special objects in the scene.
    pub transparency_and_composition: Option<wgpu::Texture>,
    /// A Texture allocated as described in <TODO> that is used to emit dilated depth and share with following effects.
    pub dilated_depth: wgpu::Texture,
    /// A Texture allocated as described in <TODO> that is used to emit dilated motion vectors and share with following effects.
    pub dilated_motion_vectors: wgpu::Texture,
    /// A Buffer allocated as described in <TODO> that is used to emit reconstructed previous nearest depth and share with following effects.
    pub reconstructed_previous_depth: wgpu::Buffer,
    /// A Texture containing the output color buffer for the current frame (at presentation resolution).
    pub output: wgpu::Texture,

    /// The subpixel jitter offset applied to the camera.
    pub jitter_offset: [f32; 2],
    /// The scale factor to apply to motion vectors.
    pub motion_vector_scale: [f32; 2],

    /// The resolution that was used for rendering the input resources.
    pub render_size: [u32; 2],
    /// The resolution that the upscaler will output.
    pub upscale_size: [u32; 2],

    /// Enable an additional sharpening pass.
    pub enable_sharpening: bool,
    /// The sharpness value between 0 and 1, where 0 is no additional sharpness and 1 is maximum additional sharpness.
    pub sharpness: f32,
    /// The time elapsed since the last frame (expressed in milliseconds).
    pub frame_time_delta: f32,
    /// The pre exposure value (must be > 0.0f)
    pub pre_exposure: f32,
    /// A boolean value which when set to true, indicates the camera has moved discontinuously.
    pub reset_history: bool,
    /// The distance to the near plane of the camera.
    pub camera_near: f32,
    /// The distance to the far plane of the camera.
    pub camera_far: f32,
    /// The camera angle field of view in the vertical direction (expressed in radians).
    pub camera_fov_y: f32,
    /// The scale factor to convert view space units to meters
    pub view_space_to_meters_factor: f32,

    /// combination of FfxFsr3UpscalerDispatchFlags
    pub flags: FsrDispatchFlags,
}

bitflags::bitflags! {
    /// Configuration options for a single FSR dispatch.
    pub struct FsrDispatchFlags: u32 {
        /// A bit indicating that the interpolated output resource will contain debug views with relevant information.
        const FFX_FSR3UPSCALER_DISPATCH_DRAW_DEBUG_VIEW = 1 << 0;
    }
}

#[derive(Debug, Clone, Copy, Default)]
enum FrameKind {
    #[default]
    Even,
    Odd,
}

impl FrameKind {
    fn advance(&mut self) {
        *self = match self {
            FrameKind::Even => FrameKind::Odd,
            FrameKind::Odd => FrameKind::Even,
        }
    }
}
