use wgpu::util::DeviceExt;

mod lanczos2;

pub struct FsrContext {
    device: wgpu::Device,

    constants: Constants,
    resources: FsrResources,

    first_execution: bool,
    previous_jitter_offset: [f32; 2],
    pre_exposure: f32,
    previous_frame_pre_exposure: f32,
}

impl FsrContext {
    pub fn new(info: FsrContextInfo) -> Self {
        let constants = FsrConstants {
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

enum FsrPass {
    /// A pass which prepares game inputs for later passes
    PrepareInputs,
    /// A pass which generates the luminance mipmap chain for the current frame.
    LumaPyramid,
    /// A pass which generates the shading change detection mipmap chain for the current frame.
    ShadingChangePyramid,
    /// A pass which estimates shading changes for the current frame
    ShadingChange,
    /// A pass which prepares accumulation relevant information
    PrepareReactivity,
    /// A pass which estimates temporal instability of the luminance changes.
    LumaInstability,
    /// A pass which performs upscaling.
    Accumulate,
    /// A pass which performs upscaling when sharpening is used.
    AccumulateSharpen,
    /// A pass which performs sharpening.
    Rcas,
    /// A pass which draws some internal resources, for debugging purposes
    DebugView,
    /// An optional pass to generate a reactive mask.
    GenerateReactive,
}

impl FsrPass {
    fn label(&self) -> &'static str {
        match self {
            FsrPass::PrepareInputs => "FSR3 Prepare Inputs",
            FsrPass::LumaPyramid => "FSR3 Luma Pyramid",
            FsrPass::ShadingChangePyramid => "FSR3 Shading Change Pyramid",
            FsrPass::ShadingChange => "FSR3 Shading Change",
            FsrPass::PrepareReactivity => "FSR3 Prepare Reactivity",
            FsrPass::LumaInstability => "FSR3 Luma Instability",
            FsrPass::Accumulate => "FSR3 Accumulate",
            FsrPass::AccumulateSharpen => "FSR3 Accumulate Sharpen",
            FsrPass::Rcas => "FSR3 RCAS",
            FsrPass::DebugView => "FSR3 Debug View",
            FsrPass::GenerateReactive => "FSR3 Generate Reactive",
        }
    }

    fn resources(&self, flags: FsrContextFlags) -> Vec<ResourceAccess> {
        use AccessType::*;
        use FsrResourceName::*;

        let motion_vectors = if flags.contains(FsrContextFlags::DISPLAY_RESOLUTION_MOTION_VECTORS) {
            OutputDilatedMotionVectors
        } else {
            InputMotionVectors
        };

        #[rustfmt::skip]
        let ret = match self {
            FsrPass::Accumulate | FsrPass::AccumulateSharpen => vec![
                ResourceAccess { name: InputExposure, access_type: SRV, desc: None },
                ResourceAccess { name: DilatedReactiveMasks, access_type: SRV, desc: None },
                ResourceAccess { name: motion_vectors, access_type: SRV, desc: None },
                ResourceAccess { name: InternalUpscaled, access_type: SRV, desc: None },
                ResourceAccess { name: Lanczos2Lut, access_type: SRV, desc: None },
                ResourceAccess { name: FarthestDepthMip1, access_type: SRV, desc: None },
                ResourceAccess { name: Luma, access_type: SRV, desc: None },
                ResourceAccess { name: LumaInstability, access_type: SRV, desc: None },
                ResourceAccess { name: InputColor, access_type: SRV, desc: None },
                ResourceAccess { name: InternalUpscaled, access_type: UAV, desc: None },
                ResourceAccess { name: OutputColor, access_type: UAV, desc: None },
                ResourceAccess { name: NewLocks, access_type: UAV, desc: None },
                ResourceAccess { name: Constants, access_type: SRV, desc: None },
            ],
            FsrPass::GenerateReactive => todo!("Need to map FfxFsr3UpscalerGenerateReactiveDescription"),
            FsrPass::DebugView => vec![
                ResourceAccess { name: DilatedReactiveMasks, access_type: SRV, desc: None },
                ResourceAccess { name: OutputDilatedMotionVectors, access_type: SRV, desc: None },
                ResourceAccess { name: OutputDilatedDepth, access_type: SRV, desc: None },
                ResourceAccess { name: InternalUpscaled, access_type: SRV, desc: None },
                ResourceAccess { name: InputExposure, access_type: SRV, desc: None },
                ResourceAccess { name: OutputColor, access_type: UAV, desc: None },
                ResourceAccess { name: Constants, access_type: SRV, desc: None },
            ],
            FsrPass::LumaInstability => vec![
                ResourceAccess { name: InputExposure, access_type: SRV, desc: None },
                ResourceAccess { name: DilatedReactiveMasks, access_type: SRV, desc: None },
                ResourceAccess { name: OutputDilatedMotionVectors, access_type: SRV, desc: None },
                ResourceAccess { name: FrameInfo, access_type: SRV, desc: None },
                ResourceAccess { name: LumaHistory, access_type: SRV, desc: None },
                ResourceAccess { name: FarthestDepthMip1, access_type: SRV, desc: None },
                ResourceAccess { name: Luma, access_type: SRV, desc: None },
                ResourceAccess { name: LumaHistory, access_type: UAV, desc: None },
                ResourceAccess { name: LumaInstability, access_type: UAV, desc: None },
                ResourceAccess { name: Constants, access_type: SRV, desc: None },
            ],
            FsrPass::LumaPyramid => vec![
                ResourceAccess { name: Luma, access_type: SRV, desc: None },
                ResourceAccess { name: FarthestDepth, access_type: SRV, desc: None },
                ResourceAccess { name: SpdAtomicCount, access_type: UAV, desc: None },
                ResourceAccess { name: FrameInfo, access_type: UAV, desc: None },
                ResourceAccess {
                    name: SpdMips,
                    access_type: UAV,
                    desc: Some(wgpu::TextureViewDescriptor {
                        aspect: wgpu::TextureAspect::All,
                        base_mip_level: 0,
                        mip_level_count: Some(1),
                        ..Default::default()
                    }),
                },
                ResourceAccess {
                    name: SpdMips,
                    access_type: UAV,
                    desc: Some(wgpu::TextureViewDescriptor {
                        aspect: wgpu::TextureAspect::All,
                        base_mip_level: 1,
                        mip_level_count: Some(1),
                        ..Default::default()
                    }),
                },
                ResourceAccess {
                    name: SpdMips,
                    access_type: UAV,
                    desc: Some(wgpu::TextureViewDescriptor {
                        aspect: wgpu::TextureAspect::All,
                        base_mip_level: 2,
                        mip_level_count: Some(1),
                        ..Default::default()
                    }),
                },
                ResourceAccess {
                    name: SpdMips,
                    access_type: UAV,
                    desc: Some(wgpu::TextureViewDescriptor {
                        aspect: wgpu::TextureAspect::All,
                        base_mip_level: 3,
                        mip_level_count: Some(1),
                        ..Default::default()
                    }),
                },
                ResourceAccess {
                    name: SpdMips,
                    access_type: UAV,
                    desc: Some(wgpu::TextureViewDescriptor {
                        aspect: wgpu::TextureAspect::All,
                        base_mip_level: 4,
                        mip_level_count: Some(1),
                        ..Default::default()
                    }),
                },
                ResourceAccess {
                    name: SpdMips,
                    access_type: UAV,
                    desc: Some(wgpu::TextureViewDescriptor {
                        aspect: wgpu::TextureAspect::All,
                        base_mip_level: 5,
                        mip_level_count: None,
                        ..Default::default()
                    }),
                },
                ResourceAccess { name: FarthestDepthMip1, access_type: UAV, desc: None },
                ResourceAccess { name: Constants, access_type: SRV, desc: None },
            ],
            FsrPass::PrepareInputs => vec![
                ResourceAccess { name: InputMotionVectors, access_type: SRV, desc: None },
                ResourceAccess { name: InputDepth, access_type: SRV, desc: None },
                ResourceAccess { name: InputColor, access_type: SRV, desc: None },
                ResourceAccess { name: OutputDilatedMotionVectors, access_type: UAV, desc: None },
                ResourceAccess { name: OutputDilatedDepth, access_type: UAV, desc: None },
                ResourceAccess { name: OutputReconstructedPreviousDepth, access_type: UAV, desc: None },
                ResourceAccess { name: FarthestDepth, access_type: UAV, desc: None },
                ResourceAccess { name: Luma, access_type: UAV, desc: None },
                ResourceAccess { name: Constants, access_type: SRV, desc: None },
            ],
            FsrPass::PrepareReactivity => vec![
                ResourceAccess { name: OutputReconstructedPreviousDepth, access_type: SRV, desc: None },
                ResourceAccess { name: OutputDilatedMotionVectors, access_type: SRV, desc: None },
                ResourceAccess { name: OutputDilatedDepth, access_type: SRV, desc: None },
                ResourceAccess { name: InputReactiveMask, access_type: SRV, desc: None },
                ResourceAccess { name: InputTransparencyAndComposition, access_type: SRV, desc: None },
                ResourceAccess { name: Accumulation, access_type: SRV, desc: None },
                ResourceAccess { name: ShadingChange, access_type: SRV, desc: None },
                ResourceAccess { name: Luma, access_type: SRV, desc: None },
                ResourceAccess { name: InputExposure, access_type: SRV, desc: None },

                ResourceAccess { name: DilatedReactiveMasks, access_type: UAV, desc: None },
                ResourceAccess { name: NewLocks, access_type: UAV, desc: None },
                ResourceAccess { name: Accumulation, access_type: UAV, desc: None },

                ResourceAccess { name: Constants, access_type: SRV, desc: None },
            ],
            FsrPass::Rcas => vec![
                ResourceAccess { name: InputExposure, access_type: SRV, desc: None },
                ResourceAccess { name: InternalUpscaled, access_type: SRV, desc: None },
                ResourceAccess { name: OutputColor, access_type: UAV, desc: None },
                ResourceAccess { name: Constants, access_type: SRV, desc: None },
            ],
            FsrPass::ShadingChange => vec![
                ResourceAccess { name: SpdMips, access_type: SRV, desc: None },
                ResourceAccess { name: ShadingChange, access_type: UAV, desc: None },
                ResourceAccess { name: Constants, access_type: SRV, desc: None },
            ],
            FsrPass::ShadingChangePyramid => vec![
                // SRV bindings
                ResourceAccess { name: FsrResourceName::Luma, access_type: AccessType::SRV, desc: None },
                ResourceAccess { name: FsrResourceName::LumaHistory, access_type: AccessType::SRV, desc: None },
                ResourceAccess { name: FsrResourceName::OutputDilatedMotionVectors, access_type: AccessType::SRV, desc: None },
                ResourceAccess { name: FsrResourceName::InputExposure, access_type: AccessType::SRV, desc: None },
                ResourceAccess { name: FsrResourceName::SpdAtomicCount, access_type: AccessType::UAV, desc: None },
                ResourceAccess {
                    name: FsrResourceName::SpdMips,
                    access_type: AccessType::UAV,
                    desc: Some(wgpu::TextureViewDescriptor {
                        base_mip_level: 0,
                        mip_level_count: Some(1),
                        ..Default::default()
                    }),
                },
                ResourceAccess {
                    name: FsrResourceName::SpdMips,
                    access_type: AccessType::UAV,
                    desc: Some(wgpu::TextureViewDescriptor {
                        base_mip_level: 1,
                        mip_level_count: Some(1),
                        ..Default::default()
                    }),
                },
                ResourceAccess {
                    name: FsrResourceName::SpdMips,
                    access_type: AccessType::UAV,
                    desc: Some(wgpu::TextureViewDescriptor {
                        base_mip_level: 2,
                        mip_level_count: Some(1),
                        ..Default::default()
                    }),
                },
                ResourceAccess {
                    name: FsrResourceName::SpdMips,
                    access_type: AccessType::UAV,
                    desc: Some(wgpu::TextureViewDescriptor {
                        base_mip_level: 3,
                        mip_level_count: Some(1),
                        ..Default::default()
                    }),
                },
                ResourceAccess {
                    name: FsrResourceName::SpdMips,
                    access_type: AccessType::UAV,
                    desc: Some(wgpu::TextureViewDescriptor {
                        base_mip_level: 4,
                        mip_level_count: Some(1),
                        ..Default::default()
                    }),
                },
                ResourceAccess {
                    name: FsrResourceName::SpdMips,
                    access_type: AccessType::UAV,
                    desc: Some(wgpu::TextureViewDescriptor {
                        base_mip_level: 5,
                        mip_level_count: None,
                        ..Default::default()
                    }),
                },
                ResourceAccess { name: FsrResourceName::Constants, access_type: AccessType::SRV, desc: None }, // CONSTANTS
            ]
        };
        ret
    }
}

struct ResourceAccess {
    name: FsrResourceName,
    access_type: AccessType,
    desc: Option<wgpu::TextureViewDescriptor<'static>>,
}

#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Constants {
    fsr: FsrConstants,
    generate_auto_reactive: GenerateAutoReactiveConstants,
    rcas: RcasConstants,
    generate_reactive: GenerateReactiveConstants,
    spd: SpdConstants,
}

#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct FsrConstants {
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

#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GenerateAutoReactiveConstants {
    tc_threshold: f32, // 0.1 is a good starting value, lower will result in more TC pixels
    tc_scale: f32,
    reactive_scale: f32,
    reactive_max: f32,
}

#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct RcasConstants {
    rcas_config: [u32; 4],
}

#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GenerateReactiveConstants {
    gen_reactive_scale: f32,
    gen_reactive_threshold: f32,
    gen_reactive_binary_value: f32,
    gen_reactive_flags: u32,
}

#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct SpdConstants {
    mips: u32,
    num_work_groups: u32,
    work_group_offset: [u32; 2],
    render_size: [u32; 2],
}

enum AccessType {
    SRV,
    UAV,
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

#[derive(Debug, Clone, Copy)]
enum FsrResourceName {
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
}

impl FsrResourceName {
    fn format(&self) -> wgpu::TextureFormat {
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
            FsrResourceName::Luma => wgpu::TextureFormat::R16Float,
            FsrResourceName::LumaInstability | FsrResourceName::FarthestDepth => {
                wgpu::TextureFormat::R16Float
            }
            FsrResourceName::ShadingChange => wgpu::TextureFormat::R8Unorm,
            FsrResourceName::NewLocks => wgpu::TextureFormat::R8Uint,
            FsrResourceName::InternalUpscaled => wgpu::TextureFormat::Rgba16Float,
            FsrResourceName::SpdMips => wgpu::TextureFormat::Rg16Float,
            FsrResourceName::FarthestDepthMip1 => wgpu::TextureFormat::R16Float,
            FsrResourceName::LumaHistory => wgpu::TextureFormat::Rgba16Float,
            FsrResourceName::SpdAtomicCount => {
                panic!("SpdAtomicCount is a buffer")
            }
            FsrResourceName::DilatedReactiveMasks => wgpu::TextureFormat::Rgba8Unorm,
            FsrResourceName::Lanczos2Lut => wgpu::TextureFormat::R16Snorm,
            FsrResourceName::DefaultReactivityMask => wgpu::TextureFormat::R8Unorm,
            FsrResourceName::DefaultExposure => wgpu::TextureFormat::Rg32Float,
            FsrResourceName::FrameInfo => wgpu::TextureFormat::Rgba32Float,
        }
    }

    fn to_bgl_entry(&self, binding: u32, access_type: AccessType) -> wgpu::BindGroupLayoutEntry {
        match (self, access_type) {
            (
                FsrResourceName::InputColor
                | FsrResourceName::InputDepth
                | FsrResourceName::InputMotionVectors
                | FsrResourceName::InputExposure
                | FsrResourceName::InputReactiveMask
                | FsrResourceName::InputTransparencyAndComposition,
                AccessType::UAV,
            ) => {
                panic!("Input resources cannot be UAVs")
            }
            (FsrResourceName::Constants, AccessType::UAV) => {
                panic!("Constants cannot be UAVs")
            }
            (FsrResourceName::Constants, AccessType::SRV) => wgpu::BindGroupLayoutEntry {
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
                AccessType::SRV,
            ) => wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            (
                FsrResourceName::OutputReconstructedPreviousDepth | FsrResourceName::SpdAtomicCount,
                AccessType::UAV,
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

            (_, AccessType::UAV) => wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: self.format(),
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            (_, AccessType::SRV) => wgpu::BindGroupLayoutEntry {
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

struct FsrResources {
    constant_buffer: wgpu::Buffer,

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
    spd_atomic_counter: wgpu::Buffer,
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

        let constant_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FSR3 Constants"),
            size: std::mem::size_of::<Constants>() as u64,
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
        }
    }

    fn to_view<'a>(
        &'a self,
        dispatch: &FsrDispatchInfo,
        name: FsrResourceName,
        index: u8,
        descriptor: Option<wgpu::TextureViewDescriptor>,
    ) -> ViewOrBuffer {
        let descriptor = descriptor.unwrap_or_default();

        match name {
            FsrResourceName::InputColor => {
                ViewOrBuffer::View(dispatch.color.create_view(&descriptor))
            }
            FsrResourceName::InputDepth => {
                ViewOrBuffer::View(dispatch.depth.create_view(&descriptor))
            }
            FsrResourceName::InputMotionVectors => {
                ViewOrBuffer::View(dispatch.motion_vectors.create_view(&descriptor))
            }
            FsrResourceName::InputExposure => {
                if let Some(exposure) = &dispatch.exposure {
                    ViewOrBuffer::View(exposure.create_view(&descriptor))
                } else {
                    ViewOrBuffer::View(self.default_exposure.create_view(&descriptor))
                }
            }
            FsrResourceName::InputReactiveMask => {
                if let Some(reactive_mask) = &dispatch.reactive_mask {
                    ViewOrBuffer::View(reactive_mask.create_view(&descriptor))
                } else {
                    ViewOrBuffer::View(self.default_reactivity_mask.create_view(&descriptor))
                }
            }
            FsrResourceName::InputTransparencyAndComposition => {
                if let Some(transparency_and_composition) = &dispatch.transparency_and_composition {
                    ViewOrBuffer::View(transparency_and_composition.create_view(&descriptor))
                } else {
                    // Note: We use the default reactivity mask here.
                    ViewOrBuffer::View(self.default_reactivity_mask.create_view(&descriptor))
                }
            }
            FsrResourceName::OutputColor => {
                ViewOrBuffer::View(dispatch.output.create_view(&descriptor))
            }
            FsrResourceName::OutputDilatedDepth => {
                ViewOrBuffer::View(dispatch.dilated_depth.create_view(&descriptor))
            }
            FsrResourceName::OutputDilatedMotionVectors => {
                ViewOrBuffer::View(dispatch.dilated_motion_vectors.create_view(&descriptor))
            }
            FsrResourceName::OutputReconstructedPreviousDepth => {
                ViewOrBuffer::Buffer(dispatch.reconstructed_previous_depth.clone())
            }

            FsrResourceName::Constants => ViewOrBuffer::Buffer(self.constant_buffer.clone()),

            FsrResourceName::Accumulation => {
                if index == 0 {
                    ViewOrBuffer::View(self.accumulation_1.create_view(&descriptor))
                } else {
                    ViewOrBuffer::View(self.accumulation_2.create_view(&descriptor))
                }
            }
            FsrResourceName::Luma => {
                if index == 0 {
                    ViewOrBuffer::View(self.luma_1.create_view(&descriptor))
                } else {
                    ViewOrBuffer::View(self.luma_2.create_view(&descriptor))
                }
            }
            FsrResourceName::LumaInstability | FsrResourceName::FarthestDepth => {
                ViewOrBuffer::View(self.intermediate_fp16x1.create_view(&descriptor))
            }
            FsrResourceName::ShadingChange => {
                ViewOrBuffer::View(self.shading_change.create_view(&descriptor))
            }
            FsrResourceName::NewLocks => {
                ViewOrBuffer::View(self.new_locks.create_view(&descriptor))
            }
            FsrResourceName::InternalUpscaled => {
                if index == 0 {
                    ViewOrBuffer::View(self.internal_upscaled_1.create_view(&descriptor))
                } else {
                    ViewOrBuffer::View(self.internal_upscaled_2.create_view(&descriptor))
                }
            }
            FsrResourceName::SpdMips => ViewOrBuffer::View(self.spd_mips.create_view(&descriptor)),
            FsrResourceName::FarthestDepthMip1 => {
                ViewOrBuffer::View(self.farthest_depth_mip1.create_view(&descriptor))
            }
            FsrResourceName::LumaHistory => {
                if index == 0 {
                    ViewOrBuffer::View(self.luma_history1.create_view(&descriptor))
                } else {
                    ViewOrBuffer::View(self.luma_history2.create_view(&descriptor))
                }
            }
            FsrResourceName::SpdAtomicCount => {
                ViewOrBuffer::Buffer(self.spd_atomic_counter.clone())
            }
            FsrResourceName::DilatedReactiveMasks => {
                ViewOrBuffer::View(self.dilated_reactive_masks.create_view(&descriptor))
            }
            FsrResourceName::Lanczos2Lut => {
                ViewOrBuffer::View(self.lanczos2_lut.create_view(&descriptor))
            }
            FsrResourceName::DefaultReactivityMask => {
                ViewOrBuffer::View(self.default_reactivity_mask.create_view(&descriptor))
            }
            FsrResourceName::DefaultExposure => {
                ViewOrBuffer::View(self.default_exposure.create_view(&descriptor))
            }
            FsrResourceName::FrameInfo => {
                ViewOrBuffer::View(self.frame_info.create_view(&descriptor))
            }
        }
    }
}

enum ViewOrBuffer {
    View(wgpu::TextureView),
    Buffer(wgpu::Buffer),
}

impl<'a> From<&'a ViewOrBuffer> for wgpu::BindingResource<'a> {
    fn from(value: &'a ViewOrBuffer) -> Self {
        match value {
            ViewOrBuffer::View(v) => wgpu::BindingResource::TextureView(v),
            ViewOrBuffer::Buffer(b) => b.as_entire_binding(),
        }
    }
}
