use crate::{FsrContextFlags, resources::AccessType, resources::FsrResourceName};

use wgpu_ffx_shaders_spv::fsr3upscaler::Shaders;

pub(crate) enum FsrPass {
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
    pub fn label(&self) -> &'static str {
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

    pub fn shader(&self, shaders: &Shaders) -> &'static [u8] {
        match self {
            FsrPass::PrepareInputs => &shaders.prepare_inputs,
            FsrPass::LumaPyramid => &shaders.luma_pyramid,
            FsrPass::ShadingChangePyramid => &shaders.shading_change_pyramid,
            FsrPass::ShadingChange => &shaders.shading_change,
            FsrPass::PrepareReactivity => &shaders.prepare_reactivity,
            FsrPass::LumaInstability => &shaders.luma_instability,
            FsrPass::Accumulate => &shaders.accumulate,
            FsrPass::AccumulateSharpen => todo!(),
            FsrPass::Rcas => &shaders.rcas,
            FsrPass::DebugView => &shaders.debug_view,
            FsrPass::GenerateReactive => todo!(),
        }
    }

    pub fn resources(&self, flags: FsrContextFlags) -> Vec<ResourceAccess> {
        use crate::resources::AccessType::*;
        use crate::resources::FsrResourceName::*;

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
                ResourceAccess { name: FsrResourceName::Constants, access_type: AccessType::SRV, desc: None },
            ]
        };
        ret
    }
}

pub(crate) struct ResourceAccess {
    name: FsrResourceName,
    access_type: AccessType,
    desc: Option<wgpu::TextureViewDescriptor<'static>>,
}
