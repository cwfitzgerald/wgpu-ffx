#![allow(dead_code)]

mod clear_buffer;
mod constants;
mod jitter;
mod lanczos2;
mod pass;
mod resources;

use std::mem;

use wgpu_ffx_shaders_spv::fsr3upscaler::*;

use crate::{
    constants::FsrConstants,
    resources::{FsrResourceName, OwnedBindingResource},
};

pub struct FsrContext {
    device: wgpu::Device,

    constants: constants::Constants,
    resources: resources::FsrResources,

    pass_prepare_inputs: pass::FsrPass,
    pass_prepare_reactivity: pass::FsrPass,
    pass_shading_change: pass::FsrPass,
    pass_accumulate: pass::FsrPass,
    pass_rcas: pass::FsrPass,
    pass_luma_pyramid: pass::FsrPass,
    // pass_generate_reactive: pass::FsrPass,
    pass_shading_change_pyramid: pass::FsrPass,
    pass_luma_instability: pass::FsrPass,

    first_execution: bool,
    previous_jitter_offset: [f32; 2],
    pre_exposure: f32,
    previous_frame_pre_exposure: f32,
    frame_kind: FrameKind,

    flags: FsrContextFlags,
}

impl FsrContext {
    pub fn new(info: FsrContextInfo) -> Self {
        let fsr_constants = constants::FsrConstants {
            max_render_size: info.max_render_size,
            max_upscale_size: info.max_upscale_size,
            velocity_factor: 1.0,
            reactiveness_scale: 1.0,
            shading_change_scale: 1.0,
            accumulation_added_per_frame: 1.0 / 3.0,
            min_disocclusion_accumulation: -1.0 / 3.0,
            ..Default::default()
        };

        let shaders = wgpu_ffx_shaders_spv::fsr3upscaler::choose_shaders(
            Fsr3upscalerOptionApplySharpening::Off,
            Fsr3upscalerOptionHdrColorInput::On,
            Fsr3upscalerOptionInvertedDepth::On,
            Fsr3upscalerOptionJitteredMotionVectors::On,
            Fsr3upscalerOptionLowResolutionMotionVectors::On,
            Fsr3upscalerOptionReprojectUseLanczosType::Off,
            Half::Off,
            Wave64::Off,
        );

        let pass_prepare_inputs = pass::FsrPass::new(
            &info.device,
            pass::FsrPassKind::PrepareInputs,
            info.flags,
            &shaders,
        );
        let pass_prepare_reactivity = pass::FsrPass::new(
            &info.device,
            pass::FsrPassKind::PrepareReactivity,
            info.flags,
            &shaders,
        );
        let pass_shading_change = pass::FsrPass::new(
            &info.device,
            pass::FsrPassKind::ShadingChange,
            info.flags,
            &shaders,
        );
        let pass_accumulate = pass::FsrPass::new(
            &info.device,
            pass::FsrPassKind::Accumulate,
            info.flags,
            &shaders,
        );
        let pass_rcas =
            pass::FsrPass::new(&info.device, pass::FsrPassKind::Rcas, info.flags, &shaders);
        let pass_luma_pyramid = pass::FsrPass::new(
            &info.device,
            pass::FsrPassKind::LumaPyramid,
            info.flags,
            &shaders,
        );
        // let pass_generate_reactive = pass::FsrPass::new(
        //     &info.device,
        //     pass::FsrPassKind::GenerateReactive,
        //     info.flags,
        //     &shaders,
        // );
        let pass_shading_change_pyramid = pass::FsrPass::new(
            &info.device,
            pass::FsrPassKind::ShadingChangePyramid,
            info.flags,
            &shaders,
        );
        let pass_luma_instability = pass::FsrPass::new(
            &info.device,
            pass::FsrPassKind::LumaInstability,
            info.flags,
            &shaders,
        );

        Self {
            constants: constants::Constants {
                fsr: fsr_constants,
                ..Default::default()
            },
            resources: resources::FsrResources::new(
                &info.device,
                &info.queue,
                info.max_render_size,
                info.max_upscale_size,
            ),
            device: info.device,

            pass_prepare_inputs,
            pass_prepare_reactivity,
            pass_shading_change,
            pass_accumulate,
            pass_rcas,
            pass_luma_pyramid,
            // pass_generate_reactive,
            pass_shading_change_pyramid,
            pass_luma_instability,

            first_execution: true,
            previous_jitter_offset: [0.0, 0.0],
            pre_exposure: 1.0,
            previous_frame_pre_exposure: 1.0,
            frame_kind: FrameKind::Even,

            flags: info.flags,
        }
    }

    fn setup_device_depth_to_view_space_depth_params(
        flags: FsrContextFlags,
        fsrc: &mut FsrConstants,
        info: &FsrDispatchInfo,
    ) {
        let b_inverted = flags.contains(FsrContextFlags::DEPTH_INVERTED);
        let b_infinite = flags.contains(FsrContextFlags::DEPTH_INFINITE);

        let mut f_min = f32::min(info.camera_near, info.camera_far);
        let mut f_max = f32::max(info.camera_near, info.camera_far);

        if b_inverted {
            mem::swap(&mut f_min, &mut f_max);
        }

        // a 0 0 0   x
        // 0 b 0 0   y
        // 0 0 c d   z
        // 0 0 e 0   1

        let f_q = f_max / (f_min - f_max);
        let d = -1.0f32;

        let matrix_elem_c = [
            [
                f_q,                    // non reversed, non infinite
                -1.0f32 - f32::EPSILON, // non reversed, infinite
            ],
            [
                f_q,                   // reversed, non infinite
                0.0f32 + f32::EPSILON, // reversed, infinite
            ],
        ];

        let matrix_elem_e = [
            [
                f_q * f_min,           // non reversed, non infinite
                -f_min - f32::EPSILON, // non reversed, infinite
            ],
            [
                f_q * f_min, // reversed, non infinite
                f_max,       // reversed, infinite
            ],
        ];

        fsrc.device_to_view_depth[0] = d * matrix_elem_c[b_inverted as usize][b_infinite as usize];
        fsrc.device_to_view_depth[1] = matrix_elem_e[b_inverted as usize][b_infinite as usize];

        let aspect = info.render_size[0] as f32 / info.render_size[1] as f32;
        let cot_half_fov_y =
            f32::cos(0.5f32 * info.camera_fov_y) / f32::sin(0.5f32 * info.camera_fov_y);
        let a = cot_half_fov_y / aspect;
        let b = cot_half_fov_y;

        fsrc.device_to_view_depth[2] = f32::recip(a);
        fsrc.device_to_view_depth[3] = f32::recip(b);
    }

    fn dispatch(&mut self, info: &mut FsrDispatchInfo) {
        let reset_accumulation = info.reset_history || self.first_execution;
        self.first_execution = false;

        let fsrc = &mut self.constants.fsr;

        fsrc.previous_frame_jitter_offset = fsrc.jitter_offset;
        fsrc.previous_frame_upscale_size = fsrc.upscale_size;
        fsrc.previous_frame_render_size = fsrc.render_size;

        fsrc.jitter_offset = info.jitter_offset;
        fsrc.upscale_size = info.upscale_size;
        fsrc.render_size = info.render_size;
        fsrc.downscale_factor = [
            info.render_size[0] as f32 / info.upscale_size[0] as f32,
            info.render_size[1] as f32 / info.upscale_size[1] as f32,
        ];

        // compute the horizontal FOV for the shader from the vertical one.
        let aspect_ratio = info.render_size[0] as f32 / info.render_size[1] as f32;
        let camera_angle_horizontal =
            f32::atan(f32::tan(info.camera_fov_y * 0.5) * aspect_ratio) * 2.0;
        fsrc.tan_half_fov = f32::tan(camera_angle_horizontal * 0.5);
        fsrc.view_space_to_meters_factor = info.view_space_to_meters_factor;

        Self::setup_device_depth_to_view_space_depth_params(self.flags, fsrc, info);

        // calculate pre-exposure relevant factors
        self.previous_frame_pre_exposure = self.pre_exposure;
        self.pre_exposure = info.pre_exposure;

        if self.previous_frame_pre_exposure > 0.0 {
            fsrc.delta_pre_exposure = self.pre_exposure / self.previous_frame_pre_exposure;
        } else {
            fsrc.delta_pre_exposure = 1.0;
        }

        // motion vector scale
        let motion_vectors_target_size = if self
            .flags
            .contains(FsrContextFlags::DISPLAY_RESOLUTION_MOTION_VECTORS)
        {
            info.upscale_size
        } else {
            info.render_size
        };
        fsrc.motion_vector_scale = std::array::from_fn(|i| {
            info.motion_vector_scale[i] * motion_vectors_target_size[i] as f32
        });

        // compute jitter cancellation
        if self
            .flags
            .contains(FsrContextFlags::MOTION_VECTORS_JITTER_CANCELLATION)
        {
            fsrc.motion_vector_jitter_cancellation = std::array::from_fn(|i| {
                (self.previous_jitter_offset[i] - info.jitter_offset[i])
                    / motion_vectors_target_size[i] as f32
            });
        }

        let jitter_phase_count =
            jitter::get_jitter_phase_count(info.render_size[0] as i32, info.upscale_size[0] as i32);

        if reset_accumulation || fsrc.jitter_phase_count == 0.0 {
            fsrc.jitter_phase_count = jitter_phase_count as f32;
        } else {
            let jitter_phase_count_delta = jitter_phase_count - fsrc.jitter_phase_count as i32;
            match jitter_phase_count_delta.cmp(&0) {
                std::cmp::Ordering::Greater => {
                    fsrc.jitter_phase_count += 1.0;
                }
                std::cmp::Ordering::Less => {
                    fsrc.jitter_phase_count -= 1.0;
                }
                std::cmp::Ordering::Equal => {}
            }
        }

        // convert delta time to seconds and clamp to [0, 1].
        fsrc.delta_time = f32::clamp(info.frame_time_delta * 0.001, 0.0, 1.0);

        if reset_accumulation {
            fsrc.frame_index = 0.0;
        } else {
            fsrc.frame_index += 1.0;
        }

        let thread_group_work_region_dim = 8;
        let workgroups_src_x = fsrc.render_size[0].div_ceil(thread_group_work_region_dim);
        let workgroups_src_y = fsrc.render_size[1].div_ceil(thread_group_work_region_dim);
        let workgroups_dst_x = fsrc.upscale_size[0].div_ceil(thread_group_work_region_dim);
        let workgroups_dst_y = fsrc.upscale_size[1].div_ceil(thread_group_work_region_dim);
        let workgroups_shading_change_x =
            (fsrc.render_size[0] / 2).div_ceil(thread_group_work_region_dim);
        let workgroups_shading_change_y =
            (fsrc.render_size[1] / 2).div_ceil(thread_group_work_region_dim);

        // Clear reconstructed depth for max depth store.
        if reset_accumulation {
            let zeroed_resources = [FsrResourceName::Accumulation, FsrResourceName::SpdMips];
            for name in zeroed_resources {
                let OwnedBindingResource::View(accumulation_texture) =
                    self.resources.to_view(&info, name, self.frame_kind, None)
                else {
                    unreachable!()
                };

                info.encoder.clear_texture(
                    accumulation_texture.texture(),
                    &wgpu::ImageSubresourceRange::default(),
                );
            }

            let clear_values_exposure = [-1.0f32, 1.0, 0.0, 0.0];
            info.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.resources.frame_info,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&clear_values_exposure),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(16),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.frame_kind.advance();
    }
}

pub struct FsrContextInfo {
    /// The wgpu device to use for GPU operations.
    pub device: wgpu::Device,
    /// The wgpu queue to use for submitting uploads.
    pub queue: wgpu::Queue,
    /// The maximum resolution in pixels that the application will render will at.
    pub max_render_size: [u32; 2],
    /// The maximum resolution in pixels that FSR will upscale to.
    pub max_upscale_size: [u32; 2],
    /// Configuration options for the FSR context.
    pub flags: FsrContextFlags,
}

bitflags::bitflags! {
    /// Configuration options for the FSR context.
    #[derive(Debug, Clone, Copy)]
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
    /// The wgpu queue to use for submitting uploads.
    pub queue: wgpu::Queue,
    /// The wgpu CommandEncoder to record FSR3 rendering commands into.
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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

#[test]
fn fsr_smoke() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("Failed to find an appropriate adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: adapter.features(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
        label: None,
    }))
    .expect("Failed to create device");

    let _fsr_context = FsrContext::new(FsrContextInfo {
        device,
        queue,
        max_render_size: [1920, 1080],
        max_upscale_size: [3840, 2160],
        flags: FsrContextFlags::empty(),
    });
}
