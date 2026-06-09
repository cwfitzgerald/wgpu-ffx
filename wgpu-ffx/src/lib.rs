//! AMD FidelityFX Super Resolution 3 (FSR3) upscaler for wgpu.
//!
//! This crate provides a Rust implementation of AMD's FSR3 temporal upscaler,
//! allowing applications to render at a lower resolution and reconstruct
//! high-quality output at display resolution using temporal accumulation.
//!
//! # Usage
//!
//! 1. Create an [`FsrContext`] with [`FsrContextInfo`] describing your device
//!    and feature flags. This compiles shader pipelines but allocates no textures.
//! 2. Create an [`FsrView`] via [`FsrContext::create_view`] with the maximum
//!    render and upscale resolutions. This allocates internal GPU resources.
//! 3. Each frame, fill an [`FsrDispatchInfo`] with the current frame's textures,
//!    camera parameters, and jitter offset, then call [`FsrContext::dispatch`].
//!
//! The [`FsrView`] can be resized independently of the [`FsrContext`], avoiding
//! expensive pipeline recompilation when resolution changes.
//!
//! Jitter must be applied to the projection matrix and the same offsets passed
//! to [`FsrDispatchInfo::jitter_offset`]. Use [`get_jitter_phase_count`] and
//! [`get_jitter_offset`] to compute Halton-sequence jitter values.
//!
//! Dispatch returns [`FsrDispatchError`] if any parameters fail validation.

#![allow(dead_code)] // TODO: remove once GenerateReactive and DebugView are wired up

mod clear_buffer;
mod constants;
mod format_profile;
mod jitter;
mod lanczos2;
mod pass;
mod rcas;
mod resources;
mod spd;
mod spd_core;
mod validation;

use std::mem;

use wgpu::util::DeviceExt as _;
use wgpu_ffx_shaders_spv::fsr3upscaler::*;

use crate::{
    constants::FsrConstants,
    pass::ResourceAccess,
    resources::{FsrResourceName, OwnedBindingResource},
};

// Re-export validation types
pub use format_profile::{FormatProfile, FsrFormats};
pub use jitter::*;
pub use validation::FsrDispatchError;

/// The main FSR3 upscaler context.
///
/// Holds compiled GPU pipelines needed for temporal upscaling. Create one per
/// set of feature flags with [`FsrContext::new`], then create [`FsrView`]s
/// for each resolution target via [`FsrContext::create_view`].
///
/// Pipelines are expensive to compile but views are cheap to create and resize.
pub struct FsrContext {
    device: wgpu::Device,

    buffer_clearer: clear_buffer::BufferClearer,

    pass_prepare_inputs: pass::FsrPass,
    pass_prepare_reactivity: pass::FsrPass,
    pass_shading_change: pass::FsrPass,
    pass_accumulate: pass::FsrPass,
    pass_accumulate_sharpen: pass::FsrPass,
    pass_rcas: pass::FsrPass,
    pass_luma_pyramid: spd_core::SpdPyramid,
    // pass_generate_reactive: pass::FsrPass,
    pass_shading_change_pyramid: spd_core::SpdPyramid,
    pass_luma_instability: pass::FsrPass,
    pass_debug_view: pass::FsrPass,

    flags: FsrContextFlags,
    format_profile: FormatProfile,
}

/// Per-resolution, per-camera state for FSR3 upscaling.
///
/// Owns all internal GPU textures and temporal accumulation state. Create via
/// [`FsrContext::create_view`] and pass to [`FsrContext::dispatch`] each frame.
///
/// Call [`FsrView::resize`] to change the maximum render or upscale resolution
/// without recreating the parent [`FsrContext`] (avoiding pipeline recompilation).
pub struct FsrView {
    device: wgpu::Device,
    format_profile: FormatProfile,

    constants: constants::Constants,
    resources: resources::FsrResources,

    max_render_size: [u32; 2],
    max_upscale_size: [u32; 2],

    first_execution: bool,
    previous_jitter_offset: [f32; 2],
    pre_exposure: f32,
    previous_frame_pre_exposure: f32,
    frame_kind: FrameKind,
}

impl FsrContext {
    /// Create a new FSR3 upscaler context.
    ///
    /// Compiles all shader pipelines based on the feature flags specified in
    /// `info`. No GPU textures are allocated — use [`FsrContext::create_view`]
    /// to create per-resolution state.
    pub fn new(info: FsrContextInfo) -> Self {
        let format_profile = info
            .format_profile
            .unwrap_or_else(|| FormatProfile::from_device(&info.device));

        let buffer_clearer = clear_buffer::BufferClearer::new(&info.device);

        let flags = info.flags;
        let shaders = wgpu_ffx_shaders_spv::fsr3upscaler::choose_shaders(
            match format_profile {
                FormatProfile::Core => WgpuFormatProfile::Core,
                FormatProfile::Tier2 => WgpuFormatProfile::Tier2,
                FormatProfile::Native => WgpuFormatProfile::Native,
            },
            Half::Off,
            Wave64::Off,
            // LUT vs reference lanczos. LUT is used on GPUs with 32-64 wave lane range;
            // since we don't query this from wgpu yet, default to reference (Off).
            Fsr3upscalerOptionReprojectUseLanczosType::Off,
            if flags.contains(FsrContextFlags::HIGH_DYNAMIC_RANGE) {
                Fsr3upscalerOptionHdrColorInput::On
            } else {
                Fsr3upscalerOptionHdrColorInput::Off
            },
            if flags.contains(FsrContextFlags::DISPLAY_RESOLUTION_MOTION_VECTORS) {
                Fsr3upscalerOptionLowResolutionMotionVectors::Off
            } else {
                Fsr3upscalerOptionLowResolutionMotionVectors::On
            },
            if flags.contains(FsrContextFlags::MOTION_VECTORS_JITTER_CANCELLATION) {
                Fsr3upscalerOptionJitteredMotionVectors::On
            } else {
                Fsr3upscalerOptionJitteredMotionVectors::Off
            },
            if flags.contains(FsrContextFlags::DEPTH_INVERTED) {
                Fsr3upscalerOptionInvertedDepth::On
            } else {
                Fsr3upscalerOptionInvertedDepth::Off
            },
        );

        let pass_prepare_inputs = pass::FsrPass::new(
            &info.device,
            pass::FsrPassKind::PrepareInputs,
            info.flags,
            format_profile,
            &shaders,
        );
        let pass_prepare_reactivity = pass::FsrPass::new(
            &info.device,
            pass::FsrPassKind::PrepareReactivity,
            info.flags,
            format_profile,
            &shaders,
        );
        let pass_shading_change = pass::FsrPass::new(
            &info.device,
            pass::FsrPassKind::ShadingChange,
            info.flags,
            format_profile,
            &shaders,
        );
        let pass_accumulate = pass::FsrPass::new(
            &info.device,
            pass::FsrPassKind::Accumulate,
            info.flags,
            format_profile,
            &shaders,
        );
        let pass_accumulate_sharpen = pass::FsrPass::new(
            &info.device,
            pass::FsrPassKind::AccumulateSharpen,
            info.flags,
            format_profile,
            &shaders,
        );
        let pass_rcas = pass::FsrPass::new(
            &info.device,
            pass::FsrPassKind::Rcas,
            info.flags,
            format_profile,
            &shaders,
        );
        let pass_luma_pyramid = spd_core::SpdPyramid::new(
            &info.device,
            spd_core::SpdPyramidKind::Luma,
            info.flags,
            format_profile,
            &shaders,
        );
        // let pass_generate_reactive = pass::FsrPass::new(
        //     &info.device,
        //     pass::FsrPassKind::GenerateReactive,
        //     info.flags,
        //     &shaders,
        // );
        let pass_shading_change_pyramid = spd_core::SpdPyramid::new(
            &info.device,
            spd_core::SpdPyramidKind::ShadingChange,
            info.flags,
            format_profile,
            &shaders,
        );
        let pass_luma_instability = pass::FsrPass::new(
            &info.device,
            pass::FsrPassKind::LumaInstability,
            info.flags,
            format_profile,
            &shaders,
        );
        let pass_debug_view = pass::FsrPass::new(
            &info.device,
            pass::FsrPassKind::DebugView,
            info.flags,
            format_profile,
            &shaders,
        );

        Self {
            device: info.device,

            buffer_clearer,

            pass_prepare_inputs,
            pass_prepare_reactivity,
            pass_shading_change,
            pass_accumulate,
            pass_accumulate_sharpen,
            pass_rcas,
            pass_luma_pyramid,
            // pass_generate_reactive,
            pass_shading_change_pyramid,
            pass_luma_instability,
            pass_debug_view,

            flags: info.flags,
            format_profile,
        }
    }

    /// The [`FormatProfile`] this context was created with (either the value
    /// forced via [`FsrContextInfo::format_profile`] or the one auto-detected
    /// from the device).
    pub fn format_profile(&self) -> FormatProfile {
        self.format_profile
    }

    /// The formats the caller must use for the textures it provides to
    /// [`FsrContext::dispatch`]. Equivalent to `self.format_profile().formats()`.
    pub fn formats(&self) -> FsrFormats {
        self.format_profile.formats()
    }

    /// Allocate internal GPU resources for a given maximum resolution pair.
    ///
    /// The returned [`FsrView`] holds all textures and temporal accumulation
    /// state. Multiple views can be created from the same context for
    /// multi-camera or split-screen rendering.
    ///
    /// # Panics
    ///
    /// Panics if any dimension of `max_render_size` is less than 2, or any
    /// dimension of `max_upscale_size` is 0.
    pub fn create_view(
        &self,
        queue: &wgpu::Queue,
        max_render_size: [u32; 2],
        max_upscale_size: [u32; 2],
    ) -> FsrView {
        FsrView::validate_sizes(max_render_size, max_upscale_size);
        FsrView::new(
            self.device.clone(),
            queue,
            self.format_profile,
            max_render_size,
            max_upscale_size,
        )
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

    /// Validate dispatch parameters for correctness.
    fn check(&self, view: &FsrView, info: &FsrDispatchInfo) -> Result<(), FsrDispatchError> {
        validation::check_dispatch(
            info,
            self.flags,
            self.format_profile.formats(),
            view.max_render_size,
            view.max_upscale_size,
        )
    }

    /// Record the FSR3 upscaling compute passes into the provided command encoder.
    ///
    /// Validates `info` parameters, updates `view`'s internal state, and records
    /// all compute passes into `encoder`. The encoder is **not** submitted — the
    /// caller is responsible for finishing and submitting it.
    pub fn dispatch(
        &self,
        view: &mut FsrView,
        encoder: &mut wgpu::CommandEncoder,
        info: &FsrDispatchInfo,
    ) -> Result<(), FsrDispatchError> {
        self.check(view, info)?;

        let reset_accumulation = info.reset_history || view.first_execution;
        view.first_execution = false;

        let fsrc = &mut view.constants.fsr;

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
        view.previous_frame_pre_exposure = view.pre_exposure;
        view.pre_exposure = info.pre_exposure;

        if view.previous_frame_pre_exposure > 0.0 {
            fsrc.delta_pre_exposure = view.pre_exposure / view.previous_frame_pre_exposure;
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
            info.motion_vector_scale[i] / motion_vectors_target_size[i] as f32
        });

        // compute jitter cancellation
        if self
            .flags
            .contains(FsrContextFlags::MOTION_VECTORS_JITTER_CANCELLATION)
        {
            fsrc.motion_vector_jitter_cancellation = std::array::from_fn(|i| {
                (view.previous_jitter_offset[i] - info.jitter_offset[i])
                    / motion_vectors_target_size[i] as f32
            });

            view.previous_jitter_offset = info.jitter_offset;
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
        // The SPD pyramids compute their own dispatch dimensions from the render
        // size (the Core profile issues a per-mip chain rather than one dispatch).

        let error_scope_guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        // Clear reconstructed depth for max depth store.
        if reset_accumulation {
            let zeroed_resources = [
                // We always clear the previous accumulation view here
                // as we are clearing what we're _reading_ from.
                ResourceAccess {
                    name: FsrResourceName::AccumulationPrevious,
                    access_type: resources::AccessType::Srv,
                    desc: None,
                },
                // We also need to clear the SPD mips, this doesn't
                // change based on frame kind.
                ResourceAccess {
                    name: FsrResourceName::SpdMips,
                    access_type: resources::AccessType::Srv,
                    desc: None,
                },
            ];
            for access in zeroed_resources {
                let OwnedBindingResource::View(accumulation_texture) =
                    view.resources.to_view(info, access, view.frame_kind)
                else {
                    unreachable!()
                };

                encoder.clear_texture(
                    accumulation_texture.texture(),
                    &wgpu::ImageSubresourceRange::default(),
                );
            }

            let clear_values_frame_info = [-1.0f32, 1.0, 0.0, 0.0];
            let staging_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("frame_info_staging_buffer"),
                        contents: bytemuck::cast_slice(&clear_values_frame_info),
                        usage: wgpu::BufferUsages::COPY_SRC,
                    });

            encoder.copy_buffer_to_buffer(
                &staging_buffer,
                0,
                &view.resources.frame_info,
                0,
                std::mem::size_of_val(&clear_values_frame_info) as wgpu::BufferAddress,
            );
        }

        let clear_value = if self.flags.contains(FsrContextFlags::DEPTH_INVERTED) {
            [0.0_f32; 4]
        } else {
            [1.0_f32; 4]
        };

        self.buffer_clearer.dispatch(
            &self.device,
            &info.reconstructed_previous_depth,
            encoder,
            bytemuck::cast(clear_value),
        );

        self.buffer_clearer.dispatch(
            &self.device,
            &view.resources.spd_atomic_counter,
            encoder,
            [0, 0, 0, 0],
        );

        view.constants.spd = constants::SpdConstants::new(spd::RectInput::new(
            info.render_size[0],
            info.render_size[1],
        ));

        let sharpness_remapped = (-2.0 * info.sharpness) + 2.0;
        view.constants.rcas = rcas::populate_rcas_constants(sharpness_remapped);

        let constants_staging_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("FsrContext::constants_staging_buffer"),
                    contents: bytemuck::bytes_of(&view.constants),
                    usage: wgpu::BufferUsages::COPY_SRC,
                });

        encoder.copy_buffer_to_buffer(
            &constants_staging_buffer,
            0,
            &view.resources.constant_buffer,
            0,
            mem::size_of::<constants::Constants>() as wgpu::BufferAddress,
        );

        encoder.clear_texture(
            &view.resources.spd_mips,
            &wgpu::ImageSubresourceRange::default(),
        );

        if let Some(err) = pollster::block_on(error_scope_guard.pop()) {
            panic!("Error during Clearing: {err}");
        }

        let mut compute_pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FsrContext::dispatch::compute_pass"),
                timestamp_writes: None,
            })
            .forget_lifetime();

        self.pass_prepare_inputs.dispatch(
            &self.device,
            &mut compute_pass,
            &view.resources,
            info,
            self.flags,
            view.frame_kind,
            workgroups_src_x,
            workgroups_src_y,
        );
        self.pass_luma_pyramid.dispatch(
            &self.device,
            &mut compute_pass,
            &view.resources,
            info,
            self.flags,
            view.frame_kind,
            info.render_size,
        );
        self.pass_shading_change_pyramid.dispatch(
            &self.device,
            &mut compute_pass,
            &view.resources,
            info,
            self.flags,
            view.frame_kind,
            info.render_size,
        );
        self.pass_shading_change.dispatch(
            &self.device,
            &mut compute_pass,
            &view.resources,
            info,
            self.flags,
            view.frame_kind,
            workgroups_shading_change_x,
            workgroups_shading_change_y,
        );
        self.pass_prepare_reactivity.dispatch(
            &self.device,
            &mut compute_pass,
            &view.resources,
            info,
            self.flags,
            view.frame_kind,
            workgroups_src_x,
            workgroups_src_y,
        );
        self.pass_luma_instability.dispatch(
            &self.device,
            &mut compute_pass,
            &view.resources,
            info,
            self.flags,
            view.frame_kind,
            workgroups_src_x,
            workgroups_src_y,
        );
        let accumulate_pass = if info.enable_sharpening {
            &self.pass_accumulate_sharpen
        } else {
            &self.pass_accumulate
        };
        accumulate_pass.dispatch(
            &self.device,
            &mut compute_pass,
            &view.resources,
            info,
            self.flags,
            view.frame_kind,
            workgroups_dst_x,
            workgroups_dst_y,
        );
        if info.enable_sharpening {
            let thread_group_work_region_dim_rcas = 16;
            let workgroups_rcas_x =
                info.upscale_size[0].div_ceil(thread_group_work_region_dim_rcas);
            let workgroups_rcas_y =
                info.upscale_size[1].div_ceil(thread_group_work_region_dim_rcas);
            self.pass_rcas.dispatch(
                &self.device,
                &mut compute_pass,
                &view.resources,
                info,
                self.flags,
                view.frame_kind,
                workgroups_rcas_x,
                workgroups_rcas_y,
            );
        }
        if info.flags.contains(FsrDispatchFlags::DRAW_DEBUG_VIEW) {
            self.pass_debug_view.dispatch(
                &self.device,
                &mut compute_pass,
                &view.resources,
                info,
                self.flags,
                view.frame_kind,
                workgroups_dst_x,
                workgroups_dst_y,
            );
        }

        drop(compute_pass);

        view.frame_kind.advance();

        Ok(())
    }
}

impl FsrView {
    fn validate_sizes(max_render_size: [u32; 2], max_upscale_size: [u32; 2]) {
        assert!(
            max_render_size[0] >= 2 && max_render_size[1] >= 2,
            "max_render_size must be at least [2, 2], got [{}, {}] \
             (internal half-resolution textures require dimensions >= 1)",
            max_render_size[0],
            max_render_size[1],
        );
        assert!(
            max_upscale_size[0] >= 1 && max_upscale_size[1] >= 1,
            "max_upscale_size must be at least [1, 1], got [{}, {}]",
            max_upscale_size[0],
            max_upscale_size[1],
        );
    }

    fn new(
        device: wgpu::Device,
        queue: &wgpu::Queue,
        format_profile: FormatProfile,
        max_render_size: [u32; 2],
        max_upscale_size: [u32; 2],
    ) -> Self {
        Self {
            constants: constants::Constants {
                fsr: FsrConstants {
                    max_render_size,
                    max_upscale_size,
                    velocity_factor: 1.0,
                    reactiveness_scale: 1.0,
                    shading_change_scale: 1.0,
                    accumulation_added_per_frame: 1.0 / 3.0,
                    min_disocclusion_accumulation: -1.0 / 3.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            resources: resources::FsrResources::new(
                &device,
                queue,
                format_profile,
                max_render_size,
                max_upscale_size,
            ),
            device,
            format_profile,
            max_render_size,
            max_upscale_size,
            first_execution: true,
            previous_jitter_offset: [0.0, 0.0],
            pre_exposure: 1.0,
            previous_frame_pre_exposure: 1.0,
            frame_kind: FrameKind::Even,
        }
    }

    /// Reallocate internal textures for new maximum resolutions.
    ///
    /// Resets all temporal history — the next dispatch will behave as the
    /// first frame.
    ///
    /// # Panics
    ///
    /// Panics if any dimension of `max_render_size` is less than 2, or any
    /// dimension of `max_upscale_size` is 0.
    pub fn resize(
        &mut self,
        queue: &wgpu::Queue,
        max_render_size: [u32; 2],
        max_upscale_size: [u32; 2],
    ) {
        Self::validate_sizes(max_render_size, max_upscale_size);
        self.resources = resources::FsrResources::new(
            &self.device,
            queue,
            self.format_profile,
            max_render_size,
            max_upscale_size,
        );
        self.max_render_size = max_render_size;
        self.max_upscale_size = max_upscale_size;
        self.constants = constants::Constants {
            fsr: FsrConstants {
                max_render_size,
                max_upscale_size,
                velocity_factor: 1.0,
                reactiveness_scale: 1.0,
                shading_change_scale: 1.0,
                accumulation_added_per_frame: 1.0 / 3.0,
                min_disocclusion_accumulation: -1.0 / 3.0,
                ..Default::default()
            },
            ..Default::default()
        };
        self.first_execution = true;
        self.previous_jitter_offset = [0.0, 0.0];
        self.pre_exposure = 1.0;
        self.previous_frame_pre_exposure = 1.0;
        self.frame_kind = FrameKind::Even;
    }

    /// The maximum render resolution this view was allocated for.
    pub fn max_render_size(&self) -> [u32; 2] {
        self.max_render_size
    }

    /// The maximum upscale resolution this view was allocated for.
    pub fn max_upscale_size(&self) -> [u32; 2] {
        self.max_upscale_size
    }

    /// Estimated GPU memory usage for internal textures and buffers, in bytes.
    ///
    /// This is a lower-bound estimate based on texture formats and dimensions.
    /// Actual driver allocations may be larger due to alignment and padding.
    pub fn estimated_memory_usage(&self) -> u64 {
        let [rw, rh] = self.max_render_size;
        let [uw, uh] = self.max_upscale_size;
        let hrw = rw / 2;
        let hrh = rh / 2;

        let render_pixels = rw as u64 * rh as u64;
        let upscale_pixels = uw as u64 * uh as u64;
        let half_render_pixels = hrw as u64 * hrh as u64;

        let mut total: u64 = 0;

        // Constant buffer
        total += mem::size_of::<constants::Constants>() as u64;

        // accumulation_1 + accumulation_2: R8Unorm at render size
        total += render_pixels * 2;
        // luma_1 + luma_2: R16Float at render size
        total += render_pixels * 2 * 2;
        // intermediate_fp16x1: R16Float at render size
        total += render_pixels * 2;
        // luma_history1 + luma_history2: Rgba16Float at render size
        total += render_pixels * 8 * 2;
        // dilated_reactive_masks: Rgba8Unorm at render size
        total += render_pixels * 4;

        // shading_change: R8Unorm at half render size
        total += half_render_pixels;
        // farthest_depth_mip1: R16Float at half render size
        total += half_render_pixels * 2;

        // spd_mips mip chain at half render size. Native stores Rg16Float
        // (4 bytes/texel); Core and Tier2 widen to Rgba16Float (8 bytes/texel).
        let half_render_extent = wgpu::Extent3d {
            width: hrw,
            height: hrh,
            depth_or_array_layers: 1,
        };
        let spd_bytes_per_texel: u64 = match self.format_profile {
            FormatProfile::Native => 4,
            FormatProfile::Core | FormatProfile::Tier2 => 8,
        };
        let mip_count = half_render_extent.max_mips(wgpu::TextureDimension::D2);
        let mut mip_w = hrw;
        let mut mip_h = hrh;
        for _ in 0..mip_count {
            total += mip_w as u64 * mip_h as u64 * spd_bytes_per_texel;
            mip_w = (mip_w / 2).max(1);
            mip_h = (mip_h / 2).max(1);
        }

        // new_locks: R8Unorm at upscale size
        total += upscale_pixels;
        // internal_upscaled_1 + internal_upscaled_2: Rgba16Float at upscale size
        total += upscale_pixels * 8 * 2;

        // Fixed-size textures
        total += 128 * 2; // lanczos2_lut: 128 entries × R16Snorm (2 bytes)
        total += 1; // default_reactivity_mask: 1×1 R8Unorm
        total += 8; // default_exposure: 1×1 Rg32Float
        total += 16; // frame_info: vec4<f32> storage buffer

        // Buffers
        total += 4; // spd_atomic_counter: 4 bytes
        // spd_level_buffer: one 256-byte-aligned slot per SPD mip level (Core
        // write-only chain). Built from max_render / 2's mip count.
        total += mip_count as u64 * 256;

        total
    }
}

/// Configuration for creating an [`FsrContext`].
pub struct FsrContextInfo {
    /// The wgpu device to use for GPU operations.
    pub device: wgpu::Device,
    /// Configuration options for the FSR context.
    pub flags: FsrContextFlags,
    /// The storage-format profile to drive the device at.
    ///
    /// `None` auto-detects the richest profile the device supports via
    /// [`FormatProfile::from_device`]. `Some` forces a specific profile (the
    /// device must support it) — for example to exercise the [`FormatProfile::Core`]
    /// path on a capable desktop adapter.
    pub format_profile: Option<FormatProfile>,
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

/// Per-frame dispatch parameters for the FSR3 upscaler.
///
/// Contains all textures, buffers, camera parameters, and settings needed
/// for a single upscaling frame. See the field documentation for format and
/// size requirements.
pub struct FsrDispatchInfo {
    /// A Texture containing the color buffer for the current frame (at render resolution).
    pub color: wgpu::Texture,
    /// A Texture containing 32bit depth values for the current frame (at render resolution).
    pub depth: wgpu::Texture,
    /// A Texture containing 2-dimensional motion vectors (at render resolution if [`FsrContextFlags::DISPLAY_RESOLUTION_MOTION_VECTORS`] is not set).
    pub motion_vectors: wgpu::Texture,
    /// An optional Texture containing a 1x1 exposure value.
    pub exposure: Option<wgpu::Texture>,
    /// An optional Texture containing alpha value of reactive objects in the scene.
    pub reactive_mask: Option<wgpu::Texture>,
    /// An optional Texture containing alpha value of special objects in the scene.
    pub transparency_and_composition: Option<wgpu::Texture>,
    /// A Texture with format `R32Float` at render resolution, with `STORAGE_BINDING` and `TEXTURE_BINDING` usage. Used to emit dilated depth and share with following effects.
    pub dilated_depth: wgpu::Texture,
    /// A Texture with format `Rg16Float` at render resolution, with `STORAGE_BINDING` and `TEXTURE_BINDING` usage. Used to emit dilated motion vectors and share with following effects.
    pub dilated_motion_vectors: wgpu::Texture,
    /// A Buffer of size `render_width * render_height * 4` bytes, with `STORAGE` and `COPY_DST` usage. Used to emit reconstructed previous nearest depth and share with following effects.
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

    /// Combination of [`FsrDispatchFlags`].
    pub flags: FsrDispatchFlags,
}

bitflags::bitflags! {
    /// Configuration options for a single FSR dispatch.
    pub struct FsrDispatchFlags: u32 {
        /// A bit indicating that the interpolated output resource will contain debug views with relevant information.
        const DRAW_DEBUG_VIEW = 1 << 0;
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
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("Failed to find an appropriate adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: adapter.features(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::default(),
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        trace: wgpu::Trace::Off,
        label: None,
    }))
    .expect("Failed to create device");

    let fsr_context = FsrContext::new(FsrContextInfo {
        device: device.clone(),
        flags: FsrContextFlags::empty(),
        format_profile: None,
    });

    let _view = fsr_context.create_view(&queue, [1920, 1080], [3840, 2160]);
}

#[test]
fn fsr_dispatch_smoke() {
    // Auto-detected profile (Native on a desktop adapter).
    run_dispatch_smoke(None);
}

#[test]
fn fsr_dispatch_smoke_tier2() {
    // Force the Tier2 profile. Its caller formats match Native; only the
    // internal SPD mips widen to rgba16float (single-pass SPD retained).
    run_dispatch_smoke(Some(FormatProfile::Tier2));
}

#[test]
fn fsr_dispatch_smoke_core() {
    // Force the Core profile (baseline-WebGPU formats, valid on native adapters
    // too). Exercises the write-only SPD mip chain.
    run_dispatch_smoke(Some(FormatProfile::Core));
}

fn run_dispatch_smoke(format_profile: Option<FormatProfile>) {
    // Setup device and queue
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("Failed to find an appropriate adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: adapter.features(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::default(),
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        trace: wgpu::Trace::Off,
        label: None,
    }))
    .expect("Failed to create device");

    // Setup resolution parameters
    let render_size = [640u32, 360u32];
    let upscale_size = [1280u32, 720u32];

    // Create FSR context and view
    let fsr_context = FsrContext::new(FsrContextInfo {
        device: device.clone(),
        flags: FsrContextFlags::HIGH_DYNAMIC_RANGE,
        format_profile,
    });

    // Allocate the caller-provided textures using the profile's required formats.
    let formats = fsr_context.formats();

    let mut view = fsr_context.create_view(&queue, render_size, upscale_size);

    // Create dummy input textures
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_color"),
        size: wgpu::Extent3d {
            width: render_size[0],
            height: render_size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: formats.color,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_depth"),
        size: wgpu::Extent3d {
            width: render_size[0],
            height: render_size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: formats.depth,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let motion_vectors = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_motion_vectors"),
        size: wgpu::Extent3d {
            width: render_size[0],
            height: render_size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: formats.motion_vectors,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Create dummy output textures
    let dilated_depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_dilated_depth"),
        size: wgpu::Extent3d {
            width: render_size[0],
            height: render_size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: formats.dilated_depth,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let dilated_motion_vectors = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_dilated_motion_vectors"),
        size: wgpu::Extent3d {
            width: render_size[0],
            height: render_size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: formats.dilated_motion_vectors,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_output"),
        size: wgpu::Extent3d {
            width: upscale_size[0],
            height: upscale_size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: formats.output,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // Create reconstructed previous depth buffer
    // Buffer size needs to accommodate render resolution
    let buffer_size = (render_size[0] * render_size[1] * 4) as u64; // 4 bytes per pixel for R32
    let reconstructed_previous_depth = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_reconstructed_previous_depth"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    for _ in 0..2 {
        // Create encoder
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });

        // Create dispatch info with valid parameters
        let dispatch_info = FsrDispatchInfo {
            color: color.clone(),
            depth: depth.clone(),
            motion_vectors: motion_vectors.clone(),
            exposure: None,
            reactive_mask: None,
            transparency_and_composition: None,
            dilated_depth: dilated_depth.clone(),
            dilated_motion_vectors: dilated_motion_vectors.clone(),
            reconstructed_previous_depth: reconstructed_previous_depth.clone(),
            output: output.clone(),
            jitter_offset: [0.5, 0.5],
            motion_vector_scale: [1.0, 1.0],
            render_size,
            upscale_size,
            enable_sharpening: false,
            sharpness: 0.5,
            frame_time_delta: 16.6, // ~60fps in milliseconds
            pre_exposure: 1.0,
            reset_history: false,
            camera_near: 0.1,
            camera_far: 1000.0,
            camera_fov_y: std::f32::consts::FRAC_PI_3, // 60 degrees vertical FOV
            view_space_to_meters_factor: 1.0,
            flags: FsrDispatchFlags::empty(),
        };

        // Dispatch FSR - this should complete without errors
        fsr_context
            .dispatch(&mut view, &mut encoder, &dispatch_info)
            .expect("FSR dispatch failed");

        // Submit the command buffer
        queue.submit([encoder.finish()]);
    }
}

/// Run one reset frame with a fixed gradient color input under `format_profile`
/// and read back the SPD-produced frame info `[exposure, logLuma, sceneAvgLuma,
/// _]`. Used to compare the Core write-only SPD pyramid against the Native
/// single-pass one.
#[cfg(test)]
fn run_one_frame_read_frame_info(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format_profile: FormatProfile,
    render_size: [u32; 2],
    upscale_size: [u32; 2],
) -> [f32; 4] {
    let fsr_context = FsrContext::new(FsrContextInfo {
        device: device.clone(),
        flags: FsrContextFlags::empty(),
        format_profile: Some(format_profile),
    });
    let formats = fsr_context.formats();
    let mut view = fsr_context.create_view(queue, render_size, upscale_size);

    let extent = wgpu::Extent3d {
        width: render_size[0],
        height: render_size[1],
        depth_or_array_layers: 1,
    };
    let tex = |label, format, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };

    let color = tex(
        "cmp_color",
        formats.color,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );

    // A deterministic horizontal luminance gradient (the same fp16 data for every
    // profile), so the SPD scene-average reduction has something non-trivial and
    // identical to reduce.
    let mut texels = vec![half::f16::ZERO; (render_size[0] * render_size[1] * 4) as usize];
    for y in 0..render_size[1] {
        for x in 0..render_size[0] {
            let v = half::f16::from_f32(0.1 + 2.9 * (x as f32 / render_size[0] as f32));
            let base = ((y * render_size[0] + x) * 4) as usize;
            texels[base] = v;
            texels[base + 1] = v;
            texels[base + 2] = v;
            texels[base + 3] = half::f16::from_f32(1.0);
        }
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &color,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&texels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(render_size[0] * 4 * 2),
            rows_per_image: Some(render_size[1]),
        },
        extent,
    );

    let depth = tex(
        "cmp_depth",
        formats.depth,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    let motion_vectors = tex(
        "cmp_mv",
        formats.motion_vectors,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    let dilated_depth = tex(
        "cmp_dilated_depth",
        formats.dilated_depth,
        wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let dilated_motion_vectors = tex(
        "cmp_dilated_mv",
        formats.dilated_motion_vectors,
        wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cmp_output"),
        size: wgpu::Extent3d {
            width: upscale_size[0],
            height: upscale_size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: formats.output,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let reconstructed_previous_depth = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cmp_recon_depth"),
        size: (render_size[0] * render_size[1] * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cmp_frame_info_readback"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let dispatch_info = FsrDispatchInfo {
        color,
        depth,
        motion_vectors,
        exposure: None,
        reactive_mask: None,
        transparency_and_composition: None,
        dilated_depth,
        dilated_motion_vectors,
        reconstructed_previous_depth,
        output,
        jitter_offset: [0.0, 0.0],
        motion_vector_scale: [1.0, 1.0],
        render_size,
        upscale_size,
        enable_sharpening: false,
        sharpness: 0.0,
        frame_time_delta: 16.6,
        pre_exposure: 1.0,
        reset_history: true,
        camera_near: 0.1,
        camera_far: 1000.0,
        camera_fov_y: std::f32::consts::FRAC_PI_3,
        view_space_to_meters_factor: 1.0,
        flags: FsrDispatchFlags::empty(),
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cmp_encoder"),
    });
    fsr_context
        .dispatch(&mut view, &mut encoder, &dispatch_info)
        .expect("FSR dispatch failed");
    encoder.copy_buffer_to_buffer(&view.resources.frame_info, 0, &staging, 0, 16);
    queue.submit([encoder.finish()]);

    staging.slice(..).map_async(wgpu::MapMode::Read, |r| {
        r.expect("map frame_info");
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");

    let mapped = staging.slice(..).get_mapped_range();
    let frame_info: [f32; 4] = *bytemuck::from_bytes(&mapped[..16]);
    drop(mapped);
    staging.unmap();
    frame_info
}

/// The Core write-only SPD pyramid should produce a scene-average luma (and thus
/// auto-exposure) very close to the Native single-pass SPD for identical input.
#[test]
fn fsr_spd_core_matches_native() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("Failed to find an appropriate adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: adapter.features(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::default(),
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        trace: wgpu::Trace::Off,
        label: None,
    }))
    .expect("Failed to create device");

    // Power-of-two, multiple-of-64 dimensions: FFX's single-pass SPD reduces
    // these exactly (no tile zero-padding bias), so its scene average is a true
    // mean to compare the Core write-only chain against. render == max avoids any
    // max-vs-actual padding.
    let render_size = [512u32, 512u32];
    let upscale_size = [1024u32, 1024u32];

    let native = run_one_frame_read_frame_info(
        &device,
        &queue,
        FormatProfile::Native,
        render_size,
        upscale_size,
    );
    let core = run_one_frame_read_frame_info(
        &device,
        &queue,
        FormatProfile::Core,
        render_size,
        upscale_size,
    );

    eprintln!("native frame_info = {native:?}");
    eprintln!("core   frame_info = {core:?}");

    // frame_info = [exposure, logLuma, sceneAvgLuma, _].
    let scene_avg_native = native[2];
    let scene_avg_core = core[2];
    let exposure_native = native[0];
    let exposure_core = core[0];

    let rel = |a: f32, b: f32| (a - b).abs() / a.abs().max(b.abs()).max(1e-6);

    assert!(
        rel(scene_avg_native, scene_avg_core) < 0.02,
        "scene-average luma diverged between Native and Core: \
         native={scene_avg_native}, core={scene_avg_core}"
    );
    assert!(
        rel(exposure_native, exposure_core) < 0.02,
        "auto-exposure diverged between Native and Core: \
         native={exposure_native}, core={exposure_core}"
    );
}
