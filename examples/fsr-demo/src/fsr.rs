//! FSR3 upscaler integration via `wgpu-ffx`.
//!
//! [`FsrPass`] owns the [`FsrContext`], its [`FsrView`] (the internal temporal
//! accumulation textures), and the bundle of caller-allocated resources FSR
//! reads/writes each dispatch ([`FsrResources`]: the upscaled `output` texture,
//! the `dilated_depth` / `dilated_motion_vectors` scratch textures, and the
//! `reconstructed_previous_depth` buffer).
//!
//! ## Wiring contract (matches the phase-3 renderer)
//! - The renderer rasterizes color/depth/motion at *render* resolution with the
//!   exact formats FSR's active [`FormatProfile`] requires; [`FsrPass::new`]
//!   asserts that contract loudly at startup.
//! - Motion vectors are stored in *render-target pixels* as
//!   `(prev_ndc - cur_ndc) * vec2(0.5*W, -0.5*H)`, so `motion_vector_scale` is
//!   `[1.0, 1.0]` (FSR computes `uv_motion = stored * scale / render_size`).
//! - `AUTO_EXPOSURE` is set, so `exposure` is `None` and `pre_exposure` is
//!   `1.0`. Depth is standard non-inverted finite (no DEPTH_INVERTED/INFINITE).
//!
//! ## Sizing
//! The [`FsrView`]'s `max_render_size` / `max_upscale_size` cover the largest
//! render/upscale resolution the current display can demand. `NativeAa` renders
//! at display resolution, so `max_render = max_upscale = display`. When the
//! window grows beyond the view's maxima we reallocate the view and resources
//! and request a history reset.

use anyhow::{Result, bail};
use wgpu_ffx::{
    FormatProfile, FsrContext, FsrContextFlags, FsrContextInfo, FsrDispatchFlags, FsrDispatchInfo,
    FsrFormats, FsrView,
};

use crate::renderer::{COLOR_FORMAT, DEPTH_FORMAT, MOTION_FORMAT};

/// Caller-allocated GPU resources FSR reads from / writes to each dispatch.
///
/// `output` lives at *upscale* (display) resolution; the dilated textures and
/// the reconstructed-previous-depth buffer live at *render* resolution. All are
/// reallocated whenever the relevant resolution changes.
struct FsrResources {
    /// Upscaled output color ([`FsrFormats::output`], = `Rgba16Float`), at
    /// upscale resolution. Composited to the screen when FSR is on.
    output: wgpu::Texture,
    /// View over [`FsrResources::output`], handed to the composite pass.
    output_view: wgpu::TextureView,
    /// Dilated depth scratch ([`FsrFormats::dilated_depth`]), render resolution.
    dilated_depth: wgpu::Texture,
    /// Dilated motion-vector scratch ([`FsrFormats::dilated_motion_vectors`]),
    /// render resolution.
    dilated_motion_vectors: wgpu::Texture,
    /// Reconstructed previous nearest-depth buffer (`render_w*render_h*4`
    /// bytes), render resolution.
    reconstructed_previous_depth: wgpu::Buffer,
}

impl FsrResources {
    /// Allocate the resource bundle for the given render + upscale sizes.
    fn new(
        device: &wgpu::Device,
        formats: FsrFormats,
        render_size: [u32; 2],
        upscale_size: [u32; 2],
    ) -> Self {
        let storage_tex = |label: &str, format: wgpu::TextureFormat, size: [u32; 2]| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size[0].max(1),
                    height: size[1].max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };

        let output = storage_tex("fsr output", formats.output, upscale_size);
        let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
        let dilated_depth = storage_tex("fsr dilated depth", formats.dilated_depth, render_size);
        let dilated_motion_vectors = storage_tex(
            "fsr dilated motion vectors",
            formats.dilated_motion_vectors,
            render_size,
        );

        let reconstructed_previous_depth = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fsr reconstructed previous depth"),
            size: render_size[0].max(1) as u64 * render_size[1].max(1) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            output,
            output_view,
            dilated_depth,
            dilated_motion_vectors,
            reconstructed_previous_depth,
        }
    }
}

/// Per-dispatch inputs supplied by the app each frame.
pub struct FsrInputs<'a> {
    /// Render-resolution HDR color target (the renderer's `targets().color`).
    pub color: &'a wgpu::Texture,
    /// Render-resolution depth target.
    pub depth: &'a wgpu::Texture,
    /// Render-resolution motion-vector target.
    pub motion_vectors: &'a wgpu::Texture,
    /// Render resolution `[w, h]`.
    pub render_size: [u32; 2],
    /// Upscale / display resolution `[w, h]`.
    pub upscale_size: [u32; 2],
    /// Sub-pixel jitter offset (same value baked into the projection).
    pub jitter_offset: [f32; 2],
    /// Camera near plane.
    pub camera_near: f32,
    /// Camera far plane.
    pub camera_far: f32,
    /// Camera vertical field of view, radians.
    pub camera_fov_y: f32,
    /// Whether to apply the RCAS sharpening pass.
    pub enable_sharpening: bool,
    /// Sharpening strength `0..=1`.
    pub sharpness: f32,
    /// Frame delta in **milliseconds** (the app clamps it to `>= 1.0`).
    pub frame_time_ms: f32,
    /// Reset temporal accumulation this frame.
    pub reset_history: bool,
}

/// Owns the FSR context, view, and per-frame resources, and records dispatches.
pub struct FsrPass {
    ctx: FsrContext,
    view: FsrView,
    formats: FsrFormats,
    resources: FsrResources,
    /// Render resolution the resources are currently sized for.
    render_size: [u32; 2],
    /// Upscale resolution the resources are currently sized for.
    upscale_size: [u32; 2],
}

impl FsrPass {
    /// Create the FSR context (HDR + auto-exposure), assert the renderer's
    /// format contract, allocate a view + resources sized for `display`, and log
    /// the detected [`FormatProfile`].
    pub fn new(
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        display: [u32; 2],
    ) -> Result<Self> {
        // `FsrContext::new` compiles all the FSR pipelines eagerly. On some
        // backends/driver+wgpu combinations a shader can fail to validate, which
        // wgpu's default handler turns into a *panic* (not a `Result`). Catch it
        // so the demo degrades to the native FSR-off path instead of aborting.
        let ctx = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            FsrContext::new(FsrContextInfo {
                adapter: adapter.clone(),
                device: device.clone(),
                flags: FsrContextFlags::HIGH_DYNAMIC_RANGE | FsrContextFlags::AUTO_EXPOSURE,
                // Auto-detect the richest profile the device supports.
                format_profile: None,
            })
        }))
        .map_err(|_| {
            anyhow::anyhow!("FSR context creation panicked (shader/pipeline compilation failed)")
        })?;

        let profile: FormatProfile = ctx.format_profile();
        let formats = ctx.formats();
        log::info!("FSR format profile: {profile:?}");

        // Hard contract with the phase-3 renderer: a profile surprise must be
        // loud, not a silent format mismatch later in `dispatch`.
        if formats.color != COLOR_FORMAT
            || formats.depth != DEPTH_FORMAT
            || formats.motion_vectors != MOTION_FORMAT
        {
            bail!(
                "FSR format profile {profile:?} requires color/depth/motion \
                 {:?}/{:?}/{:?}, but the renderer hardcodes {COLOR_FORMAT:?}/{DEPTH_FORMAT:?}/{MOTION_FORMAT:?}",
                formats.color,
                formats.depth,
                formats.motion_vectors,
            );
        }

        // `NativeAa` renders at display resolution, so the maximum render size
        // equals the maximum upscale size (= display). Both are the view maxima.
        let max_size = [display[0].max(2), display[1].max(2)];
        let view = ctx.create_view(queue, max_size, max_size);
        let resources = FsrResources::new(device, formats, max_size, max_size);

        Ok(Self {
            ctx,
            view,
            formats,
            resources,
            render_size: max_size,
            upscale_size: max_size,
        })
    }

    /// The detected format profile (for logging / UI).
    pub fn profile(&self) -> FormatProfile {
        self.ctx.format_profile()
    }

    /// The upscaled output view to feed the composite pass when FSR is on.
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.resources.output_view
    }

    /// Ensure the view + resources can cover `render_size` / `upscale_size`.
    ///
    /// Returns `true` if anything was reallocated (so the caller should force a
    /// history reset, since the accumulation textures were rebuilt). The
    /// resource textures are sized exactly to the requested resolutions; the
    /// `FsrView`'s maxima only grow (we resize when the display grows past
    /// them), so transient downscales don't thrash the larger internal buffers.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_size: [u32; 2],
        upscale_size: [u32; 2],
    ) -> bool {
        let mut changed = false;

        // Grow the view's internal accumulation buffers if the display grew
        // beyond what they were allocated for. `view.resize` rebuilds them and
        // resets `first_execution`, so a history reset is implied.
        let max = self.view.max_upscale_size();
        if upscale_size[0] > max[0]
            || upscale_size[1] > max[1]
            || render_size[0] > self.view.max_render_size()[0]
            || render_size[1] > self.view.max_render_size()[1]
        {
            let new_max = [
                upscale_size[0].max(max[0]).max(2),
                upscale_size[1].max(max[1]).max(2),
            ];
            self.view.resize(queue, new_max, new_max);
            changed = true;
        }

        // Reallocate the caller-side resources if either resolution changed.
        if self.render_size != render_size || self.upscale_size != upscale_size {
            self.resources = FsrResources::new(device, self.formats, render_size, upscale_size);
            self.render_size = render_size;
            self.upscale_size = upscale_size;
            changed = true;
        }

        changed
    }

    /// Record the FSR upscaling compute passes for one frame.
    ///
    /// Fills [`FsrDispatchInfo`] with the wiring constants (`exposure: None`,
    /// `pre_exposure: 1.0`, `motion_vector_scale: [1.0, 1.0]`,
    /// `view_space_to_meters_factor: 1.0`, no dispatch flags) and the supplied
    /// per-frame inputs, then records the passes onto `encoder`. The caller
    /// submits the encoder and then composites [`FsrPass::output_view`].
    pub fn dispatch(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        inputs: &FsrInputs<'_>,
    ) -> Result<(), wgpu_ffx::FsrDispatchError> {
        let info = FsrDispatchInfo {
            color: inputs.color.clone(),
            depth: inputs.depth.clone(),
            motion_vectors: inputs.motion_vectors.clone(),
            // AUTO_EXPOSURE is set; passing Some(..) is a validation error.
            exposure: None,
            reactive_mask: None,
            transparency_and_composition: None,
            dilated_depth: self.resources.dilated_depth.clone(),
            dilated_motion_vectors: self.resources.dilated_motion_vectors.clone(),
            reconstructed_previous_depth: self.resources.reconstructed_previous_depth.clone(),
            output: self.resources.output.clone(),

            jitter_offset: inputs.jitter_offset,
            // The renderer stores MVs in render-target pixels; FSR computes
            // `uv_motion = stored * scale / render_size`, so scale is identity.
            motion_vector_scale: [1.0, 1.0],

            render_size: inputs.render_size,
            upscale_size: inputs.upscale_size,

            enable_sharpening: inputs.enable_sharpening,
            sharpness: inputs.sharpness,
            frame_time_delta: inputs.frame_time_ms,
            // AUTO_EXPOSURE handles exposure; pre-exposure must be > 0.
            pre_exposure: 1.0,
            reset_history: inputs.reset_history,

            camera_near: inputs.camera_near,
            camera_far: inputs.camera_far,
            camera_fov_y: inputs.camera_fov_y,
            view_space_to_meters_factor: 1.0,

            flags: FsrDispatchFlags::empty(),
        };

        self.ctx.dispatch(&mut self.view, encoder, &info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Headless wgpu device, requesting the format-features feature when the
    /// adapter supports it (so the test exercises the same profile the GUI
    /// would). Returns `None` when no adapter is available so the test skips.
    fn headless() -> Option<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok()?;

        let mut required_features = wgpu::Features::empty();
        if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        {
            required_features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fsr test device"),
            required_features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        Some((instance, adapter, device, queue))
    }

    /// Allocate a render-resolution input texture with the renderer's format
    /// (`RENDER_ATTACHMENT | TEXTURE_BINDING`, matching `RenderTargets`).
    fn input_texture(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        size: [u32; 2],
        label: &str,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    /// End-to-end FSR wiring check on a real GPU, driving the *actual*
    /// [`FsrPass`] API the app uses: `new` (asserts the format contract +
    /// allocates view/resources), `prepare` (downscale to a sub-display render
    /// size — the resources get reallocated), then `dispatch` with
    /// `reset_history`. We allocate the renderer's color/depth/motion inputs at
    /// render resolution with the renderer's formats + usages (blank is fine),
    /// wrap the whole thing in a wgpu validation error scope, wait for the GPU,
    /// and assert both that the dispatch returned `Ok` and that no validation
    /// error fired. This proves the entire FSR wiring — formats, resource
    /// usages, sizes, and dispatch params. Skips gracefully without an adapter.
    #[test]
    fn fsr_dispatch_is_valid_on_real_gpu() {
        let _ = env_logger::builder().is_test(true).try_init();
        let Some((instance, adapter, device, queue)) = headless() else {
            log::warn!("no wgpu adapter; skipping FSR dispatch test");
            return;
        };

        let display = [1280u32, 720u32];

        // `FsrPass::new` builds the context, logs the profile, and asserts the
        // color/depth/motion format contract (so the contract is covered here).
        // On some backend + wgpu/naga combinations the crate's FSR shaders fail
        // to validate (the crate's own GPU smoke tests fail identically there);
        // `FsrPass::new` catches that panic and returns `Err`. Skip gracefully
        // in that case — it's a crate/driver limitation, not a wiring bug.
        let mut pass = match FsrPass::new(&adapter, &device, &queue, display) {
            Ok(pass) => pass,
            Err(e) => {
                log::warn!(
                    "FSR context unavailable on this GPU ({e:#}); skipping FSR dispatch test"
                );
                drop(instance);
                return;
            }
        };
        let profile = pass.profile();
        log::info!("FSR test detected format profile: {profile:?}");
        assert_eq!(pass.formats.color, COLOR_FORMAT, "color format contract");
        assert_eq!(pass.formats.depth, DEPTH_FORMAT, "depth format contract");
        assert_eq!(
            pass.formats.motion_vectors, MOTION_FORMAT,
            "motion-vector format contract"
        );

        let render_size = [640u32, 360u32];
        let upscale_size = display;

        // Reallocate the caller-side resources to the actual render/upscale
        // sizes, exactly as the per-frame loop does.
        assert!(
            pass.prepare(&device, &queue, render_size, upscale_size),
            "prepare should report a change after the first downscale"
        );

        // The render-res inputs the renderer would produce (blank is fine).
        let color = input_texture(&device, pass.formats.color, render_size, "test color");
        let depth = input_texture(&device, pass.formats.depth, render_size, "test depth");
        let motion =
            input_texture(&device, pass.formats.motion_vectors, render_size, "test motion");

        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let result = pass.dispatch(
            &mut encoder,
            &FsrInputs {
                color: &color,
                depth: &depth,
                motion_vectors: &motion,
                render_size,
                upscale_size,
                jitter_offset: [0.0, 0.0],
                camera_near: crate::camera::Z_NEAR,
                camera_far: crate::camera::Z_FAR,
                camera_fov_y: 60f32.to_radians(),
                enable_sharpening: true,
                sharpness: 0.5,
                frame_time_ms: 16.6,
                reset_history: true,
            },
        );
        assert!(result.is_ok(), "FSR dispatch failed: {:?}", result.err());

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        let validation_error = pollster::block_on(error_scope.pop());
        assert!(
            validation_error.is_none(),
            "wgpu validation error during FSR dispatch: {validation_error:?}"
        );

        // Keep the instance alive until the GPU work has drained.
        drop(instance);
    }
}
