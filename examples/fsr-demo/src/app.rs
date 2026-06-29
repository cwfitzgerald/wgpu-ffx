//! Winit application handler: window lifecycle, event routing, and the
//! per-frame redraw loop.
//!
//! [`App`] owns the [`Settings`], the [`Gpu`] state, the egui overlay, and the
//! phase-3 renderer/camera/scene. [`App::redraw`] advances the camera and scene
//! animation, records the geometry + composite passes via the [`Renderer`], and
//! draws the overlay on top.

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::camera::Camera;
use crate::egui_overlay::{EguiOverlay, FrameStats, RenderContext};
use crate::fsr::{FsrInputs, FsrPass};
use crate::gpu::Gpu;
use crate::renderer::{Frame, Renderer};
use crate::scene::Scene;
use crate::settings::Settings;
use wgpu_ffx::{get_jitter_offset, get_jitter_phase_count};

const WINDOW_TITLE: &str = "wgpu-ffx — FSR3 demo";

/// Which composite source the renderer is currently pointed at, so we only
/// rebuild the composite bind group when it actually changes (not per frame).
#[derive(Clone, Copy, PartialEq, Eq)]
enum CompositeSource {
    /// Raw HDR color target (FSR off).
    Hdr,
    /// FSR upscaled output (FSR on).
    Fsr,
}

/// Human-readable label for the FSR format profile, for the overlay.
fn profile_label(profile: wgpu_ffx::FormatProfile) -> &'static str {
    match profile {
        wgpu_ffx::FormatProfile::Core => "Core",
        wgpu_ffx::FormatProfile::Tier2 => "Tier2",
        wgpu_ffx::FormatProfile::Native => "Native",
    }
}

/// Per-window state, created once the event loop resumes.
struct Active {
    window: Arc<Window>,
    gpu: Gpu,
    overlay: EguiOverlay,
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    /// FSR upscaler pass. `None` if FSR failed to initialize (the demo then
    /// runs the native FSR-off path only).
    fsr: Option<FsrPass>,
    /// Previous-frame unjittered view-proj, for camera motion vectors.
    prev_view_proj: glam::Mat4,
    /// Whether the cursor is currently grabbed/hidden for mouse-look.
    cursor_grabbed: bool,
    /// Halton jitter frame index, incremented per rendered frame and reset to 0
    /// on a history reset.
    frame_index: u32,
    /// `fsr_enabled` last frame, to detect a toggle (forces a history reset).
    prev_fsr_enabled: bool,
    /// Render resolution last frame, to detect a change.
    prev_render_size: [u32; 2],
    /// Display resolution last frame, to detect a change.
    prev_display: [u32; 2],
    /// Which source the composite pass is currently bound to.
    composite_source: CompositeSource,
    /// True until the first frame is rendered (forces a history reset).
    first_frame: bool,
}

/// The demo application.
#[derive(Default)]
pub struct App {
    settings: Settings,
    active: Option<Active>,
    last_frame: Option<Instant>,
    fps: f32,
}

impl App {
    /// Construct a fresh app with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update the smoothed FPS estimate and return current frame stats for the
    /// given display resolution. Also returns the frame delta time in seconds.
    fn tick_stats(
        &mut self,
        display: [u32; 2],
        render_size: [u32; 2],
        fsr_profile: Option<&'static str>,
    ) -> (FrameStats, f32) {
        let now = Instant::now();
        let mut dt = 0.0;
        if let Some(prev) = self.last_frame.replace(now) {
            dt = now.duration_since(prev).as_secs_f32();
            if dt > 0.0 {
                let instant_fps = 1.0 / dt;
                // Exponential moving average to keep the readout stable.
                self.fps = if self.fps == 0.0 {
                    instant_fps
                } else {
                    self.fps * 0.9 + instant_fps * 0.1
                };
            }
        }
        // Clamp dt so a hitch (or the first frame) can't teleport the camera.
        let dt = dt.clamp(0.0, 0.1);

        let stats = FrameStats {
            fps: self.fps,
            render_size,
            display_size: display,
            fsr_profile,
        };
        (stats, dt)
    }

    /// Render a single frame: advance camera + scene, run the geometry and
    /// composite passes, then draw the overlay.
    fn redraw(&mut self) {
        let Some(active) = self.active.as_mut() else {
            return;
        };

        let frame = match active.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                // Surface configuration is stale; reconfigure and retry next frame.
                active
                    .gpu
                    .surface
                    .configure(&active.gpu.device, &active.gpu.config);
                return;
            }
            other => {
                // Timeout / Occluded / Validation: skip this frame and try again.
                log::debug!("skipping frame: {other:?}");
                return;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let display = [active.gpu.config.width, active.gpu.config.height];
        // FSR can only run if it initialized; treat a missing pass as "off".
        let fsr_enabled = self.settings.fsr_enabled && active.fsr.is_some();
        // Render resolution: full display unless FSR is enabled and scaling.
        // (When FSR is off we render natively at display res — the aliased,
        // no-temporal-AA baseline the demo compares against.)
        let render_size = if fsr_enabled {
            self.settings.quality.render_size(display)
        } else {
            display
        };

        // (Re)allocate the render targets only when the size actually changes.
        // A resize rebuilds the HDR color view and re-points the composite at
        // it internally, so we must re-point the composite afterward.
        let renderer_resized = active.renderer.render_size() != render_size;
        active.renderer.resize(&active.gpu.device, render_size);

        // Decide whether to reset FSR temporal history this frame: on the first
        // frame, when FSR is toggled on, when render/display size changes, when
        // FSR (re)allocated its view/resources, or when the user asked.
        let mut reset_history = active.first_frame
            || (fsr_enabled && !active.prev_fsr_enabled)
            || render_size != active.prev_render_size
            || display != active.prev_display
            || std::mem::take(&mut self.settings.reset_accumulation);

        // (Re)allocate FSR view + resources when sizes change. `prepare`
        // rebuilds accumulation buffers on a grow, which also needs a reset.
        // When it reallocates the output texture, its view is recreated, so the
        // composite bind group must be re-pointed at the fresh `output_view`.
        let mut fsr_realloc = false;
        if fsr_enabled
            && let Some(fsr) = active.fsr.as_mut()
            && fsr.prepare(&active.gpu.device, &active.gpu.queue, render_size, display)
        {
            reset_history = true;
            fsr_realloc = true;
        }

        active.prev_fsr_enabled = fsr_enabled;
        active.prev_render_size = render_size;
        active.prev_display = display;
        active.first_frame = false;

        let fsr_profile = active.fsr.as_ref().map(|f| profile_label(f.profile()));
        let (stats, dt) = self.tick_stats(display, render_size, fsr_profile);
        let active = self.active.as_mut().expect("active set above");

        // Sync cursor grab with look mode (best-effort; ignore errors).
        active.sync_cursor_grab();

        // Advance simulation.
        active.camera.update(dt, self.settings.camera_speed);
        active.scene.update(dt, self.settings.paused);

        // Jitter: Halton when FSR is on, none otherwise. The SAME offset is fed
        // to the projection (so the rasterized color is jittered) and to FSR's
        // `jitter_offset`. Reset the phase index on a history reset.
        if reset_history {
            active.frame_index = 0;
        }
        let jitter_px = if fsr_enabled {
            let phase_count =
                get_jitter_phase_count(render_size[0] as i32, display[0] as i32).max(1);
            get_jitter_offset(active.frame_index as i32, phase_count)
        } else {
            [0.0_f32, 0.0_f32]
        };

        // Camera matrices.
        let aspect = render_size[0] as f32 / render_size[1].max(1) as f32;
        let view_proj = active.camera.view_proj(aspect);
        let view_proj_jittered =
            active
                .camera
                .jittered_proj(aspect, jitter_px, render_size)
                * active.camera.view_matrix();
        let camera_fov_y = active.camera.fov_y;
        let camera_near = active.camera.znear;
        let camera_far = active.camera.zfar;

        // Build the overlay UI for this frame.
        let output = active
            .overlay
            .run(&active.window, &mut self.settings, stats);

        let mut encoder =
            active
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frame encoder"),
                });

        // Flatten the scene and record the geometry (MRT) pass.
        let (instances, draws) = active.scene.draw_data();
        let frame_data = Frame {
            view_proj_jittered,
            view_proj,
            prev_view_proj: active.prev_view_proj,
            camera_pos: active.camera.position,
            lighting: crate::renderer::Lighting::default(),
            instances: &instances,
            draws: &draws,
        };
        active.renderer.render_geometry(
            &active.gpu.device,
            &active.gpu.queue,
            &mut encoder,
            &frame_data,
        );
        drop(instances);
        drop(draws);

        // FSR dispatch (when on) between geometry and composite, then point the
        // composite at the right source. Only rebuild the bind group on change.
        let want_source = if fsr_enabled {
            let targets = active.renderer.targets();
            let inputs = FsrInputs {
                color: &targets.color,
                depth: &targets.depth,
                motion_vectors: &targets.motion,
                render_size,
                upscale_size: display,
                jitter_offset: jitter_px,
                camera_near,
                camera_far,
                camera_fov_y,
                enable_sharpening: self.settings.sharpening,
                sharpness: self.settings.sharpness,
                // FSR wants milliseconds, >= 1.0.
                frame_time_ms: (dt * 1000.0).max(1.0),
                reset_history,
            };
            let fsr = active.fsr.as_mut().expect("fsr_enabled implies Some");
            match fsr.dispatch(&mut encoder, &inputs) {
                Ok(()) => CompositeSource::Fsr,
                Err(e) => {
                    // Don't spam every frame; log and fall back to raw HDR.
                    log::error!("FSR dispatch failed, compositing raw HDR: {e}");
                    CompositeSource::Hdr
                }
            }
        } else {
            CompositeSource::Hdr
        };

        // Re-point the composite source only when needed: when the desired
        // source kind changed, when a renderer resize rebuilt (and re-bound) the
        // HDR color view (which invalidates a stale FSR binding), or when FSR
        // reallocated its output texture (whose view is now fresh).
        if active.composite_source != want_source || renderer_resized || fsr_realloc {
            match want_source {
                CompositeSource::Fsr => {
                    let view = active
                        .fsr
                        .as_ref()
                        .expect("Fsr source implies Some")
                        .output_view()
                        .clone();
                    active
                        .renderer
                        .set_composite_source(&active.gpu.device, &view);
                }
                CompositeSource::Hdr => {
                    // `Renderer::resize` already re-bound the fresh HDR view, so
                    // only re-point explicitly when the kind changed.
                    if active.composite_source != want_source {
                        let view = active.renderer.targets().color_view_clone();
                        active
                            .renderer
                            .set_composite_source(&active.gpu.device, &view);
                    }
                }
            }
            active.composite_source = want_source;
        }

        active.renderer.composite(&mut encoder, &view);

        // Roll the unjittered view-proj forward for next frame's motion vectors.
        active.prev_view_proj = view_proj;
        // Advance the jitter phase for the next rendered frame.
        active.frame_index = active.frame_index.wrapping_add(1);

        let size_in_pixels = [active.gpu.config.width, active.gpu.config.height];
        active.overlay.render(
            RenderContext {
                device: &active.gpu.device,
                queue: &active.gpu.queue,
                encoder: &mut encoder,
                window: active.window.as_ref(),
                view: &view,
                size_in_pixels,
            },
            output,
        );

        active.gpu.queue.submit(Some(encoder.finish()));
        active.window.pre_present_notify();
        frame.present();

        // Keep animating: request the next frame immediately.
        active.window.request_redraw();
    }
}

impl Active {
    /// Grab + hide the cursor while looking, release it otherwise. Best effort;
    /// platform errors are logged at debug level and ignored.
    fn sync_cursor_grab(&mut self) {
        let want = self.camera.looking;
        if want == self.cursor_grabbed {
            return;
        }
        if want {
            let grab = self
                .window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Confined));
            if let Err(e) = grab {
                log::debug!("cursor grab failed: {e}");
            }
            self.window.set_cursor_visible(false);
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
            self.window.set_cursor_visible(true);
        }
        self.cursor_grabbed = want;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.active.is_some() {
            return;
        }

        let attributes = Window::default_attributes().with_title(WINDOW_TITLE);
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(e) => {
                log::error!("failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        let gpu = match Gpu::new(window.clone()) {
            Ok(gpu) => gpu,
            Err(e) => {
                log::error!("failed to initialize wgpu: {e:#}");
                event_loop.exit();
                return;
            }
        };

        let overlay = EguiOverlay::new(&gpu.device, &window, gpu.surface_format());

        // Build the renderer at the initial render resolution.
        let display = [gpu.config.width, gpu.config.height];
        let render_size = if self.settings.fsr_enabled {
            self.settings.quality.render_size(display)
        } else {
            display
        };
        let mut renderer = Renderer::new(&gpu.device, &gpu.queue, gpu.surface_format(), render_size);

        let scene = match Scene::new(&gpu.device, &gpu.queue, &mut renderer) {
            Ok(scene) => scene,
            Err(e) => {
                log::error!("failed to build scene: {e:#}");
                event_loop.exit();
                return;
            }
        };

        let camera = Camera::default();
        let prev_view_proj = {
            let aspect = render_size[0] as f32 / render_size[1].max(1) as f32;
            camera.view_proj(aspect)
        };

        // Build the FSR upscaler. A failure here is non-fatal: log it and run
        // the native FSR-off path only (the demo still works as a baseline).
        let fsr = match FsrPass::new(&gpu.adapter, &gpu.device, &gpu.queue, display) {
            Ok(fsr) => Some(fsr),
            Err(e) => {
                log::error!("failed to initialize FSR (running without it): {e:#}");
                None
            }
        };

        window.request_redraw();
        self.active = Some(Active {
            window,
            gpu,
            overlay,
            renderer,
            scene,
            camera,
            fsr,
            cursor_grabbed: false,
            prev_view_proj,
            frame_index: 0,
            prev_fsr_enabled: false,
            prev_render_size: render_size,
            prev_display: display,
            composite_source: CompositeSource::Hdr,
            first_frame: true,
        });
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        // Raw, unaccelerated mouse motion drives the free-fly look.
        if let DeviceEvent::MouseMotion { delta } = event {
            active.camera.on_mouse_motion(delta.0, delta.1);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(active) = self.active.as_mut() else {
            return;
        };

        // egui gets first crack at the event; if it consumes it, stop here
        // (except for window-level events we always honour).
        let egui_consumed = active.overlay.on_window_event(&active.window, &event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
                return;
            }
            WindowEvent::Resized(size) => {
                active.gpu.resize(size.width, size.height);
                active.window.request_redraw();
                return;
            }
            WindowEvent::RedrawRequested => {
                self.redraw();
                return;
            }
            WindowEvent::Focused(false) => {
                // Drop held keys on focus loss so movement doesn't stick.
                active.camera.release_all_keys();
                active.camera.looking = false;
                return;
            }
            _ => {}
        }

        // Camera input. Always honour key/button *releases* so they never stick
        // even if egui swallowed the press; only consume presses if egui didn't.
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let is_release = event.state == winit::event::ElementState::Released;
                if !egui_consumed || is_release {
                    active.camera.on_key(event.physical_key, event.state);
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                let is_release = state == winit::event::ElementState::Released;
                if !egui_consumed || is_release {
                    active.camera.on_mouse_button(button, state);
                }
            }
            _ => {}
        }
    }
}
