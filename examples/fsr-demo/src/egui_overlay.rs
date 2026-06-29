//! egui overlay: input handling (egui-winit) and rendering (egui-wgpu).
//!
//! [`EguiOverlay`] bundles the winit input integration with the wgpu renderer
//! and owns the panel layout. The app feeds it window events, then once per
//! frame calls [`EguiOverlay::run`] to build the UI and [`EguiOverlay::render`]
//! to record it into the frame's render pass.

use std::sync::Arc;

use egui_wgpu::ScreenDescriptor;
use winit::window::Window;

use crate::settings::{QualityMode, Settings};

/// Read-only stats shown in the overlay. Populated by the app each frame;
/// placeholder values are fine until later phases wire up real numbers.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    /// Smoothed frames per second.
    pub fps: f32,
    /// Internal render resolution `[w, h]`.
    pub render_size: [u32; 2],
    /// Display (output) resolution `[w, h]`.
    pub display_size: [u32; 2],
    /// Detected FSR format profile label (e.g. `"Native"`), or `None` if FSR
    /// failed to initialize.
    pub fsr_profile: Option<&'static str>,
}

/// Per-frame handles needed to record the overlay into the frame's encoder.
pub struct RenderContext<'a> {
    /// The logical device.
    pub device: &'a wgpu::Device,
    /// The submission queue.
    pub queue: &'a wgpu::Queue,
    /// The frame's command encoder.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The window (for platform output, e.g. cursor and clipboard).
    pub window: &'a Window,
    /// The color target view to draw the overlay onto.
    pub view: &'a wgpu::TextureView,
    /// The target size in physical pixels `[w, h]`.
    pub size_in_pixels: [u32; 2],
}

/// egui integration state and renderer.
pub struct EguiOverlay {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl EguiOverlay {
    /// Create the overlay for the given window and surface format.
    pub fn new(
        device: &wgpu::Device,
        window: &Arc<Window>,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let ctx = egui::Context::default();
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );

        Self {
            ctx,
            state,
            renderer,
        }
    }

    /// Forward a window event to egui. Returns `true` if egui consumed the
    /// event (and the app should not act on it).
    pub fn on_window_event(
        &mut self,
        window: &Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        self.state.on_window_event(window, event).consumed
    }

    /// Build the overlay UI for this frame, mutating `settings` in place.
    ///
    /// Returns the egui [`FullOutput`](egui::FullOutput) to be tessellated and
    /// rendered by [`EguiOverlay::render`].
    pub fn run(
        &mut self,
        window: &Window,
        settings: &mut Settings,
        stats: FrameStats,
    ) -> egui::FullOutput {
        let raw_input = self.state.take_egui_input(window);
        self.ctx.run_ui(raw_input, |ui| {
            build_ui(ui, settings, stats);
        })
    }

    /// Record the previously built UI into a render pass targeting `view`.
    ///
    /// Must be called after [`EguiOverlay::run`] with the same frame's output.
    pub fn render(&mut self, ctx: RenderContext<'_>, output: egui::FullOutput) {
        let RenderContext {
            device,
            queue,
            encoder,
            window,
            view,
            size_in_pixels,
        } = ctx;

        self.state
            .handle_platform_output(window, output.platform_output);

        let pixels_per_point = self.ctx.pixels_per_point();
        let paint_jobs = self
            .ctx
            .tessellate(output.shapes, pixels_per_point);

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels,
            pixels_per_point,
        };

        for (id, image_delta) in &output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }
        self.renderer
            .update_buffers(device, queue, encoder, &paint_jobs, &screen_descriptor);

        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // egui-wgpu requires a `'static` render pass lifetime.
            let mut render_pass = render_pass.forget_lifetime();
            self.renderer
                .render(&mut render_pass, &paint_jobs, &screen_descriptor);
        }

        for id in &output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}

/// Build the side panel and stats overlay.
fn build_ui(ui: &mut egui::Ui, settings: &mut Settings, stats: FrameStats) {
    egui::Panel::left("controls")
        .resizable(false)
        .default_size(260.0)
        .show(ui, |ui| {
            ui.heading("FSR3 Demo");
            ui.separator();

            ui.label("Upscaling");
            ui.checkbox(&mut settings.fsr_enabled, "Enable FSR");
            // The quality preset only governs the FSR render resolution; when
            // FSR is off the demo renders at native display resolution (the
            // aliased, no-temporal-AA baseline), so disable the combo.
            ui.add_enabled_ui(settings.fsr_enabled, |ui| {
                egui::ComboBox::from_label("Quality (FSR on)")
                    .selected_text(settings.quality.label())
                    .show_ui(ui, |ui| {
                        for mode in QualityMode::ALL {
                            ui.selectable_value(&mut settings.quality, mode, mode.label());
                        }
                    });
            });
            ui.add_enabled_ui(settings.fsr_enabled, |ui| {
                ui.checkbox(&mut settings.sharpening, "Sharpening");
                ui.add_enabled(
                    settings.sharpening,
                    egui::Slider::new(&mut settings.sharpness, 0.0..=1.0).text("Sharpness"),
                );
                if ui.button("Reset accumulation").clicked() {
                    settings.reset_accumulation = true;
                }
            });
            if let Some(profile) = stats.fsr_profile {
                ui.small(format!("FSR profile: {profile}"));
            } else {
                ui.small("FSR profile: unavailable");
            }

            ui.separator();
            ui.label("Camera & Scene");
            ui.add(
                egui::Slider::new(&mut settings.camera_speed, 0.5..=50.0).text("Camera speed"),
            );
            ui.checkbox(&mut settings.paused, "Pause animation");

            ui.separator();
            ui.checkbox(&mut settings.show_stats, "Show stats");
            if settings.show_stats {
                ui.label(format!("FPS: {:.1}", stats.fps));
                ui.label(format!(
                    "Render -> Display: {}x{} -> {}x{}",
                    stats.render_size[0],
                    stats.render_size[1],
                    stats.display_size[0],
                    stats.display_size[1],
                ));
            }

            ui.separator();
            ui.label("Controls");
            ui.small("WASD - move");
            ui.small("Q / E - down / up");
            ui.small("Right mouse - look");
        });
}
