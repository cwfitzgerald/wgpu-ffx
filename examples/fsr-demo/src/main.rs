//! `fsr-demo`: an interactive 3D demo of AMD FidelityFX FSR3 upscaling on top
//! of wgpu.
//!
//! This is the foundation phase: a windowed wgpu app with an egui control
//! overlay and the shared [`settings::Settings`] contract. glTF loading, a PBR
//! renderer, and FSR integration land in later phases.

mod app;
// Renderer-friendly intermediate types produced by the loader and consumed by
// the phase-3 renderer/scene. A few fields/helpers (e.g. `GpuTexture::texture`,
// `Aabb::is_valid`) are part of the foundational API or only used by tests.
#[allow(dead_code)]
mod assets;
mod camera;
mod egui_overlay;
mod fsr;
// glTF -> GPU asset loader (phase 2), driven by the phase-3 scene. Still covered
// by the inline load test.
mod gltf_loader;
mod gpu;
mod renderer;
mod scene;
mod settings;

use anyhow::{Context as _, Result};
use winit::event_loop::{ControlFlow, EventLoop};

use crate::app::App;

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn"),
    )
    .init();

    let event_loop = EventLoop::new().context("failed to create event loop")?;
    // Animate continuously; the app also requests redraws per frame.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop
        .run_app(&mut app)
        .context("event loop terminated with an error")?;

    Ok(())
}
