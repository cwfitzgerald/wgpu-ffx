//! wgpu device, surface, and swapchain management.
//!
//! [`Gpu`] owns everything tied to the rendering device and the window
//! surface. Later phases pass these handles (`device`, `queue`, `adapter`)
//! straight into `wgpu-ffx`, so they must come from this single wgpu build.

use std::sync::Arc;

use anyhow::{Context as _, Result};
use winit::window::Window;

/// Owns the wgpu instance, surface, adapter, device, queue, and the current
/// surface configuration.
///
/// `instance` and `adapter` are unused this phase but are kept because later
/// phases hand the adapter to `wgpu-ffx` (`FsrContextInfo`) and may need to
/// recreate the surface from the instance.
#[allow(dead_code)]
pub struct Gpu {
    /// The wgpu instance the surface and adapter were created from.
    pub instance: wgpu::Instance,
    /// The window surface rendered to each frame.
    pub surface: wgpu::Surface<'static>,
    /// The adapter backing the device (needed by `wgpu-ffx` later).
    pub adapter: wgpu::Adapter,
    /// The logical device used to create GPU resources.
    pub device: wgpu::Device,
    /// The queue used to submit work.
    pub queue: wgpu::Queue,
    /// Current surface configuration (kept in sync with the window size).
    pub config: wgpu::SurfaceConfiguration,
}

impl Gpu {
    /// Initialize wgpu for the given window.
    ///
    /// Requests an adapter compatible with the window surface, a device with no
    /// special features, and configures the surface with an sRGB format when
    /// one is available.
    pub fn new(window: Arc<Window>) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let surface = instance
            .create_surface(window.clone())
            .context("failed to create surface")?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .context("failed to find a compatible adapter")?;

        log::info!("using adapter: {:?}", adapter.get_info());

        // Request `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` only when the
        // adapter advertises it (Metal/Vulkan/DX12 do). It unlocks the faster
        // Tier2/Native FSR format profile (e.g. `rg16float` storage textures);
        // without it `wgpu-ffx` falls back to the `Core` profile. Requesting a
        // feature the adapter lacks would fail device creation, so we gate on
        // `adapter.features()` for graceful fallback.
        let mut required_features = wgpu::Features::empty();
        if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        {
            required_features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fsr-demo device"),
            required_features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .context("failed to create device")?;

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Ok(Self {
            instance,
            surface,
            adapter,
            device,
            queue,
            config,
        })
    }

    /// The surface texture format, e.g. for building render pipelines.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Resize the surface to the new window dimensions. A zero dimension is
    /// ignored (the window is minimized).
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }
}
