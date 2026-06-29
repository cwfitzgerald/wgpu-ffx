//! Shared settings contract.
//!
//! [`Settings`] is the single source of truth for user-controllable demo
//! parameters. The egui overlay mutates it, and later phases (camera, scene
//! animation, FSR integration) read from it. Keep this struct plain-old-data
//! so it stays cheap to copy and easy to reason about across modules.

/// FSR quality preset, selecting the ratio between render and display
/// resolution.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QualityMode {
    /// Native resolution anti-aliasing (no upscaling, 1.0x).
    NativeAa,
    /// 1.5x upscale.
    Quality,
    /// 1.7x upscale.
    Balanced,
    /// 2.0x upscale.
    Performance,
    /// 3.0x upscale.
    UltraPerformance,
}

impl QualityMode {
    /// Every quality mode, in UI display order.
    pub const ALL: [QualityMode; 5] = [
        Self::NativeAa,
        Self::Quality,
        Self::Balanced,
        Self::Performance,
        Self::UltraPerformance,
    ];

    /// FSR scale factor (`display_dim / render_dim`).
    pub fn scale_factor(self) -> f32 {
        match self {
            Self::NativeAa => 1.0,
            Self::Quality => 1.5,
            Self::Balanced => 1.7,
            Self::Performance => 2.0,
            Self::UltraPerformance => 3.0,
        }
    }

    /// Human-readable label for the UI, e.g. `"Quality (1.5x)"`.
    pub fn label(self) -> &'static str {
        match self {
            Self::NativeAa => "Native AA (1.0x)",
            Self::Quality => "Quality (1.5x)",
            Self::Balanced => "Balanced (1.7x)",
            Self::Performance => "Performance (2.0x)",
            Self::UltraPerformance => "Ultra Performance (3.0x)",
        }
    }

    /// Render resolution for a given display resolution.
    ///
    /// Each dimension is `round(display / scale_factor)`, clamped to a minimum
    /// of 1 so the result is always a valid texture extent.
    pub fn render_size(self, display: [u32; 2]) -> [u32; 2] {
        let scale = self.scale_factor();
        let dim = |d: u32| ((d as f32 / scale).round() as u32).max(1);
        [dim(display[0]), dim(display[1])]
    }
}

/// User-controllable demo settings, mutated by the egui overlay and consumed by
/// the renderer in later phases.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Active FSR quality preset.
    pub quality: QualityMode,
    /// Whether FSR upscaling is enabled at all.
    pub fsr_enabled: bool,
    /// Whether RCAS sharpening is applied.
    pub sharpening: bool,
    /// Sharpening strength, clamped to `0.0..=1.0`.
    pub sharpness: f32,
    /// Camera movement speed in world units per second.
    pub camera_speed: f32,
    /// Pause procedural object animation.
    pub paused: bool,
    /// Show the stats overlay (FPS / resolution).
    pub show_stats: bool,
    /// One-shot request to reset FSR temporal accumulation. The overlay sets it
    /// when the user clicks "Reset accumulation"; the app consumes it (turning
    /// it into a `reset_history` dispatch) and clears it the same frame.
    pub reset_accumulation: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            quality: QualityMode::Quality,
            fsr_enabled: true,
            sharpening: true,
            sharpness: 0.5,
            camera_speed: 5.0,
            paused: false,
            show_stats: true,
            reset_accumulation: false,
        }
    }
}
