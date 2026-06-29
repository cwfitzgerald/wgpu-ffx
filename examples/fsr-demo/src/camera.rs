//! Free-fly camera with mouse-look and WASD movement.
//!
//! [`Camera`] holds the view/projection state and accumulates input that the
//! app routes from winit `window_event` (keyboard, mouse button) and
//! `device_event` (raw mouse motion). [`Camera::update`] consumes that
//! accumulated input once per frame to advance the position and orientation.
//!
//! ## Conventions (phase 4 / FSR depends on these)
//! - Right-handed view space, wgpu clip space (depth `0..1`) via
//!   [`glam::Mat4::perspective_rh`]. Standard (non-inverted, finite) depth.
//! - `znear = 0.1`, `zfar = 1000.0`. Phase 4 passes these to FSR as
//!   `camera_near` / `camera_far` and must NOT set DEPTH_INVERTED / INFINITE.
//! - Jitter is applied to the projection only via [`Camera::jittered_proj`];
//!   motion vectors are always computed from the *unjittered* matrices.

use glam::{Mat4, Vec3};
use winit::event::{ElementState, MouseButton};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Near clip plane shared by the camera and (in phase 4) FSR's `camera_near`.
pub const Z_NEAR: f32 = 0.1;
/// Far clip plane shared by the camera and (in phase 4) FSR's `camera_far`.
pub const Z_FAR: f32 = 1000.0;

/// Maximum pitch magnitude, just under straight up/down to avoid gimbal flip.
const PITCH_LIMIT: f32 = 89.0_f32.to_radians();
/// Radians of yaw/pitch per pixel of mouse motion while looking.
const LOOK_SENSITIVITY: f32 = 0.0025;
/// Movement speed multiplier while the boost (Shift) key is held.
const BOOST_MULTIPLIER: f32 = 4.0;

/// A free-fly perspective camera.
pub struct Camera {
    /// World-space eye position.
    pub position: Vec3,
    /// Yaw in radians (rotation about world +Y); 0 looks down -Z.
    pub yaw: f32,
    /// Pitch in radians, clamped to +/-[`PITCH_LIMIT`].
    pub pitch: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clip distance.
    pub znear: f32,
    /// Far clip distance.
    pub zfar: f32,

    /// Held-key movement state, consumed each [`Camera::update`].
    keys: KeyState,
    /// Accumulated raw mouse-look delta in pixels `(dx, dy)`, applied and
    /// cleared each [`Camera::update`].
    look_delta: (f32, f32),
    /// Whether the right mouse button is currently held (look mode active).
    pub looking: bool,
}

/// Tracks which movement keys are currently held.
#[derive(Default)]
struct KeyState {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    boost: bool,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 2.0, 8.0),
            // Look slightly down toward the scene center.
            yaw: 0.0,
            pitch: -0.15,
            fov_y: 60.0_f32.to_radians(),
            znear: Z_NEAR,
            zfar: Z_FAR,
            keys: KeyState::default(),
            look_delta: (0.0, 0.0),
            looking: false,
        }
    }
}

impl Camera {
    /// The unit forward direction implied by the current yaw/pitch.
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        // yaw=0, pitch=0 -> -Z. +yaw rotates toward +X.
        Vec3::new(sy * cp, sp, -cy * cp).normalize()
    }

    /// The view matrix (world -> view).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), Vec3::Y)
    }

    /// The projection matrix for the given aspect ratio (view -> clip, depth
    /// `0..1`).
    pub fn proj_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect.max(1e-4), self.znear, self.zfar)
    }

    /// The combined unjittered view-projection matrix.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj_matrix(aspect) * self.view_matrix()
    }

    /// Projection with a sub-pixel jitter offset baked into its third column.
    ///
    /// `jitter_px` is the desired pixel-space offset of the sample point, and
    /// `render_size` is the render-target resolution the image is rasterized
    /// at. The combined jittered view-proj is `jittered_proj(..) * view_matrix()`.
    ///
    /// The sign pairing below (`+x`, `-y`) is the FSR convention. If phase-5
    /// verification shows the image fails to converge or smears, THIS is the
    /// single place to flip the signs.
    pub fn jittered_proj(&self, aspect: f32, jitter_px: [f32; 2], render_size: [u32; 2]) -> Mat4 {
        let mut proj = self.proj_matrix(aspect);
        let render_w = render_size[0].max(1) as f32;
        let render_h = render_size[1].max(1) as f32;
        proj.z_axis.x += 2.0 * jitter_px[0] / render_w;
        proj.z_axis.y -= 2.0 * jitter_px[1] / render_h;
        proj
    }

    /// Advance the camera by `dt` seconds using `speed` world units/sec,
    /// consuming accumulated mouse-look and applying held-key movement.
    pub fn update(&mut self, dt: f32, speed: f32) {
        // Apply accumulated look delta (radians per pixel).
        self.yaw += self.look_delta.0 * LOOK_SENSITIVITY;
        self.pitch -= self.look_delta.1 * LOOK_SENSITIVITY;
        self.pitch = self.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
        self.look_delta = (0.0, 0.0);

        // Movement basis: forward (full 3D) and a horizontal right vector.
        let forward = self.forward();
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);

        let mut dir = Vec3::ZERO;
        if self.keys.forward {
            dir += forward;
        }
        if self.keys.back {
            dir -= forward;
        }
        if self.keys.right {
            dir += right;
        }
        if self.keys.left {
            dir -= right;
        }
        if self.keys.up {
            dir += Vec3::Y;
        }
        if self.keys.down {
            dir -= Vec3::Y;
        }

        if dir != Vec3::ZERO {
            let boost = if self.keys.boost {
                BOOST_MULTIPLIER
            } else {
                1.0
            };
            self.position += dir.normalize() * speed * boost * dt;
        }
    }

    /// Feed a keyboard event. Release events are always honoured (so keys never
    /// "stick" when egui or look-mode swallows the press).
    pub fn on_key(&mut self, key: PhysicalKey, state: ElementState) {
        let pressed = state == ElementState::Pressed;
        let PhysicalKey::Code(code) = key else {
            return;
        };
        match code {
            KeyCode::KeyW => self.keys.forward = pressed,
            KeyCode::KeyS => self.keys.back = pressed,
            KeyCode::KeyA => self.keys.left = pressed,
            KeyCode::KeyD => self.keys.right = pressed,
            KeyCode::KeyE | KeyCode::Space => self.keys.up = pressed,
            KeyCode::KeyQ | KeyCode::ControlLeft | KeyCode::ControlRight => {
                self.keys.down = pressed
            }
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.keys.boost = pressed,
            _ => {}
        }
    }

    /// Feed a mouse-button event. The right button toggles look mode.
    pub fn on_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Right {
            self.looking = state == ElementState::Pressed;
        }
    }

    /// Feed a raw mouse-motion delta (from `DeviceEvent::MouseMotion`).
    /// Accumulated only while look mode is active.
    pub fn on_mouse_motion(&mut self, dx: f64, dy: f64) {
        if self.looking {
            self.look_delta.0 += dx as f32;
            self.look_delta.1 += dy as f32;
        }
    }

    /// Drop all held movement keys (e.g. on focus loss) so the camera stops.
    pub fn release_all_keys(&mut self) {
        self.keys = KeyState::default();
    }
}
