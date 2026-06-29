//! Procedural showcase scene: loads the bundled glTF models, normalizes them to
//! a common world scale, composes a small arrangement designed to stress
//! anti-aliasing, and animates two of the objects.
//!
//! The scene owns the loaded models, a procedural ground plane, the per-material
//! bind groups, and the per-frame instance transforms (current + previous, for
//! motion vectors). [`Scene::update`] advances animation and recomputes
//! transforms; [`Scene::draw_data`] flattens everything into the
//! [`InstanceData`] + [`DrawItem`] lists the renderer consumes.

use std::path::PathBuf;

use anyhow::{Context as _, Result};
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt as _;

use crate::assets::{AlphaMode, Aabb, GpuTexture, LoadedModel, Material, Primitive, Vertex};
use crate::gltf_loader::load_glb;
use crate::renderer::{DrawItem, InstanceData, Renderer};

/// Target world-space height each model is normalized to (its longest axis is
/// scaled to roughly this many units).
const TARGET_SIZE: f32 = 2.0;

/// A loaded model plus its precomputed normalization transform and per-material
/// bind groups.
struct SceneModel {
    model: LoadedModel,
    /// Uniform scale + ground-sitting translation fitting the model to
    /// [`TARGET_SIZE`] with its base at `y = 0`.
    normalize: Mat4,
    /// One bind group per material (parallel to `model.materials`).
    material_bind_groups: Vec<wgpu::BindGroup>,
    /// Keep material uniform buffers alive for the lifetime of the scene.
    _material_buffers: Vec<wgpu::Buffer>,
}

/// How an instance animates over scene time.
#[derive(Clone, Copy)]
enum Animation {
    /// No motion; previous == current.
    Static,
    /// Orbit a center point at `radius`, `speed` rad/s, while spinning to face
    /// travel; also bobs vertically.
    Orbit {
        center: Vec3,
        radius: f32,
        speed: f32,
    },
    /// Spin in place about +Y at `speed` rad/s and bob vertically.
    SpinBob {
        speed: f32,
        bob_amplitude: f32,
        bob_speed: f32,
    },
}

/// One placed, possibly animated object referencing a [`SceneModel`].
struct Instance {
    /// Index into [`Scene::models`].
    model_index: usize,
    /// Static world placement applied on top of the model's normalization.
    placement: Mat4,
    /// Animation behavior.
    animation: Animation,
    /// Current-frame model matrix (`placement * anim * normalize`).
    current_model: Mat4,
    /// Previous-frame model matrix (for motion vectors).
    previous_model: Mat4,
}

/// The full showcase scene.
pub struct Scene {
    models: Vec<SceneModel>,
    instances: Vec<Instance>,
    /// Accumulated animation time in seconds (frozen while paused).
    time: f32,
}

impl Scene {
    /// Load the bundled models, build the ground plane, normalize + place
    /// everything, and create all material bind groups.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, renderer: &mut Renderer) -> Result<Self> {
        // --- ground plane (procedural model, index 0) ---
        let ground = build_ground_plane(device);
        let mut models = vec![scene_model_from(device, renderer, ground)];

        // --- bundled glTF models (indices 1..=4) ---
        let names = [
            "MetalRoughSpheres.glb",
            "DamagedHelmet.glb",
            "Duck.glb",
            "Avocado.glb",
        ];
        for name in names {
            let loaded = load_glb(device, queue, &asset_path(name))
                .with_context(|| format!("failed to load scene model {name}"))?;
            models.push(scene_model_from(device, renderer, loaded));
        }

        // Stable indices into `models`.
        const GROUND: usize = 0;
        const SPHERES: usize = 1;
        const HELMET: usize = 2;
        const DUCK: usize = 3;
        const AVOCADO: usize = 4;

        // --- instances ---
        // The ground plane is already world-sized (no normalization), so its
        // placement is identity. Everything else is normalized to TARGET_SIZE.
        let instances = vec![
            // Ground plane, flat at y=0.
            Instance::new(GROUND, Mat4::IDENTITY, Animation::Static),
            // Hero: damaged helmet, centered, lifted to eye height. Kept static;
            // its fine normal-mapped detail is the AA showcase.
            Instance::new(
                HELMET,
                Mat4::from_translation(Vec3::new(0.0, 1.5, 0.0)),
                Animation::Static,
            ),
            // Metal-rough spheres: prominent to the left; sharp specular
            // highlights are the key temporal-AA showcase.
            Instance::new(
                SPHERES,
                Mat4::from_translation(Vec3::new(-3.5, 1.0, -0.5))
                    * Mat4::from_scale(Vec3::splat(1.5)),
                Animation::Static,
            ),
            // Duck: orbits the helmet.
            Instance::new(
                DUCK,
                Mat4::IDENTITY,
                Animation::Orbit {
                    center: Vec3::new(0.0, 1.0, 0.0),
                    radius: 4.0,
                    speed: 0.6,
                },
            ),
            // Avocado: spins in place and bobs, to the right.
            Instance::new(
                AVOCADO,
                Mat4::from_translation(Vec3::new(3.0, 1.2, 0.5)),
                Animation::SpinBob {
                    speed: 1.2,
                    bob_amplitude: 0.4,
                    bob_speed: 1.5,
                },
            ),
        ];

        let mut scene = Self {
            models,
            instances,
            time: 0.0,
        };
        // Initialize current == previous so the first frame has zero motion.
        scene.recompute_transforms();
        for inst in &mut scene.instances {
            inst.previous_model = inst.current_model;
        }
        Ok(scene)
    }

    /// Advance animation by `dt` seconds (no-op when `paused`), rolling the
    /// previous-frame transforms forward first so motion vectors are correct.
    pub fn update(&mut self, dt: f32, paused: bool) {
        for inst in &mut self.instances {
            inst.previous_model = inst.current_model;
        }
        if !paused {
            self.time += dt;
        }
        self.recompute_transforms();
    }

    /// Recompute every instance's `current_model` from the current scene time.
    fn recompute_transforms(&mut self) {
        for inst in &mut self.instances {
            let normalize = self.models[inst.model_index].normalize;
            let anim = match inst.animation {
                Animation::Static => Mat4::IDENTITY,
                Animation::Orbit {
                    center,
                    radius,
                    speed,
                } => {
                    let angle = self.time * speed;
                    let pos = center
                        + Vec3::new(angle.cos() * radius, 0.0, angle.sin() * radius);
                    let bob = (self.time * 1.3).sin() * 0.25;
                    // Face the direction of travel (tangent to the orbit).
                    let facing = Quat::from_rotation_y(-angle + std::f32::consts::FRAC_PI_2);
                    Mat4::from_translation(pos + Vec3::new(0.0, bob, 0.0))
                        * Mat4::from_quat(facing)
                }
                Animation::SpinBob {
                    speed,
                    bob_amplitude,
                    bob_speed,
                } => {
                    let bob = (self.time * bob_speed).sin() * bob_amplitude;
                    Mat4::from_translation(Vec3::new(0.0, bob, 0.0))
                        * Mat4::from_rotation_y(self.time * speed)
                }
            };
            inst.current_model = inst.placement * anim * normalize;
        }
    }

    /// Flatten the scene into the renderer's instance + draw lists.
    ///
    /// One `InstanceData` entry and one `DrawItem` are produced *per primitive*,
    /// kept 1:1 so `DrawItem::instance_index` indexes straight into the
    /// returned `InstanceData`. The primitive's in-model node transform
    /// (`local_transform`) is folded into the instance matrix here, since the
    /// loader bakes the model's bounds in that same post-`local_transform`
    /// space that the normalization assumes.
    pub fn draw_data<'a>(&'a self) -> (Vec<InstanceData>, Vec<DrawItem<'a>>) {
        let mut instance_data = Vec::new();
        let mut draws = Vec::new();

        for inst in &self.instances {
            let scene_model = &self.models[inst.model_index];
            for prim in &scene_model.model.primitives {
                let material = &scene_model.model.materials[prim.material];
                let index = instance_data.len() as u32;
                instance_data.push(InstanceData {
                    model: inst.current_model * prim.local_transform,
                    prev_model: inst.previous_model * prim.local_transform,
                });
                draws.push(DrawItem {
                    vertex_buffer: &prim.vertex_buffer,
                    index_buffer: &prim.index_buffer,
                    index_count: prim.index_count,
                    material_bind_group: &scene_model.material_bind_groups[prim.material],
                    instance_index: index,
                    double_sided: material.double_sided,
                });
            }
        }

        (instance_data, draws)
    }
}

impl Instance {
    fn new(model_index: usize, placement: Mat4, animation: Animation) -> Self {
        Self {
            model_index,
            placement,
            animation,
            current_model: Mat4::IDENTITY,
            previous_model: Mat4::IDENTITY,
        }
    }
}

/// Build a [`SceneModel`] from a loaded model: compute its normalization
/// transform and create one material bind group per material.
fn scene_model_from(
    device: &wgpu::Device,
    renderer: &mut Renderer,
    model: LoadedModel,
) -> SceneModel {
    let normalize = normalize_transform(&model.bounds);

    let mut material_bind_groups = Vec::with_capacity(model.materials.len());
    let mut material_buffers = Vec::with_capacity(model.materials.len());
    for material in &model.materials {
        let (bg, buf) = renderer.create_material_bind_group(device, &model, material);
        material_bind_groups.push(bg);
        material_buffers.push(buf);
    }

    SceneModel {
        model,
        normalize,
        material_bind_groups,
        _material_buffers: material_buffers,
    }
}

/// Compute a uniform scale + translation that fits `bounds` to [`TARGET_SIZE`]
/// along its longest axis and sits its base on the `y = 0` plane, centered in
/// X/Z. Degenerate/already-world-sized bounds (the ground plane) collapse to
/// identity-ish behavior because their longest extent equals TARGET handling.
fn normalize_transform(bounds: &Aabb) -> Mat4 {
    let size = bounds.size();
    let longest = size.x.max(size.y).max(size.z);
    if !longest.is_finite() || longest <= 1e-6 {
        return Mat4::IDENTITY;
    }
    let scale = TARGET_SIZE / longest;
    let center = bounds.center();
    // After scaling about the origin, the model's center sits at `center *
    // scale`. Translate so X/Z center to 0 and the box base (min.y) sits at 0.
    let scaled_min_y = bounds.min.y * scale;
    let translation = Vec3::new(
        -center.x * scale,
        -scaled_min_y,
        -center.z * scale,
    );
    Mat4::from_translation(translation) * Mat4::from_scale(Vec3::splat(scale))
}

/// Build a large procedural ground plane as a one-primitive [`LoadedModel`].
///
/// A subdivided grid spanning `[-EXTENT, EXTENT]` in X/Z at `y = 0`, with
/// upward normals and a flat mid-gray, semi-rough dielectric material. The
/// returned `bounds` are deliberately a 2x2 footprint centered at the origin so
/// that [`normalize_transform`] resolves to the identity (longest extent ==
/// `TARGET_SIZE`, base already at `y = 0`, centered) and the plane keeps its
/// world-scale geometry.
fn build_ground_plane(device: &wgpu::Device) -> LoadedModel {
    const EXTENT: f32 = 30.0;
    const SUBDIV: u32 = 64;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let step = (2.0 * EXTENT) / SUBDIV as f32;
    // UVs tile so the (shader-side) checker has many cells across the plane.
    let uv_scale = SUBDIV as f32;
    for z in 0..=SUBDIV {
        for x in 0..=SUBDIV {
            let px = -EXTENT + x as f32 * step;
            let pz = -EXTENT + z as f32 * step;
            vertices.push(Vertex {
                position: [px, 0.0, pz],
                normal: [0.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                uv: [
                    x as f32 / SUBDIV as f32 * uv_scale,
                    z as f32 / SUBDIV as f32 * uv_scale,
                ],
            });
        }
    }
    let row = SUBDIV + 1;
    for z in 0..SUBDIV {
        for x in 0..SUBDIV {
            let i0 = z * row + x;
            let i1 = i0 + 1;
            let i2 = i0 + row;
            let i3 = i2 + 1;
            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ground vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ground indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let material = Material {
        base_color_factor: [0.45, 0.45, 0.47, 1.0],
        metallic_factor: 0.0,
        roughness_factor: 0.8,
        emissive_factor: [0.0, 0.0, 0.0],
        normal_scale: 1.0,
        occlusion_strength: 1.0,
        alpha_mode: AlphaMode::Opaque,
        double_sided: false,
        base_color_texture: None,
        metallic_roughness_texture: None,
        normal_texture: None,
        occlusion_texture: None,
        emissive_texture: None,
    };

    LoadedModel {
        primitives: vec![Primitive {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            material: 0,
            local_transform: Mat4::IDENTITY,
        }],
        materials: vec![material],
        textures: Vec::<GpuTexture>::new(),
        // Footprint chosen so `normalize_transform` yields the identity (see the
        // doc comment): longest extent == TARGET_SIZE, base at y=0, centered.
        bounds: Aabb {
            min: Vec3::new(-TARGET_SIZE * 0.5, 0.0, -TARGET_SIZE * 0.5),
            max: Vec3::new(TARGET_SIZE * 0.5, 0.0, TARGET_SIZE * 0.5),
        },
    }
}

/// Absolute path to a bundled asset, resolved relative to the crate manifest so
/// it works regardless of the current working directory.
fn asset_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join(name)
}
