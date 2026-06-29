//! Forward PBR renderer: geometry MRT pass + temporary tonemap composite.
//!
//! The renderer owns the render-resolution targets (HDR color, depth, motion
//! vectors), the GPU pipelines, the per-material bind groups, and the frame /
//! instance uniform buffers. Each frame the app calls [`Renderer::render`] with
//! the current [`Frame`] data; the renderer records:
//!
//! 1. **Geometry pass** — MRT into HDR color (location 0) + motion vectors
//!    (location 1) with a depth attachment, rasterized with the *jittered*
//!    view-proj but writing motion vectors from the *unjittered* matrices.
//! 2. **Composite pass** — a fullscreen tonemap of the HDR target onto the sRGB
//!    surface (bilinear upscale when render size < display size). This is the
//!    future "FSR OFF" path; phase 4 keeps it and adds an FSR path alongside.
//!
//! ## Render target formats — MUST match what FSR requires
//! Phase 4 will assert these equal `wgpu_ffx::FsrContext::formats()`.
//! All three live at *render* resolution with `RENDER_ATTACHMENT |
//! TEXTURE_BINDING` (FSR samples them).

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use wgpu::util::DeviceExt as _;

use crate::assets::{AlphaMode, LoadedModel, Material, SamplerInfo};

/// HDR scene color format. MUST equal `wgpu_ffx::FsrContext::formats()` color.
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Depth format (standard non-inverted finite depth). MUST equal FSR's depth.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Motion-vector format. MUST equal `wgpu_ffx::FsrContext::formats()` MV.
pub const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// Depth clear value. Standard depth: 1.0 = far, compare `Less`.
const DEPTH_CLEAR: f32 = 1.0;

// ---------------------------------------------------------------------------
// GPU uniform layouts (Rust <-> WGSL). All structs are `#[repr(C)]` and padded
// to keep `vec3`/`mat4x4` alignment identical to the WGSL `struct` layout
// (vec3 aligns to 16; mat4x4 to 16). Phase 4 must keep these in lockstep with
// `pbr.wgsl` when adding FSR-related fields.
// ---------------------------------------------------------------------------

/// Per-frame camera + lighting uniform (`@group(0) @binding(0)`).
///
/// WGSL `struct FrameUniform` in `pbr.wgsl` mirrors this field-for-field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    /// Current jittered view-proj, used for `@builtin(position)`.
    view_proj_jittered: [[f32; 4]; 4],
    /// Current unjittered view-proj, used for motion vectors.
    view_proj: [[f32; 4]; 4],
    /// Previous-frame unjittered view-proj, used for motion vectors.
    prev_view_proj: [[f32; 4]; 4],
    /// Camera world position (`.xyz`); `.w` padding.
    camera_pos: [f32; 4],
    /// Direction *toward* the sun (`.xyz`, normalized); `.w` padding.
    sun_direction: [f32; 4],
    /// Sun radiance: color in `.xyz`, intensity scalar in `.w`.
    sun_color: [f32; 4],
    /// Hemisphere ambient sky color (`.xyz`); `.w` padding.
    sky_color: [f32; 4],
    /// Hemisphere ambient ground color (`.xyz`); `.w` padding.
    ground_color: [f32; 4],
    /// Render-target size in pixels (`.xy`); `.zw` padding.
    render_size: [f32; 4],
}

/// Per-instance transforms (`@group(2) @binding(0)`, storage array indexed by
/// `@builtin(instance_index)`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceUniform {
    /// Current model matrix.
    model: [[f32; 4]; 4],
    /// Previous-frame model matrix (for motion vectors).
    prev_model: [[f32; 4]; 4],
    /// Normal matrix (inverse-transpose of the model upper 3x3), stored as a
    /// `mat4x4` so each column is 16-byte aligned. Only the upper-left 3x3 is
    /// meaningful; the rest is identity padding.
    normal_matrix: [[f32; 4]; 4],
}

/// Per-material factors + flags (`@group(1) @binding(0)`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialUniform {
    /// Linear base-color multiplier (RGBA).
    base_color_factor: [f32; 4],
    /// Emissive color (`.xyz`) and metallic factor (`.w`).
    emissive_metallic: [f32; 4],
    /// `[roughness_factor, normal_scale, occlusion_strength, alpha_cutoff]`.
    rough_normal_occ_cutoff: [f32; 4],
    /// `[texture_present_flags, alpha_mode, pad, pad]` as `u32`.
    flags_alpha: [u32; 4],
}

// Texture-present bit flags packed into `MaterialUniform::flags_alpha[0]`.
const FLAG_BASE_COLOR_TEX: u32 = 1 << 0;
const FLAG_MR_TEX: u32 = 1 << 1;
const FLAG_NORMAL_TEX: u32 = 1 << 2;
const FLAG_OCCLUSION_TEX: u32 = 1 << 3;
const FLAG_EMISSIVE_TEX: u32 = 1 << 4;

// Alpha-mode discriminants packed into `MaterialUniform::flags_alpha[1]`.
const ALPHA_OPAQUE: u32 = 0;
const ALPHA_MASK: u32 = 1;
// Blend is treated as opaque for this demo (noted in the report).

/// Render-resolution targets sampled later by the composite pass (and, in phase
/// 4, by FSR). Reallocated only when the render size changes.
///
/// The `*_view` fields are what phase 3 binds; the owning `Texture`s
/// (`depth`/`motion`) and the FSR-facing accessors are kept for phase 4 to pass
/// to `wgpu_ffx::FsrContext`, hence the `dead_code` allowance.
#[allow(dead_code)]
pub struct RenderTargets {
    /// Current render resolution `[w, h]`.
    pub size: [u32; 2],
    /// HDR scene color ([`COLOR_FORMAT`]).
    pub color: wgpu::Texture,
    /// HDR color view.
    pub color_view: wgpu::TextureView,
    /// Depth buffer ([`DEPTH_FORMAT`]).
    pub depth: wgpu::Texture,
    /// Depth view.
    pub depth_view: wgpu::TextureView,
    /// Motion vectors ([`MOTION_FORMAT`]).
    pub motion: wgpu::Texture,
    /// Motion-vector view.
    pub motion_view: wgpu::TextureView,
}

impl RenderTargets {
    /// Allocate all three targets at `size` render resolution.
    fn new(device: &wgpu::Device, size: [u32; 2]) -> Self {
        let make = |label: &str, format: wgpu::TextureFormat| {
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
                // FSR samples these later, hence TEXTURE_BINDING.
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };

        let color = make("hdr color", COLOR_FORMAT);
        let depth = make("scene depth", DEPTH_FORMAT);
        let motion = make("motion vectors", MOTION_FORMAT);

        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let motion_view = motion.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            size,
            color,
            color_view,
            depth,
            depth_view,
            motion,
            motion_view,
        }
    }
}

/// A directional "sun" plus hemisphere-ambient lighting environment.
#[derive(Clone, Copy)]
pub struct Lighting {
    /// Direction the sunlight travels *from* the surface toward the sun.
    pub sun_direction: Vec3,
    /// Sun color (linear).
    pub sun_color: Vec3,
    /// Sun intensity multiplier.
    pub sun_intensity: f32,
    /// Ambient color from above.
    pub sky_color: Vec3,
    /// Ambient color from below.
    pub ground_color: Vec3,
}

impl Default for Lighting {
    fn default() -> Self {
        Self {
            sun_direction: Vec3::new(0.4, 0.8, 0.45).normalize(),
            sun_color: Vec3::new(1.0, 0.96, 0.9),
            sun_intensity: 3.0,
            sky_color: Vec3::new(0.25, 0.32, 0.45),
            ground_color: Vec3::new(0.12, 0.11, 0.10),
        }
    }
}

/// One draw the renderer should record: a primitive's buffers, its material
/// bind group, the per-instance index, and whether it is double-sided.
pub struct DrawItem<'a> {
    /// Vertex buffer ([`crate::assets::Vertex`] layout).
    pub vertex_buffer: &'a wgpu::Buffer,
    /// `u32` index buffer.
    pub index_buffer: &'a wgpu::Buffer,
    /// Number of indices to draw.
    pub index_count: u32,
    /// Material bind group for `@group(1)`.
    pub material_bind_group: &'a wgpu::BindGroup,
    /// Index into the instance storage buffer (`@builtin(instance_index)`).
    pub instance_index: u32,
    /// Whether to use the cull-none pipeline.
    pub double_sided: bool,
}

/// Everything the renderer needs for one frame.
pub struct Frame<'a> {
    /// Jittered current view-proj (`@builtin(position)` source).
    pub view_proj_jittered: Mat4,
    /// Unjittered current view-proj (motion vectors).
    pub view_proj: Mat4,
    /// Previous-frame unjittered view-proj (motion vectors).
    pub prev_view_proj: Mat4,
    /// Camera world position.
    pub camera_pos: Vec3,
    /// Lighting environment.
    pub lighting: Lighting,
    /// Per-instance transforms, one entry per `instance_index` referenced.
    pub instances: &'a [InstanceData],
    /// The draws to record, in submission order.
    pub draws: &'a [DrawItem<'a>],
}

/// CPU-side per-instance transform data the app fills each frame.
#[derive(Clone, Copy)]
pub struct InstanceData {
    /// Current model matrix.
    pub model: Mat4,
    /// Previous-frame model matrix.
    pub prev_model: Mat4,
}

/// The forward PBR renderer.
pub struct Renderer {
    /// Render-resolution targets, reallocated on size change.
    targets: RenderTargets,

    // Geometry pass.
    frame_buffer: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    instance_bind_group_layout: wgpu::BindGroupLayout,
    instance_bind_group: wgpu::BindGroup,
    material_bind_group_layout: wgpu::BindGroupLayout,
    pipeline_cull_back: wgpu::RenderPipeline,
    pipeline_cull_none: wgpu::RenderPipeline,

    // Composite pass.
    composite_pipeline: wgpu::RenderPipeline,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    composite_sampler: wgpu::Sampler,
    composite_bind_group: wgpu::BindGroup,

    // Shared default 1x1 textures for absent material maps.
    default_white_srgb: wgpu::TextureView,
    default_white_linear: wgpu::TextureView,
    default_normal: wgpu::TextureView,
    default_sampler: wgpu::Sampler,

    /// Cache of per-`SamplerInfo` samplers so repeated material textures reuse.
    sampler_cache: HashMap<SamplerKey, wgpu::Sampler>,
}

/// Hashable key for caching samplers (wgpu enums lack `Hash`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct SamplerKey {
    address_u: u8,
    address_v: u8,
    mag: u8,
    min: u8,
    mip: u8,
}

impl Renderer {
    /// Build the renderer for a given render size and surface format.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        render_size: [u32; 2],
    ) -> Self {
        let targets = RenderTargets::new(device, render_size);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbr shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pbr.wgsl").into()),
        });
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/composite.wgsl").into()),
        });

        // --- group 0: frame uniform -------------------------------------
        let frame_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("frame bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame uniform"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame bg"),
            layout: &frame_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buffer.as_entire_binding(),
            }],
        });

        // --- group 1: material ------------------------------------------
        // binding 0: material uniform
        // bindings 1..=5: base/MR/normal/occlusion/emissive texture views
        // bindings 6..=10: matching samplers
        let mut material_entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for binding in 1..=5 {
            material_entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        for binding in 6..=10 {
            material_entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("material bgl"),
                entries: &material_entries,
            });

        // --- group 2: instance storage ----------------------------------
        let instance_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("instance bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let instance_capacity = 64;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance storage"),
            size: (instance_capacity * std::mem::size_of::<InstanceUniform>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("instance bg"),
            layout: &instance_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buffer.as_entire_binding(),
            }],
        });

        // --- geometry pipelines (cull-back + cull-none) ------------------
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbr pipeline layout"),
            bind_group_layouts: &[
                Some(&frame_bind_group_layout),
                Some(&material_bind_group_layout),
                Some(&instance_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<crate::assets::Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
                0 => Float32x3, // position
                1 => Float32x3, // normal
                2 => Float32x4, // tangent
                3 => Float32x2, // uv
            ],
        };

        let color_targets = [
            // location 0: HDR color
            Some(wgpu::ColorTargetState {
                format: COLOR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            // location 1: motion vectors
            Some(wgpu::ColorTargetState {
                format: MOTION_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];

        let depth_stencil = Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        let make_pipeline = |cull: Option<wgpu::Face>, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: std::slice::from_ref(&vertex_layout),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: cull,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: depth_stencil.clone(),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &color_targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        let pipeline_cull_back = make_pipeline(Some(wgpu::Face::Back), "pbr pipeline (cull back)");
        let pipeline_cull_none = make_pipeline(None, "pbr pipeline (cull none)");

        // --- composite pass ----------------------------------------------
        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("composite bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite pipeline layout"),
            bind_group_layouts: &[Some(&composite_bind_group_layout)],
            immediate_size: 0,
        });
        let composite_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("composite pipeline"),
                layout: Some(&composite_layout),
                vertex: wgpu::VertexState {
                    module: &composite_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &composite_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("composite sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let composite_bind_group = make_composite_bind_group(
            device,
            &composite_bind_group_layout,
            &targets.color_view,
            &composite_sampler,
        );

        // --- shared default textures -------------------------------------
        let default_white_srgb = make_default_texture(
            device,
            queue,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            [255, 255, 255, 255],
            "default white srgb",
        );
        let default_white_linear = make_default_texture(
            device,
            queue,
            wgpu::TextureFormat::Rgba8Unorm,
            [255, 255, 255, 255],
            "default white linear",
        );
        let default_normal = make_default_texture(
            device,
            queue,
            wgpu::TextureFormat::Rgba8Unorm,
            [128, 128, 255, 255],
            "default normal",
        );
        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("default sampler"),
            ..Default::default()
        });

        Self {
            targets,
            frame_buffer,
            frame_bind_group,
            instance_buffer,
            instance_capacity,
            instance_bind_group_layout,
            instance_bind_group,
            material_bind_group_layout,
            pipeline_cull_back,
            pipeline_cull_none,
            composite_pipeline,
            composite_bind_group_layout,
            composite_sampler,
            composite_bind_group,
            default_white_srgb,
            default_white_linear,
            default_normal,
            default_sampler,
            sampler_cache: HashMap::new(),
        }
    }

    /// Current render-target size `[w, h]`. (Phase-4 FSR surface.)
    #[allow(dead_code)]
    pub fn render_size(&self) -> [u32; 2] {
        self.targets.size
    }

    /// Immutable access to the render targets, so phase 4 can hand the color /
    /// depth / motion textures and views to FSR. (Phase-4 FSR surface.)
    #[allow(dead_code)]
    pub fn targets(&self) -> &RenderTargets {
        &self.targets
    }

    /// Reallocate render targets if `size` differs, and re-point the composite
    /// bind group at the new HDR color view.
    pub fn resize(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        if self.targets.size == size {
            return;
        }
        self.targets = RenderTargets::new(device, size);
        let color_view = self.targets.color_view_clone();
        self.set_composite_source(device, &color_view);
    }

    /// Re-point the composite pass at an arbitrary HDR-ish source view.
    ///
    /// Phase 4 calls this to swap the composite input between the raw HDR
    /// target (FSR off) and the FSR upscaled output (FSR on). For phase 3 it is
    /// only used internally on resize.
    pub fn set_composite_source(&mut self, device: &wgpu::Device, view: &wgpu::TextureView) {
        self.composite_bind_group = make_composite_bind_group(
            device,
            &self.composite_bind_group_layout,
            view,
            &self.composite_sampler,
        );
    }

    /// Build a material bind group for one [`Material`] of a [`LoadedModel`],
    /// resolving textures (or defaults) and a uniform buffer. The renderer owns
    /// no per-model state, so the caller stores the returned bind group +
    /// buffer alongside its model.
    pub fn create_material_bind_group(
        &mut self,
        device: &wgpu::Device,
        model: &LoadedModel,
        material: &Material,
    ) -> (wgpu::BindGroup, wgpu::Buffer) {
        let mut flags = 0u32;
        if material.base_color_texture.is_some() {
            flags |= FLAG_BASE_COLOR_TEX;
        }
        if material.metallic_roughness_texture.is_some() {
            flags |= FLAG_MR_TEX;
        }
        if material.normal_texture.is_some() {
            flags |= FLAG_NORMAL_TEX;
        }
        if material.occlusion_texture.is_some() {
            flags |= FLAG_OCCLUSION_TEX;
        }
        if material.emissive_texture.is_some() {
            flags |= FLAG_EMISSIVE_TEX;
        }
        let (alpha_mode, cutoff) = match material.alpha_mode {
            AlphaMode::Opaque | AlphaMode::Blend => (ALPHA_OPAQUE, 0.0),
            AlphaMode::Mask { cutoff } => (ALPHA_MASK, cutoff),
        };

        let uniform = MaterialUniform {
            base_color_factor: material.base_color_factor,
            emissive_metallic: [
                material.emissive_factor[0],
                material.emissive_factor[1],
                material.emissive_factor[2],
                material.metallic_factor,
            ],
            rough_normal_occ_cutoff: [
                material.roughness_factor,
                material.normal_scale,
                material.occlusion_strength,
                cutoff,
            ],
            flags_alpha: [flags, alpha_mode, 0, 0],
        };
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Phase 1 (mutable): make sure every sampler this material needs is in
        // the cache. We record the resolution as `(texture_index, sampler_key)`
        // so phase 2 can borrow `self` immutably.
        let slots = [
            (material.base_color_texture, DefaultTex::WhiteSrgb),
            (material.metallic_roughness_texture, DefaultTex::WhiteLinear),
            (material.normal_texture, DefaultTex::Normal),
            (material.occlusion_texture, DefaultTex::WhiteLinear),
            (material.emissive_texture, DefaultTex::WhiteSrgb),
        ];
        let resolved: Vec<ResolvedSlot> = slots
            .iter()
            .map(|&(index, default)| match index {
                Some(i) => {
                    let key = self.ensure_sampler(device, model.textures[i].sampler);
                    ResolvedSlot::Texture { index: i, key }
                }
                None => ResolvedSlot::Default(default),
            })
            .collect();

        // Phase 2 (immutable): gather the actual view + sampler references.
        let pair = |slot: &ResolvedSlot| -> (&wgpu::TextureView, &wgpu::Sampler) {
            match *slot {
                ResolvedSlot::Texture { index, key } => {
                    (&model.textures[index].view, &self.sampler_cache[&key])
                }
                ResolvedSlot::Default(default) => {
                    let view = match default {
                        DefaultTex::WhiteSrgb => &self.default_white_srgb,
                        DefaultTex::WhiteLinear => &self.default_white_linear,
                        DefaultTex::Normal => &self.default_normal,
                    };
                    (view, &self.default_sampler)
                }
            }
        };
        let views: Vec<(&wgpu::TextureView, &wgpu::Sampler)> =
            resolved.iter().map(pair).collect();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material bg"),
            layout: &self.material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(views[0].0) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(views[1].0) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(views[2].0) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(views[3].0) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(views[4].0) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(views[0].1) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(views[1].1) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(views[2].1) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(views[3].1) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(views[4].1) },
            ],
        });

        (bind_group, buffer)
    }

    /// Ensure a [`wgpu::Sampler`] matching `info` exists in the cache, returning
    /// its key so the caller can look it up later via `self.sampler_cache`.
    fn ensure_sampler(&mut self, device: &wgpu::Device, info: SamplerInfo) -> SamplerKey {
        let key = SamplerKey {
            address_u: info.address_mode_u as u8,
            address_v: info.address_mode_v as u8,
            mag: info.mag_filter as u8,
            min: info.min_filter as u8,
            mip: info.mipmap_filter as u8,
        };
        self.sampler_cache.entry(key).or_insert_with(|| {
            // `SamplerInfo::mipmap_filter` is a `FilterMode`; wgpu 29 wants a
            // dedicated `MipmapFilterMode`.
            let mipmap_filter = match info.mipmap_filter {
                wgpu::FilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
                wgpu::FilterMode::Linear => wgpu::MipmapFilterMode::Linear,
            };
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("material sampler"),
                address_mode_u: info.address_mode_u,
                address_mode_v: info.address_mode_v,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: info.mag_filter,
                min_filter: info.min_filter,
                mipmap_filter,
                ..Default::default()
            })
        });
        key
    }

    /// Record the geometry (MRT) pass for one frame into the render-resolution
    /// targets (HDR color + motion vectors + depth).
    ///
    /// This is the first half of the former `render`: it uploads the frame and
    /// instance uniforms and clears + draws the scene. The FSR dispatch (when
    /// enabled) is recorded on the same encoder *after* this and *before*
    /// [`Renderer::composite`], which then tonemaps the chosen composite source
    /// (raw HDR or FSR output) onto the surface.
    pub fn render_geometry(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &Frame<'_>,
    ) {
        // Upload the frame uniform.
        let render_size = self.targets.size;
        let frame_uniform = FrameUniform {
            view_proj_jittered: frame.view_proj_jittered.to_cols_array_2d(),
            view_proj: frame.view_proj.to_cols_array_2d(),
            prev_view_proj: frame.prev_view_proj.to_cols_array_2d(),
            camera_pos: [frame.camera_pos.x, frame.camera_pos.y, frame.camera_pos.z, 1.0],
            sun_direction: [
                frame.lighting.sun_direction.x,
                frame.lighting.sun_direction.y,
                frame.lighting.sun_direction.z,
                0.0,
            ],
            sun_color: [
                frame.lighting.sun_color.x,
                frame.lighting.sun_color.y,
                frame.lighting.sun_color.z,
                frame.lighting.sun_intensity,
            ],
            sky_color: [
                frame.lighting.sky_color.x,
                frame.lighting.sky_color.y,
                frame.lighting.sky_color.z,
                0.0,
            ],
            ground_color: [
                frame.lighting.ground_color.x,
                frame.lighting.ground_color.y,
                frame.lighting.ground_color.z,
                0.0,
            ],
            render_size: [render_size[0] as f32, render_size[1] as f32, 0.0, 0.0],
        };
        queue.write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(&frame_uniform));

        // Upload instances, growing the buffer if needed.
        self.ensure_instance_capacity(device, frame.instances.len());
        let instance_data: Vec<InstanceUniform> = frame
            .instances
            .iter()
            .map(|inst| InstanceUniform {
                model: inst.model.to_cols_array_2d(),
                prev_model: inst.prev_model.to_cols_array_2d(),
                normal_matrix: normal_matrix(inst.model),
            })
            .collect();
        if !instance_data.is_empty() {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&instance_data),
            );
        }

        // --- geometry pass ---
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("geometry"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.targets.color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.targets.motion_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // Clear motion to zero (static background).
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(DEPTH_CLEAR),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            pass.set_bind_group(2, &self.instance_bind_group, &[]);

            let mut current_double_sided: Option<bool> = None;
            for draw in frame.draws {
                if current_double_sided != Some(draw.double_sided) {
                    pass.set_pipeline(if draw.double_sided {
                        &self.pipeline_cull_none
                    } else {
                        &self.pipeline_cull_back
                    });
                    current_double_sided = Some(draw.double_sided);
                }
                pass.set_bind_group(1, draw.material_bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                pass.set_index_buffer(draw.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                // Encode the instance index via the instance range so the
                // vertex shader reads `instances[instance_index]`.
                pass.draw_indexed(
                    0..draw.index_count,
                    0,
                    draw.instance_index..draw.instance_index + 1,
                );
            }
        }
    }

    /// Record the composite (fullscreen tonemap) pass onto `surface_view`.
    ///
    /// Tonemaps the current composite source — set via
    /// [`Renderer::set_composite_source`] to either the raw HDR color target
    /// (FSR off) or the FSR upscaled output (FSR on) — onto the sRGB surface.
    /// egui is recorded afterward by the caller.
    pub fn composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        surface_view: &wgpu::TextureView,
    ) {
        // --- composite pass ---
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &self.composite_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Grow the instance storage buffer (and its bind group) to hold `count`
    /// instances if the current capacity is insufficient.
    fn ensure_instance_capacity(&mut self, device: &wgpu::Device, count: usize) {
        if count <= self.instance_capacity {
            return;
        }
        let new_capacity = count.next_power_of_two().max(self.instance_capacity * 2);
        self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance storage"),
            size: (new_capacity * std::mem::size_of::<InstanceUniform>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("instance bg"),
            layout: &self.instance_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.instance_buffer.as_entire_binding(),
            }],
        });
        self.instance_capacity = new_capacity;
    }
}

impl RenderTargets {
    /// Re-create the HDR color view (used when re-pointing the composite bind
    /// group, which would otherwise borrow `self` mutably and immutably).
    pub fn color_view_clone(&self) -> wgpu::TextureView {
        self.color.create_view(&wgpu::TextureViewDescriptor::default())
    }
}

/// Which default texture a material slot falls back to.
#[derive(Clone, Copy)]
enum DefaultTex {
    WhiteSrgb,
    WhiteLinear,
    Normal,
}

/// A material texture slot resolved to either a real texture (+ cached sampler
/// key) or a shared default texture.
enum ResolvedSlot {
    Texture { index: usize, key: SamplerKey },
    Default(DefaultTex),
}

/// Build the composite bind group from a source view + sampler.
fn make_composite_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("composite bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Inverse-transpose of the model's upper-left 3x3, stored as a `mat4x4`
/// (column-major) so WGSL can read it with 16-byte column alignment.
fn normal_matrix(model: Mat4) -> [[f32; 4]; 4] {
    let m3 = Mat3::from_mat4(model);
    let n = m3.inverse().transpose();
    [
        [n.x_axis.x, n.x_axis.y, n.x_axis.z, 0.0],
        [n.y_axis.x, n.y_axis.y, n.y_axis.z, 0.0],
        [n.z_axis.x, n.z_axis.y, n.z_axis.z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Create a 1x1 texture of `format` filled with `rgba`, returning its view.
fn make_default_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    rgba: [u8; 4],
    label: &str,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform struct sizes must stay padded to 16-byte boundaries so the
    /// Rust layout matches the WGSL `struct` layout (std140-ish).
    #[test]
    fn uniform_sizes_are_16_byte_aligned() {
        assert_eq!(std::mem::size_of::<FrameUniform>() % 16, 0);
        assert_eq!(std::mem::size_of::<InstanceUniform>() % 16, 0);
        assert_eq!(std::mem::size_of::<MaterialUniform>() % 16, 0);
        // Concrete sizes (catch accidental field reordering / padding drift).
        assert_eq!(std::mem::size_of::<FrameUniform>(), 288);
        assert_eq!(std::mem::size_of::<InstanceUniform>(), 192);
        assert_eq!(std::mem::size_of::<MaterialUniform>(), 64);
    }

    /// Build a headless device. Returns `None` when no adapter is available so
    /// the test skips gracefully (mirrors the loader test).
    fn headless() -> Option<(wgpu::Device, wgpu::Queue)> {
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
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("renderer test device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        Some((device, queue))
    }

    /// Constructing the renderer compiles both WGSL shaders and builds every
    /// pipeline + bind-group layout, validating that the Rust-side layouts agree
    /// with the shader bindings. This is the only compile-time-ish check we have
    /// for the WGSL, since the GUI can't run headless.
    #[test]
    fn renderer_builds_and_shaders_validate() {
        let _ = env_logger::builder().is_test(true).try_init();
        let Some((device, queue)) = headless() else {
            log::warn!("no wgpu adapter; skipping renderer build test");
            return;
        };
        // A surface format the composite pipeline can target.
        let renderer = Renderer::new(
            &device,
            &queue,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            [320, 240],
        );
        assert_eq!(renderer.render_size(), [320, 240]);
        // Render target formats must equal the FSR-required constants.
        assert_eq!(COLOR_FORMAT, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(DEPTH_FORMAT, wgpu::TextureFormat::Depth32Float);
        assert_eq!(MOTION_FORMAT, wgpu::TextureFormat::Rg16Float);
    }

    /// Render one frame of a single lit triangle to an offscreen sRGB target and
    /// read it back, asserting the center pixel is non-black. This exercises the
    /// full draw path the GUI can't: MRT geometry pass (color + motion + depth),
    /// the instance storage buffer indexed by `instance_index`, and the
    /// fullscreen composite. Catches binding/layout mistakes that only surface
    /// at draw time.
    #[test]
    fn renders_a_lit_triangle() {
        let _ = env_logger::builder().is_test(true).try_init();
        let Some((device, queue)) = headless() else {
            log::warn!("no wgpu adapter; skipping triangle render test");
            return;
        };

        let size = [64u32, 64u32];
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut renderer = Renderer::new(&device, &queue, format, size);

        // A single front-facing (CCW) triangle filling much of the view, at
        // z=0 in front of the camera, with an upward-ish normal toward the sun.
        let verts = [
            crate::assets::Vertex {
                position: [-0.6, -0.6, 0.0],
                normal: [0.0, 0.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                uv: [0.0, 0.0],
            },
            crate::assets::Vertex {
                position: [0.6, -0.6, 0.0],
                normal: [0.0, 0.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                uv: [1.0, 0.0],
            },
            crate::assets::Vertex {
                position: [0.0, 0.6, 0.0],
                normal: [0.0, 0.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                uv: [0.5, 1.0],
            },
        ];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test tri verts"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = [0u32, 1, 2];
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test tri indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // A textureless emissive material so the result is unambiguously bright.
        let model = LoadedModel {
            primitives: Vec::new(),
            materials: Vec::new(),
            textures: Vec::new(),
            bounds: crate::assets::Aabb::empty(),
        };
        let material = Material {
            base_color_factor: [0.8, 0.2, 0.2, 1.0],
            emissive_factor: [0.0, 0.5, 0.0],
            ..Material::default()
        };
        let (material_bind_group, _buf) =
            renderer.create_material_bind_group(&device, &model, &material);

        // Camera looking down -Z at the triangle.
        let view = Mat4::look_to_rh(Vec3::new(0.0, 0.0, 2.0), -Vec3::Z, Vec3::Y);
        let proj = Mat4::perspective_rh(60f32.to_radians(), 1.0, 0.1, 1000.0);
        let vp = proj * view;

        let instances = [InstanceData {
            model: Mat4::IDENTITY,
            prev_model: Mat4::IDENTITY,
        }];
        let draws = [DrawItem {
            vertex_buffer: &vertex_buffer,
            index_buffer: &index_buffer,
            index_count: 3,
            material_bind_group: &material_bind_group,
            instance_index: 0,
            double_sided: true,
        }];

        // Offscreen composite target we can copy back from.
        let out = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test out"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let out_view = out.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer.render_geometry(
            &device,
            &queue,
            &mut encoder,
            &Frame {
                view_proj_jittered: vp,
                view_proj: vp,
                prev_view_proj: vp,
                camera_pos: Vec3::new(0.0, 0.0, 2.0),
                lighting: Lighting::default(),
                instances: &instances,
                draws: &draws,
            },
        );
        // FSR-off path: composite straight from the HDR color target.
        renderer.composite(&mut encoder, &out_view);

        // Copy the center row's center pixel out via a 256-aligned readback.
        let bytes_per_row = 256u32; // >= 64*4, satisfies COPY_BYTES_PER_ROW_ALIGNMENT
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (bytes_per_row * size[1]) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &out,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(size[1]),
                },
            },
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        readback.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = readback.slice(..).get_mapped_range();

        // Center pixel (32,32).
        let cx = 32usize;
        let cy = 32usize;
        let off = cy * bytes_per_row as usize + cx * 4;
        let px = [data[off], data[off + 1], data[off + 2]];
        let brightness = px[0] as u32 + px[1] as u32 + px[2] as u32;
        assert!(
            brightness > 30,
            "center pixel too dark ({px:?}); geometry/composite path likely broken"
        );
    }

    /// Reproduce the WGSL motion-vector math on the CPU to lock in the
    /// `prev - cur`, framebuffer-Y-down convention that phase 4 / FSR depends
    /// on. Mirrors `pbr.wgsl`'s fragment-shader formula exactly; if the visuals
    /// smear under FSR, the camera *jitter* sign is the place to look, not this.
    #[test]
    fn motion_vector_math_matches_shader_convention() {
        let render_size = [1920.0_f32, 1080.0_f32];
        let proj = Mat4::perspective_rh(60f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
        // Static world point in front of the camera.
        let world = glam::Vec4::new(0.0, 0.0, 0.0, 1.0);

        // Camera pans RIGHT between prev and cur: prev at x=-0.3, cur at x=0.0.
        // The static point therefore appears further right in the *previous*
        // frame, so prev_ndc.x > cur_ndc.x.
        let view_prev = Mat4::look_to_rh(Vec3::new(-0.3, 0.0, 2.0), -Vec3::Z, Vec3::Y);
        let view_cur = Mat4::look_to_rh(Vec3::new(0.0, 0.0, 2.0), -Vec3::Z, Vec3::Y);
        let cur_clip = proj * view_cur * world;
        let prev_clip = proj * view_prev * world;

        let cur_ndc = glam::Vec2::new(cur_clip.x / cur_clip.w, cur_clip.y / cur_clip.w);
        let prev_ndc = glam::Vec2::new(prev_clip.x / prev_clip.w, prev_clip.y / prev_clip.w);
        // Exactly the WGSL expression:
        //   mv_px = (prev_ndc - cur_ndc) * vec2(0.5*W, -0.5*H)
        let mv = (prev_ndc - cur_ndc)
            * glam::Vec2::new(0.5 * render_size[0], -0.5 * render_size[1]);

        assert!(cur_ndc.x.abs() < 1e-5, "cur should be centered, got {cur_ndc}");
        assert!(prev_ndc.x > 0.0, "prev should be right of center, got {prev_ndc}");
        // (prev-cur).x > 0 times (+0.5*W) => positive stored MV.x.
        assert!(
            mv.x > 0.0,
            "rightward camera pan must give positive stored MV.x, got {mv}"
        );
    }
}
