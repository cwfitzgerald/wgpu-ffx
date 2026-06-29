//! Renderer-friendly intermediate representation produced by the glTF loader.
//!
//! These types describe a model after parsing and GPU upload: vertex/index
//! buffers, metallic-roughness materials, and the textures they reference.
//! They are deliberately decoupled from any rendering concern — the loader
//! ([`crate::gltf_loader`]) fills them in, and the phase-3 renderer consumes
//! them to build bind groups and pipelines. Nothing here creates a
//! [`wgpu::BindGroup`] or [`wgpu::BindGroupLayout`]; that is the renderer's job.

/// A fully loaded model: GPU geometry, materials, and textures, plus the
/// model-space bounding box used to center and scale it in a scene.
pub struct LoadedModel {
    /// Every drawable primitive across the model's default scene, flattened
    /// with each primitive's baked node transform in [`Primitive::local_transform`].
    pub primitives: Vec<Primitive>,
    /// Materials referenced by [`Primitive::material`].
    pub materials: Vec<Material>,
    /// Textures referenced by the `Option<usize>` fields of [`Material`].
    pub textures: Vec<GpuTexture>,
    /// Axis-aligned bounds over all primitive positions, each transformed by
    /// that primitive's [`Primitive::local_transform`].
    pub bounds: Aabb,
}

/// One drawable primitive: GPU buffers, a material index, and its model-local
/// transform.
pub struct Primitive {
    /// Vertex buffer holding a tightly packed array of [`Vertex`].
    pub vertex_buffer: wgpu::Buffer,
    /// Index buffer of `u32` indices (widened from narrower glTF index types).
    pub index_buffer: wgpu::Buffer,
    /// Number of indices in [`Primitive::index_buffer`].
    pub index_count: u32,
    /// Index into [`LoadedModel::materials`].
    pub material: usize,
    /// World transform of this primitive's node *within* the model: the product
    /// of the TRS chain from the scene root down to the node. The scene places
    /// the whole model by multiplying an additional placement matrix on top.
    pub local_transform: glam::Mat4,
}

/// A single interleaved vertex. The layout matches what the phase-3 renderer
/// will declare as its [`wgpu::VertexBufferLayout`].
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Object-space position.
    pub position: [f32; 3],
    /// Object-space normal (unit length).
    pub normal: [f32; 3],
    /// Object-space tangent; `xyz` is the unit tangent and `w` is the
    /// bitangent handedness sign (`+1.0` or `-1.0`).
    pub tangent: [f32; 4],
    /// `TEXCOORD_0` texture coordinates.
    pub uv: [f32; 2],
}

/// How a material's alpha channel is interpreted, mirroring glTF's alpha modes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlphaMode {
    /// Alpha is ignored; output is fully opaque.
    Opaque,
    /// Output is opaque where alpha `>= cutoff`, otherwise discarded.
    Mask {
        /// Alpha threshold below which fragments are discarded.
        cutoff: f32,
    },
    /// Alpha is used for conventional blending.
    Blend,
}

/// glTF metallic-roughness material parameters plus texture references.
///
/// Each `Option<usize>` indexes into [`LoadedModel::textures`]; `None` means
/// "use the scalar/vector factor alone, with no texture".
pub struct Material {
    /// Linear base-color multiplier (RGBA).
    pub base_color_factor: [f32; 4],
    /// Scalar metalness multiplier.
    pub metallic_factor: f32,
    /// Scalar roughness multiplier.
    pub roughness_factor: f32,
    /// Linear emissive color.
    pub emissive_factor: [f32; 3],
    /// Scale applied to the sampled normal map's XY.
    pub normal_scale: f32,
    /// Strength applied to the sampled occlusion value.
    pub occlusion_strength: f32,
    /// How the alpha channel is interpreted.
    pub alpha_mode: AlphaMode,
    /// Whether back faces should be rendered.
    pub double_sided: bool,
    /// sRGB base-color texture (index into [`LoadedModel::textures`]).
    pub base_color_texture: Option<usize>,
    /// Linear metallic-roughness texture (G = roughness, B = metalness).
    pub metallic_roughness_texture: Option<usize>,
    /// Linear tangent-space normal map.
    pub normal_texture: Option<usize>,
    /// Linear occlusion texture (R channel).
    pub occlusion_texture: Option<usize>,
    /// sRGB emissive texture.
    pub emissive_texture: Option<usize>,
}

impl Default for Material {
    /// The glTF default material: white dielectric, fully rough, opaque.
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic_factor: 1.0,
            roughness_factor: 1.0,
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
        }
    }
}

/// A GPU texture with a full mip chain, its view, and the glTF sampler
/// parameters that should be used to sample it.
///
/// The actual [`wgpu::Sampler`] is created by the renderer (phase 3); the
/// parameters are exposed here so it can honour per-texture wrap/filter modes
/// instead of guessing.
pub struct GpuTexture {
    /// The uploaded texture (mip 0 plus a generated box-filtered mip chain).
    pub texture: wgpu::Texture,
    /// A default 2D view over all mip levels.
    pub view: wgpu::TextureView,
    /// glTF sampler parameters (wrap + filter modes) for this texture.
    pub sampler: SamplerInfo,
}

/// glTF sampler description, translated to wgpu enums so the renderer can build
/// a matching [`wgpu::Sampler`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplerInfo {
    /// Horizontal (U) address mode.
    pub address_mode_u: wgpu::AddressMode,
    /// Vertical (V) address mode.
    pub address_mode_v: wgpu::AddressMode,
    /// Magnification filter.
    pub mag_filter: wgpu::FilterMode,
    /// Minification filter.
    pub min_filter: wgpu::FilterMode,
    /// Mip-level filter (derived from the glTF min filter).
    pub mipmap_filter: wgpu::FilterMode,
}

impl Default for SamplerInfo {
    /// glTF's defaults: repeat wrap and linear filtering with linear mips.
    fn default() -> Self {
        Self {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
        }
    }
}

/// An axis-aligned bounding box in model space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Component-wise minimum corner.
    pub min: glam::Vec3,
    /// Component-wise maximum corner.
    pub max: glam::Vec3,
}

impl Aabb {
    /// An empty box with `min = +inf` and `max = -inf`, ready to be grown by
    /// [`Aabb::expand`].
    pub fn empty() -> Self {
        Self {
            min: glam::Vec3::splat(f32::INFINITY),
            max: glam::Vec3::splat(f32::NEG_INFINITY),
        }
    }

    /// Grow the box to include `point`.
    pub fn expand(&mut self, point: glam::Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// The center of the box.
    pub fn center(&self) -> glam::Vec3 {
        (self.min + self.max) * 0.5
    }

    /// The full extent (size) of the box along each axis.
    pub fn size(&self) -> glam::Vec3 {
        self.max - self.min
    }

    /// Whether the box is non-empty and all corners are finite. Used by the
    /// load test to reject degenerate geometry.
    pub fn is_valid(&self) -> bool {
        self.min.is_finite() && self.max.is_finite() && self.min.cmple(self.max).all()
    }
}
