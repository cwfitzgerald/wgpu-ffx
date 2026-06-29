//! glTF -> GPU asset loader.
//!
//! [`load_glb`] parses a single-file `.glb`/`.gltf` via the [`gltf`] crate
//! (using [`gltf::import`] so buffers and images are decoded) and produces a
//! [`LoadedModel`]: GPU vertex/index buffers, metallic-roughness materials, and
//! mipmapped textures, plus a model-space bounding box.
//!
//! The loader is intentionally renderer-agnostic. It uploads textures with the
//! correct color spaces (sRGB for base-color/emissive, linear for the rest),
//! expands odd channel counts to RGBA8, generates a CPU box-filtered mip chain,
//! generates tangents when a primitive omits them, and bakes the glTF node TRS
//! hierarchy into each primitive's transform. It does not build bind groups,
//! pipelines, or samplers.

use std::path::Path;

use anyhow::{Context as _, Result, bail};
use glam::{Mat4, Vec2, Vec3, Vec4};
use wgpu::util::DeviceExt as _;

use crate::assets::{
    Aabb, AlphaMode, GpuTexture, LoadedModel, Material, Primitive, SamplerInfo, Vertex,
};

/// Load a single-file glTF model (`.glb` or `.gltf` with embedded/relative
/// resources) and upload its geometry and textures to the GPU.
///
/// Returns a [`LoadedModel`] whose primitives are flattened from the default
/// scene (falling back to the first scene), with node transforms baked into
/// each [`Primitive::local_transform`].
pub fn load_glb(device: &wgpu::Device, queue: &wgpu::Queue, path: &Path) -> Result<LoadedModel> {
    let (document, buffers, images) =
        gltf::import(path).with_context(|| format!("failed to import glTF: {}", path.display()))?;

    let label = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    // Upload every texture in the document once. Color space depends on how the
    // texture is used by materials, so resolve that first.
    let srgb_textures = collect_srgb_texture_indices(&document);
    let textures = document
        .textures()
        .map(|texture| {
            let image_index = texture.source().index();
            let image = &images[image_index];
            let srgb = srgb_textures.contains(&texture.index());
            upload_texture(device, queue, image, &sampler_info(&texture), srgb, label)
        })
        .collect::<Result<Vec<_>>>()?;

    let materials: Vec<Material> = document.materials().map(convert_material).collect();

    // Walk the default scene, baking the TRS chain into a world matrix per node.
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .context("glTF has no scenes")?;

    let mut primitives = Vec::new();
    let mut bounds = Aabb::empty();
    for node in scene.nodes() {
        visit_node(
            device,
            &node,
            Mat4::IDENTITY,
            &buffers,
            label,
            &mut primitives,
            &mut bounds,
        )?;
    }

    if primitives.is_empty() {
        bail!("glTF produced no triangle primitives: {}", path.display());
    }

    log::info!(
        "loaded {label}: {} primitives, {} materials, {} textures",
        primitives.len(),
        materials.len(),
        textures.len(),
    );

    Ok(LoadedModel {
        primitives,
        materials,
        textures,
        bounds,
    })
}

/// Recursively walk a node and its children, accumulating the transform and
/// emitting one [`Primitive`] per triangle primitive encountered.
fn visit_node(
    device: &wgpu::Device,
    node: &gltf::Node,
    parent: Mat4,
    buffers: &[gltf::buffer::Data],
    label: &str,
    primitives: &mut Vec<Primitive>,
    bounds: &mut Aabb,
) -> Result<()> {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent * local;

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                log::warn!(
                    "{label}: skipping primitive with unsupported mode {:?}",
                    primitive.mode()
                );
                continue;
            }
            if let Some(prim) =
                build_primitive(device, &primitive, world, buffers, label, bounds)?
            {
                primitives.push(prim);
            }
        }
    }

    for child in node.children() {
        visit_node(device, &child, world, buffers, label, primitives, bounds)?;
    }

    Ok(())
}

/// Build GPU buffers for a single triangle primitive, generating missing
/// normals/tangents/indices as needed, and grow `bounds` by its world-space
/// positions. Returns `None` for an empty primitive.
fn build_primitive(
    device: &wgpu::Device,
    primitive: &gltf::Primitive,
    world: Mat4,
    buffers: &[gltf::buffer::Data],
    label: &str,
    bounds: &mut Aabb,
) -> Result<Option<Primitive>> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    let positions: Vec<[f32; 3]> = match reader.read_positions() {
        Some(iter) => iter.collect(),
        None => {
            log::warn!("{label}: primitive has no POSITION; skipping");
            return Ok(None);
        }
    };
    if positions.is_empty() {
        return Ok(None);
    }
    let vertex_count = positions.len();

    // Indices: widen to u32, or synthesize a trivial list when absent.
    let indices: Vec<u32> = match reader.read_indices() {
        Some(read) => read.into_u32().collect(),
        None => (0..vertex_count as u32).collect(),
    };
    if indices.is_empty() {
        return Ok(None);
    }

    let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
        Some(read) => read.into_f32().collect(),
        None => vec![[0.0, 0.0]; vertex_count],
    };

    let normals: Vec<[f32; 3]> = match reader.read_normals() {
        Some(iter) => iter.collect(),
        None => compute_flat_normals(&positions, &indices),
    };

    let tangents: Vec<[f32; 4]> = match reader.read_tangents() {
        Some(iter) => iter.collect(),
        None => compute_tangents(&positions, &normals, &uvs, &indices),
    };

    // Interleave into the renderer vertex layout. Lengths can differ if an
    // attribute was sparse; clamp by repeating sensible defaults.
    let vertices: Vec<Vertex> = (0..vertex_count)
        .map(|i| Vertex {
            position: positions[i],
            normal: *normals.get(i).unwrap_or(&[0.0, 0.0, 1.0]),
            tangent: *tangents.get(i).unwrap_or(&[1.0, 0.0, 0.0, 1.0]),
            uv: *uvs.get(i).unwrap_or(&[0.0, 0.0]),
        })
        .collect();

    // Grow model bounds using world-space positions.
    for p in &positions {
        let wp = world.transform_point3(Vec3::from_array(*p));
        bounds.expand(wp);
    }

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{label} vertices")),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{label} indices")),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    Ok(Some(Primitive {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        material: primitive.material().index().unwrap_or(0),
        local_transform: world,
    }))
}

/// Compute per-vertex flat normals by accumulating triangle face normals.
fn compute_flat_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = Vec3::from_array(positions[i0]);
        let p1 = Vec3::from_array(positions[i1]);
        let p2 = Vec3::from_array(positions[i2]);
        let face = (p1 - p0).cross(p2 - p0);
        normals[i0] += face;
        normals[i1] += face;
        normals[i2] += face;
    }
    normals
        .into_iter()
        .map(|n| n.normalize_or(Vec3::Z).to_array())
        .collect()
}

/// Generate a per-vertex tangent basis from positions, normals, and UVs.
///
/// Accumulates per-triangle tangents (Lengyel's method), orthonormalizes each
/// against the vertex normal (Gram-Schmidt), and stores the bitangent
/// handedness in `w`. This is the straightforward averaged approach; it is
/// adequate for the demo's models, which are smooth and mostly ship tangents
/// already.
fn compute_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let n = positions.len();
    let mut tan = vec![Vec3::ZERO; n];
    let mut bitan = vec![Vec3::ZERO; n];

    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = Vec3::from_array(positions[i0]);
        let p1 = Vec3::from_array(positions[i1]);
        let p2 = Vec3::from_array(positions[i2]);
        let w0 = Vec2::from_array(uvs[i0]);
        let w1 = Vec2::from_array(uvs[i1]);
        let w2 = Vec2::from_array(uvs[i2]);

        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let du1 = w1 - w0;
        let du2 = w2 - w0;

        let denom = du1.x * du2.y - du2.x * du1.y;
        // Degenerate UVs produce a zero determinant; skip rather than NaN.
        if denom.abs() < 1e-12 {
            continue;
        }
        let r = 1.0 / denom;
        let t = (e1 * du2.y - e2 * du1.y) * r;
        let b = (e2 * du1.x - e1 * du2.x) * r;

        for &i in &[i0, i1, i2] {
            tan[i] += t;
            bitan[i] += b;
        }
    }

    (0..n)
        .map(|i| {
            let normal = Vec3::from_array(normals[i]);
            let t = tan[i];
            // Gram-Schmidt orthonormalize against the normal.
            let tangent = (t - normal * normal.dot(t)).normalize_or(fallback_tangent(normal));
            // Handedness: +1 if the bitangent agrees with N x T, else -1.
            let handedness = if normal.cross(tangent).dot(bitan[i]) < 0.0 {
                -1.0
            } else {
                1.0
            };
            Vec4::new(tangent.x, tangent.y, tangent.z, handedness).to_array()
        })
        .collect()
}

/// A stable tangent perpendicular to `normal`, used when UV-derived tangents
/// vanish (e.g. degenerate or missing UVs).
fn fallback_tangent(normal: Vec3) -> Vec3 {
    let axis = if normal.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    normal.cross(axis).normalize_or(Vec3::X)
}

/// Convert a glTF material into the renderer-friendly [`Material`].
fn convert_material(material: gltf::Material) -> Material {
    let pbr = material.pbr_metallic_roughness();
    let alpha_mode = match material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => AlphaMode::Mask {
            cutoff: material.alpha_cutoff().unwrap_or(0.5),
        },
        gltf::material::AlphaMode::Blend => AlphaMode::Blend,
    };

    Material {
        base_color_factor: pbr.base_color_factor(),
        metallic_factor: pbr.metallic_factor(),
        roughness_factor: pbr.roughness_factor(),
        emissive_factor: material.emissive_factor(),
        normal_scale: material
            .normal_texture()
            .map(|t| t.scale())
            .unwrap_or(1.0),
        occlusion_strength: material
            .occlusion_texture()
            .map(|t| t.strength())
            .unwrap_or(1.0),
        alpha_mode,
        double_sided: material.double_sided(),
        base_color_texture: pbr.base_color_texture().map(|t| t.texture().index()),
        metallic_roughness_texture: pbr
            .metallic_roughness_texture()
            .map(|t| t.texture().index()),
        normal_texture: material.normal_texture().map(|t| t.texture().index()),
        occlusion_texture: material.occlusion_texture().map(|t| t.texture().index()),
        emissive_texture: material.emissive_texture().map(|t| t.texture().index()),
    }
}

/// Collect the set of texture indices that must be uploaded as sRGB: the
/// base-color and emissive textures. All other textures hold linear data.
fn collect_srgb_texture_indices(document: &gltf::Document) -> std::collections::HashSet<usize> {
    let mut set = std::collections::HashSet::new();
    for material in document.materials() {
        if let Some(info) = material.pbr_metallic_roughness().base_color_texture() {
            set.insert(info.texture().index());
        }
        if let Some(info) = material.emissive_texture() {
            set.insert(info.texture().index());
        }
    }
    set
}

/// Translate a glTF sampler into wgpu address/filter modes.
fn sampler_info(texture: &gltf::Texture) -> SamplerInfo {
    use gltf::texture::{MagFilter, MinFilter, WrappingMode};

    let sampler = texture.sampler();
    let wrap = |mode| match mode {
        WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        WrappingMode::Repeat => wgpu::AddressMode::Repeat,
    };
    let mag = match sampler.mag_filter() {
        Some(MagFilter::Nearest) => wgpu::FilterMode::Nearest,
        _ => wgpu::FilterMode::Linear,
    };
    let (min, mip) = match sampler.min_filter() {
        Some(MinFilter::Nearest)
        | Some(MinFilter::NearestMipmapNearest)
        | Some(MinFilter::NearestMipmapLinear) => {
            let mip = match sampler.min_filter() {
                Some(MinFilter::NearestMipmapLinear) => wgpu::FilterMode::Linear,
                _ => wgpu::FilterMode::Nearest,
            };
            (wgpu::FilterMode::Nearest, mip)
        }
        Some(MinFilter::LinearMipmapNearest) => {
            (wgpu::FilterMode::Linear, wgpu::FilterMode::Nearest)
        }
        _ => (wgpu::FilterMode::Linear, wgpu::FilterMode::Linear),
    };

    SamplerInfo {
        address_mode_u: wrap(sampler.wrap_s()),
        address_mode_v: wrap(sampler.wrap_t()),
        mag_filter: mag,
        min_filter: min,
        mipmap_filter: mip,
    }
}

/// Expand decoded glTF image data to tightly packed RGBA8 (`width * height * 4`
/// bytes). wgpu has no 3-channel 8-bit format, and grayscale/two-channel images
/// must be padded. 16-bit and float formats are narrowed to 8-bit.
fn to_rgba8(image: &gltf::image::Data) -> Vec<u8> {
    use gltf::image::Format;

    let px = (image.width * image.height) as usize;
    let src = &image.pixels;
    let mut out = vec![0u8; px * 4];

    // Narrow a 16-bit channel (little-endian pairs) to 8 bits.
    let n16 = |bytes: &[u8], i: usize| -> u8 {
        let lo = bytes[i * 2] as u16;
        let hi = bytes[i * 2 + 1] as u16;
        ((lo | (hi << 8)) >> 8) as u8
    };
    // Narrow an f32 channel (assumed 0..=1) to 8 bits.
    let nf = |bytes: &[u8], i: usize| -> u8 {
        let b = [bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]];
        (f32::from_le_bytes(b).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    };

    match image.format {
        Format::R8 => {
            for i in 0..px {
                out[i * 4] = src[i];
                out[i * 4 + 3] = 255;
            }
        }
        Format::R8G8 => {
            for i in 0..px {
                out[i * 4] = src[i * 2];
                out[i * 4 + 1] = src[i * 2 + 1];
                out[i * 4 + 3] = 255;
            }
        }
        Format::R8G8B8 => {
            for i in 0..px {
                out[i * 4] = src[i * 3];
                out[i * 4 + 1] = src[i * 3 + 1];
                out[i * 4 + 2] = src[i * 3 + 2];
                out[i * 4 + 3] = 255;
            }
        }
        Format::R8G8B8A8 => out.copy_from_slice(&src[..px * 4]),
        Format::R16 => {
            for i in 0..px {
                out[i * 4] = n16(src, i);
                out[i * 4 + 3] = 255;
            }
        }
        Format::R16G16 => {
            for i in 0..px {
                out[i * 4] = n16(src, i * 2);
                out[i * 4 + 1] = n16(src, i * 2 + 1);
                out[i * 4 + 3] = 255;
            }
        }
        Format::R16G16B16 => {
            for i in 0..px {
                out[i * 4] = n16(src, i * 3);
                out[i * 4 + 1] = n16(src, i * 3 + 1);
                out[i * 4 + 2] = n16(src, i * 3 + 2);
                out[i * 4 + 3] = 255;
            }
        }
        Format::R16G16B16A16 => {
            for i in 0..px {
                out[i * 4] = n16(src, i * 4);
                out[i * 4 + 1] = n16(src, i * 4 + 1);
                out[i * 4 + 2] = n16(src, i * 4 + 2);
                out[i * 4 + 3] = n16(src, i * 4 + 3);
            }
        }
        Format::R32G32B32FLOAT => {
            for i in 0..px {
                out[i * 4] = nf(src, i * 3);
                out[i * 4 + 1] = nf(src, i * 3 + 1);
                out[i * 4 + 2] = nf(src, i * 3 + 2);
                out[i * 4 + 3] = 255;
            }
        }
        Format::R32G32B32A32FLOAT => {
            for i in 0..px {
                out[i * 4] = nf(src, i * 4);
                out[i * 4 + 1] = nf(src, i * 4 + 1);
                out[i * 4 + 2] = nf(src, i * 4 + 2);
                out[i * 4 + 3] = nf(src, i * 4 + 3);
            }
        }
    }

    out
}

/// Box-filter downsample an RGBA8 image to the next mip level (each dimension
/// halved, rounded down to a minimum of 1). Averages the up-to-2x2 source
/// texels covering each destination texel.
fn downsample_rgba8(src: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let dst_w = (width / 2).max(1);
    let dst_h = (height / 2).max(1);
    let mut dst = vec![0u8; (dst_w * dst_h * 4) as usize];

    for y in 0..dst_h {
        for x in 0..dst_w {
            // Source texels for this destination texel, clamped at edges so an
            // odd dimension's last column/row is still sampled.
            let sx0 = (x * 2).min(width - 1);
            let sy0 = (y * 2).min(height - 1);
            let sx1 = (x * 2 + 1).min(width - 1);
            let sy1 = (y * 2 + 1).min(height - 1);
            let coords = [(sx0, sy0), (sx1, sy0), (sx0, sy1), (sx1, sy1)];

            for c in 0..4usize {
                let sum: u32 = coords
                    .iter()
                    .map(|&(sx, sy)| src[((sy * width + sx) * 4) as usize + c] as u32)
                    .sum();
                dst[((y * dst_w + x) * 4) as usize + c] = (sum / 4) as u8;
            }
        }
    }

    (dst, dst_w, dst_h)
}

/// Upload one decoded image as a 2D texture with a full, CPU-generated mip
/// chain. `srgb` selects `Rgba8UnormSrgb` (base-color/emissive) vs.
/// `Rgba8Unorm` (linear: normal/metallic-roughness/occlusion).
fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &gltf::image::Data,
    sampler: &SamplerInfo,
    srgb: bool,
    label: &str,
) -> Result<GpuTexture> {
    let width = image.width.max(1);
    let height = image.height.max(1);
    let format = if srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };

    let mip_level_count = mip_count(width, height);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("{label} texture")),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Generate the mip chain on the CPU and upload each level.
    let mut level = to_rgba8(image);
    let mut lw = width;
    let mut lh = height;
    for mip in 0..mip_level_count {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &level,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(lw * 4),
                rows_per_image: Some(lh),
            },
            wgpu::Extent3d {
                width: lw,
                height: lh,
                depth_or_array_layers: 1,
            },
        );

        if mip + 1 < mip_level_count {
            let (next, nw, nh) = downsample_rgba8(&level, lw, lh);
            level = next;
            lw = nw;
            lh = nh;
        }
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    Ok(GpuTexture {
        texture,
        view,
        sampler: *sampler,
    })
}

/// Number of mip levels for a `width` x `height` texture: `floor(log2(max)) + 1`.
fn mip_count(width: u32, height: u32) -> u32 {
    32 - width.max(height).leading_zeros()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    /// Create a headless wgpu device (no surface). Returns `None` if no adapter
    /// is available in the environment so the test can skip gracefully.
    fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
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
            label: Some("headless test device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;

        Some((device, queue))
    }

    /// Absolute path to a bundled asset by file name.
    fn asset_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join(name)
    }

    #[test]
    fn loads_all_bundled_models() {
        let _ = env_logger::builder().is_test(true).try_init();

        let Some((device, queue)) = headless_device() else {
            log::warn!("no wgpu adapter available; skipping load test");
            return;
        };

        let models = [
            "DamagedHelmet.glb",
            "MetalRoughSpheres.glb",
            "Duck.glb",
            "Avocado.glb",
        ];

        for name in models {
            let path = asset_path(name);
            assert!(path.exists(), "missing bundled asset: {}", path.display());

            let model = load_glb(&device, &queue, &path)
                .unwrap_or_else(|e| panic!("failed to load {name}: {e:#}"));

            // Primitives exist and carry real geometry.
            assert!(!model.primitives.is_empty(), "{name}: no primitives");
            for (i, prim) in model.primitives.iter().enumerate() {
                assert!(prim.index_count > 0, "{name} prim {i}: zero indices");
                assert!(
                    prim.material < model.materials.len(),
                    "{name} prim {i}: material index {} out of range (len {})",
                    prim.material,
                    model.materials.len()
                );
            }

            // Every texture index referenced by a material is in range.
            for (i, mat) in model.materials.iter().enumerate() {
                for tex in [
                    mat.base_color_texture,
                    mat.metallic_roughness_texture,
                    mat.normal_texture,
                    mat.occlusion_texture,
                    mat.emissive_texture,
                ]
                .into_iter()
                .flatten()
                {
                    assert!(
                        tex < model.textures.len(),
                        "{name} material {i}: texture index {tex} out of range (len {})",
                        model.textures.len()
                    );
                }
            }

            // Bounds must be finite and non-degenerate.
            assert!(
                model.bounds.is_valid(),
                "{name}: invalid bounds {:?}",
                model.bounds
            );
            assert!(
                model.bounds.size().length() > 0.0,
                "{name}: degenerate bounds {:?}",
                model.bounds
            );

            log::info!(
                "{name}: {} primitives, {} materials, {} textures, bounds min={:?} max={:?}",
                model.primitives.len(),
                model.materials.len(),
                model.textures.len(),
                model.bounds.min.to_array(),
                model.bounds.max.to_array(),
            );
        }
    }
}
