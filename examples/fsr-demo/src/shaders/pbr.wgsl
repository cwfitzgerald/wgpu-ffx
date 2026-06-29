// Forward PBR shader: glTF metallic-roughness with analytic directional + cheap
// hemisphere ambient lighting. Writes linear HDR color (location 0) and screen
// motion vectors (location 1).
//
// Layout MUST stay in lockstep with `renderer.rs`:
//   FrameUniform    <-> @group(0) @binding(0)
//   MaterialUniform <-> @group(1) @binding(0)  (+ textures/samplers 1..=10)
//   InstanceUniform <-> @group(2) @binding(0)  (storage array)

struct FrameUniform {
    view_proj_jittered: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,      // .xyz position, .w pad
    sun_direction: vec4<f32>,   // .xyz toward sun, .w pad
    sun_color: vec4<f32>,       // .xyz color, .w intensity
    sky_color: vec4<f32>,       // .xyz, .w pad
    ground_color: vec4<f32>,    // .xyz, .w pad
    render_size: vec4<f32>,     // .xy size, .zw pad
};

struct InstanceUniform {
    model: mat4x4<f32>,
    prev_model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>, // upper-left 3x3 meaningful
};

struct MaterialUniform {
    base_color_factor: vec4<f32>,
    emissive_metallic: vec4<f32>,        // .xyz emissive, .w metallic
    rough_normal_occ_cutoff: vec4<f32>,  // roughness, normal_scale, occ_strength, cutoff
    flags_alpha: vec4<u32>,              // flags, alpha_mode, pad, pad
};

// Texture-present flag bits (must match renderer.rs).
const FLAG_BASE_COLOR_TEX: u32 = 1u;
const FLAG_MR_TEX: u32 = 2u;
const FLAG_NORMAL_TEX: u32 = 4u;
const FLAG_OCCLUSION_TEX: u32 = 8u;
const FLAG_EMISSIVE_TEX: u32 = 16u;

const ALPHA_MASK: u32 = 1u;

const PI: f32 = 3.14159265359;

@group(0) @binding(0) var<uniform> frame: FrameUniform;

@group(1) @binding(0) var<uniform> material: MaterialUniform;
@group(1) @binding(1) var base_color_tex: texture_2d<f32>;
@group(1) @binding(2) var mr_tex: texture_2d<f32>;
@group(1) @binding(3) var normal_tex: texture_2d<f32>;
@group(1) @binding(4) var occlusion_tex: texture_2d<f32>;
@group(1) @binding(5) var emissive_tex: texture_2d<f32>;
@group(1) @binding(6) var base_color_samp: sampler;
@group(1) @binding(7) var mr_samp: sampler;
@group(1) @binding(8) var normal_samp: sampler;
@group(1) @binding(9) var occlusion_samp: sampler;
@group(1) @binding(10) var emissive_samp: sampler;

@group(2) @binding(0) var<storage, read> instances: array<InstanceUniform>;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) uv: vec2<f32>,
    // Unjittered current/previous clip positions for motion vectors.
    @location(5) cur_clip: vec4<f32>,
    @location(6) prev_clip: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn, @builtin(instance_index) instance_index: u32) -> VsOut {
    let inst = instances[instance_index];
    let world_pos4 = inst.model * vec4<f32>(in.position, 1.0);
    let world_pos = world_pos4.xyz;

    let nm = mat3x3<f32>(
        inst.normal_matrix[0].xyz,
        inst.normal_matrix[1].xyz,
        inst.normal_matrix[2].xyz,
    );
    let normal = normalize(nm * in.normal);
    // Tangent transforms with the model matrix (not the normal matrix).
    let m3 = mat3x3<f32>(inst.model[0].xyz, inst.model[1].xyz, inst.model[2].xyz);
    let tangent = normalize(m3 * in.tangent.xyz);
    // Re-orthogonalize and build the bitangent with the stored handedness.
    let t = normalize(tangent - normal * dot(normal, tangent));
    let bitangent = cross(normal, t) * in.tangent.w;

    var out: VsOut;
    out.clip_position = frame.view_proj_jittered * world_pos4;
    out.world_pos = world_pos;
    out.normal = normal;
    out.tangent = t;
    out.bitangent = bitangent;
    out.uv = in.uv;
    // Motion vectors use the UNJITTERED matrices.
    out.cur_clip = frame.view_proj * world_pos4;
    out.prev_clip = frame.prev_view_proj * (inst.prev_model * vec4<f32>(in.position, 1.0));
    return out;
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) motion: vec2<f32>,
};

fn has_flag(flag: u32) -> bool {
    return (material.flags_alpha.x & flag) != 0u;
}

// GGX / Trowbridge-Reitz normal distribution.
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// Smith geometry term with Schlick-GGX, height-correlated approximation.
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    // --- material sampling (always sample then multiply by factor) ---
    var base_color = material.base_color_factor;
    if (has_flag(FLAG_BASE_COLOR_TEX)) {
        base_color = base_color * textureSample(base_color_tex, base_color_samp, in.uv);
    }

    // Alpha mask: discard before doing expensive shading.
    if (material.flags_alpha.y == ALPHA_MASK && base_color.a < material.rough_normal_occ_cutoff.w) {
        discard;
    }

    var metallic = material.emissive_metallic.w;
    var roughness = material.rough_normal_occ_cutoff.x;
    if (has_flag(FLAG_MR_TEX)) {
        let mr = textureSample(mr_tex, mr_samp, in.uv);
        roughness = roughness * mr.g;
        metallic = metallic * mr.b;
    }
    roughness = clamp(roughness, 0.04, 1.0);

    var occlusion = 1.0;
    if (has_flag(FLAG_OCCLUSION_TEX)) {
        let occ = textureSample(occlusion_tex, occlusion_samp, in.uv).r;
        occlusion = mix(1.0, occ, material.rough_normal_occ_cutoff.z);
    }

    var emissive = material.emissive_metallic.xyz;
    if (has_flag(FLAG_EMISSIVE_TEX)) {
        emissive = emissive * textureSample(emissive_tex, emissive_samp, in.uv).rgb;
    }

    // --- normal mapping ---
    var n = normalize(in.normal);
    if (has_flag(FLAG_NORMAL_TEX)) {
        let tn = textureSample(normal_tex, normal_samp, in.uv).xyz * 2.0 - 1.0;
        let scale = material.rough_normal_occ_cutoff.y;
        let scaled = vec3<f32>(tn.xy * scale, tn.z);
        let tbn = mat3x3<f32>(normalize(in.tangent), normalize(in.bitangent), n);
        n = normalize(tbn * scaled);
    }

    // --- lighting ---
    let albedo = base_color.rgb;
    let view_dir = normalize(frame.camera_pos.xyz - in.world_pos);
    // Two-sided shading so back faces (cull-none materials) aren't black.
    if (dot(n, view_dir) < 0.0) {
        n = -n;
    }
    let n_dot_v = max(dot(n, view_dir), 1e-4);

    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Directional sun (Cook-Torrance specular + Lambert diffuse).
    let light_dir = normalize(frame.sun_direction.xyz);
    let half_vec = normalize(view_dir + light_dir);
    let n_dot_l = max(dot(n, light_dir), 0.0);
    let n_dot_h = max(dot(n, half_vec), 0.0);
    let v_dot_h = max(dot(view_dir, half_vec), 0.0);

    let ndf = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(v_dot_h, f0);

    let specular = (ndf * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * albedo / PI;

    let sun_radiance = frame.sun_color.rgb * frame.sun_color.w;
    var color = (diffuse + specular) * sun_radiance * n_dot_l;

    // Cheap hemisphere ambient (sky from above, ground from below), modulated
    // by occlusion so cavities aren't pure black.
    let hemi = mix(frame.ground_color.rgb, frame.sky_color.rgb, n.y * 0.5 + 0.5);
    color = color + albedo * hemi * occlusion * (1.0 - metallic);
    // A touch of ambient specular so metals catch the sky.
    color = color + f0 * frame.sky_color.rgb * occlusion;

    color = color + emissive;

    // --- motion vector (prev - cur), in pixels, framebuffer Y-down ---
    let cur_ndc = in.cur_clip.xy / in.cur_clip.w;
    let prev_ndc = in.prev_clip.xy / in.prev_clip.w;
    let mv_px = (prev_ndc - cur_ndc)
        * vec2<f32>(0.5 * frame.render_size.x, -0.5 * frame.render_size.y);

    var out: FsOut;
    out.color = vec4<f32>(color, base_color.a);
    out.motion = mv_px;
    return out;
}
