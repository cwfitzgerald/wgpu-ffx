// Temporary tonemap composite: sample the HDR color target, tonemap (ACES
// fitted) to [0,1], and write to the sRGB surface. The surface is an sRGB
// format so the hardware applies the sRGB encode on store -- do NOT gamma
// encode here. Bilinear sampling upscales when render size < surface size.
//
// This pass IS the future "FSR OFF" path. Phase 4 keeps it for
// `!settings.fsr_enabled` and swaps the bound source texture (raw HDR at render
// res now; FSR output at display res later) via `Renderer::set_composite_source`.

@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_samp: sampler;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle, no vertex buffer.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    var out: VsOut;
    // (0,0), (2,0), (0,2) in UV; maps to a triangle covering the screen.
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u),
    );
    out.uv = uv;
    // UV (0..2) -> clip (-1..3); flip Y so UV origin is top-left.
    out.clip_position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return out;
}

// ACES filmic tonemap (Narkowicz fit).
fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_tex, hdr_samp, in.uv).rgb;
    let mapped = aces(hdr);
    // Output linear; the sRGB surface format encodes on store.
    return vec4<f32>(mapped, 1.0);
}
