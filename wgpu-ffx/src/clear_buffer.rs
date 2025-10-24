use bytemuck::{Pod, Zeroable};
use std::{borrow::Cow, convert::TryFrom};
use wgpu::util::DeviceExt;

const WORDS_PER_INVOCATION: u32 = 4;
const WORKGROUP_SIZE_X: u32 = 128;
const WORDS_PER_WORKGROUP: u32 = WORDS_PER_INVOCATION * WORKGROUP_SIZE_X;
const CLEAR_BUFFER_SHADER: &str = r#"
struct ClearUniforms {
    total_words: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    clear_value: vec4u,
};

@group(0) @binding(0)
var<storage, read_write> buf: array<u32>;

@group(0) @binding(1)
var<uniform> uniforms: ClearUniforms;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let index = global_id.x * 4u;
    if (index + 3 >= uniforms.total_words) {
        for (var i: u32 = index; i < uniforms.total_words; i = i + 1u) {
            buf[i] = uniforms.clear_value[i - index];
        }
        return;
    }

    buf[index + 0] = uniforms.clear_value[0];
    buf[index + 1] = uniforms.clear_value[1];
    buf[index + 2] = uniforms.clear_value[2];
    buf[index + 3] = uniforms.clear_value[3];
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClearUniformRaw {
    total_words: u32,
    _pad: [u32; 3],
    clear_value: [u32; 4],
}

pub struct BufferClearer {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl BufferClearer {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BufferClearer::shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CLEAR_BUFFER_SHADER)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BufferClearer::pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        Self {
            bind_group_layout,
            pipeline,
        }
    }

    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
        clear_value: [u32; 4],
    ) {
        let total_bytes = buffer.size();
        if total_bytes == 0 {
            return;
        }
        let total_words =
            u32::try_from(total_bytes / 4).expect("BufferClearer buffer exceeds supported size");
        let uniforms = ClearUniformRaw {
            total_words,
            _pad: [0; 3],
            clear_value,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BufferClearer::uniform_buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BufferClearer::bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let workgroups = total_words.div_ceil(WORDS_PER_WORKGROUP);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BufferClearer::pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}
