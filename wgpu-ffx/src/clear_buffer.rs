use bytemuck::{Pod, Zeroable};
use std::{borrow::Cow, convert::TryFrom};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE_X: u32 = 128;
const CLEAR_BUFFER_SHADER: &str = r#"
struct ClearUniforms {
    total_vec4s: u32;
    _pad0: u32;
    _pad1: u32;
    _pad2: u32;
    clear_value: vec4u;
};

@group(0) @binding(0)
var<storage, read_write> target: array<vec4u>;

@group(0) @binding(1)
var<uniform> uniforms: ClearUniforms;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let index = global_id.x;
    if (index >= uniforms.total_vec4s) {
        return;
    }
    target[index] = uniforms.clear_value;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClearUniformRaw {
    total_vec4s: u32,
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
        clear_value: [f32; 4],
    ) {
        let total_bytes = buffer.size();
        if total_bytes == 0 {
            return;
        }
        assert!(
            total_bytes % 16 == 0,
            "BufferClearer requires buffer size multiple of 16 bytes"
        );
        let total_vec4s =
            u32::try_from(total_bytes / 16).expect("BufferClearer buffer exceeds supported size");
        let uniforms = ClearUniformRaw {
            total_vec4s,
            _pad: [0; 3],
            clear_value: clear_value.map(f32::to_bits),
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
        let workgroups = total_vec4s.div_ceil(WORKGROUP_SIZE_X);
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
