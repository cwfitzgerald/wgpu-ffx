// Calculate halton number for index and base.
fn halton(index: i32, base: i32) -> f32 {
    let mut f = 1.0f32;
    let mut result = 0.0f32;
    let mut current_index = index;

    while current_index > 0 {
        f /= base as f32;
        result = result + f * (current_index % base) as f32;
        current_index = f32::floor(current_index as f32 / base as f32) as i32;
    }

    result
}

pub fn get_jitter_phase_count(render_width: i32, display_width: i32) -> i32 {
    const BASE_PHASE_COUNT: f32 = 8.0;
    let jitter_phase_count =
        (BASE_PHASE_COUNT * f32::powf(display_width as f32 / render_width as f32, 2.0)) as i32;
    jitter_phase_count
}

pub fn get_jitter_offset(index: i32, phase_count: i32) -> [f32; 2] {
    let x = halton((index % phase_count) + 1, 2) - 0.5f32;
    let y = halton((index % phase_count) + 1, 3) - 0.5f32;

    [x, y]
}
