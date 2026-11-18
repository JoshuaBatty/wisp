//! Palettes for the WESL example, imported by `wisp/test_wesl.wesl`.

/// A cosine rainbow (after Inigo Quilez's cosine palettes), repeating every 1.0
/// of `t`.
fn rainbow(t: f32) -> vec3<f32> {
    return 0.5 + 0.5 * cos(6.2831853 * (t + vec3<f32>(0.0, 0.33, 0.67)));
}
