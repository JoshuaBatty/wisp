//! A WESL wisp: rings around a circle, coloured by a palette imported from a
//! local module and shaped by the `shapes` package the `wesl` example registers.

import super::lib::palette::rainbow;
import shapes::circle::circle;

struct Globals {
    resolution: vec2<f32>,
    time: f32,
}
@group(0) @binding(0) var<uniform> globals: Globals;

struct Params {
    /// Radius of the circle.
    /// @min(0.05) @max(0.5) @default(0.3)
    radius: f32,
    /// Rings per unit of distance from the circle's edge.
    /// @min(1.0) @max(40.0) @default(12.0)
    rings: f32,
}
@group(1) @binding(0) var<uniform> params: Params;

@fragment
fn fragment(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let aspect = globals.resolution.x / max(globals.resolution.y, 1.0);
    let p = (uv - 0.5) * vec2<f32>(aspect, 1.0);
    let d = circle(p, params.radius);
    let colour = rainbow(d * params.rings - globals.time * 0.25);
    let inside = 1.0 - smoothstep(0.0, 0.01, d);
    return vec4<f32>(colour * mix(0.3, 1.0, inside), 1.0);
}
