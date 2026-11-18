//! A volume pass, Shadertoy style: a compute pass bakes glowing smoke columns
//! into a 3d target, reading its own last frame so the smoke rises and fades,
//! and the final pass raymarches the volume from an orbiting camera.

struct Globals {
    resolution: vec2<f32>,
    time: f32,
    time_delta: f32,
}
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var samp: sampler;

struct Params {
    /// @min(0.0) @max(4.0) @default(1.0) Drift speed of the smoke.
    speed: f32,
    /// @min(0.0) @max(4.0) @default(1.5) Seconds the smoke lingers.
    linger: f32,
    /// @min(0.0) @max(40.0) @default(8.0) Opacity of the smoke.
    density: f32,
    /// @color @default(1.0, 0.45, 0.15)
    tint: vec3<f32>,
}
@group(1) @binding(0) var<uniform> params: Params;

@group(1) @binding(1) var smoke: texture_3d<f32>;
@group(1) @binding(2) var smoke_out: texture_storage_3d<rgba16float, write>;

fn hash(p: vec3<f32>) -> f32 {
    let q = fract(p * 0.3183099 + vec3<f32>(0.1, 0.2, 0.3)) * 17.0;
    return fract(q.x * q.y * q.z * (q.x + q.y + q.z));
}

fn noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(
            mix(hash(i), hash(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(hash(i + vec3<f32>(0.0, 1.0, 0.0)), hash(i + vec3<f32>(1.0, 1.0, 0.0)), u.x),
            u.y,
        ),
        mix(
            mix(hash(i + vec3<f32>(0.0, 0.0, 1.0)), hash(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(hash(i + vec3<f32>(0.0, 1.0, 1.0)), hash(i + vec3<f32>(1.0, 1.0, 1.0)), u.x),
            u.y,
        ),
        u.z,
    );
}

/// Bake one voxel: new smoke from three swaying columns, plus last frame's
/// smoke carried up and faded. Motion and fading scale with the frame time,
/// so the look is the same at any frame rate and volume size.
/// @pass(target = "smoke", float, width = 128, height = 128, depth = 128)
@compute @workgroup_size(4, 4, 4)
fn bake(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(smoke_out);
    if any(id >= size) {
        return;
    }
    // Voxel centre in [-1, 1]^3, y up.
    let p = (vec3<f32>(id) + 0.5) / vec3<f32>(size) * 2.0 - 1.0;
    let t = globals.time * params.speed;

    var fresh = 0.0;
    for (var k = 0; k < 3; k++) {
        let a = f32(k) * 2.094 + t * 0.3;
        let centre = 0.45 * vec2<f32>(cos(a), sin(a)) + 0.08 * sin(p.y * 4.0 + t + f32(k));
        let r = length(p.xz - centre);
        fresh += exp(-r * r * 60.0);
    }
    // Only the floor emits; the feedback carries the smoke up.
    fresh *= smoothstep(-0.7, -0.9, p.y);
    fresh *= 0.5 + noise(p * 6.0 + vec3<f32>(0.0, -t * 2.0, 0.0));

    // Carry last frame's smoke: read it from where it was one frame ago.
    // Trilinear filtering moves it by fractions of a voxel.
    let dt = min(globals.time_delta, 0.1);
    let swirl = vec3<f32>(noise(p * 3.0 + t) - 0.5, 0.0, noise(p * 3.0 - t) - 0.5);
    let velocity = vec3<f32>(swirl.x, 1.0, swirl.z) * 0.8 * max(params.speed, 0.05);
    let source = (p - velocity * dt) * 0.5 + 0.5;
    let previous = textureSampleLevel(smoke, samp, source, 0.0).a;
    let fade = exp(-dt / max(params.linger, 0.01));
    let d = max(fresh, previous * fade);

    // Colour by height: hot at the floor, cooling as it rises.
    let heat = clamp(1.0 - (p.y + 1.0) * 0.5, 0.0, 1.0);
    let colour = mix(vec3<f32>(0.25, 0.3, 0.45), params.tint, heat * heat);
    textureStore(smoke_out, vec3<i32>(id), vec4<f32>(colour, d));
}

/// Where a ray enters and leaves the [-1, 1]^3 box; `x > y` misses.
fn box_hits(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
    let inv = 1.0 / rd;
    let a = (-1.0 - ro) * inv;
    let b = (1.0 - ro) * inv;
    let near = min(a, b);
    let far = max(a, b);
    return vec2<f32>(max(max(near.x, near.y), max(near.z, 0.0)), min(min(far.x, far.y), far.z));
}

@fragment
fn present(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let res = globals.resolution;
    let ndc = (uv * 2.0 - 1.0) * vec2<f32>(res.x / res.y, -1.0);
    let a = globals.time * 0.2;
    let ro = vec3<f32>(2.6 * sin(a), 0.5, 2.6 * cos(a));
    let forward = normalize(-ro);
    let right = normalize(cross(forward, vec3<f32>(0.0, 1.0, 0.0)));
    let up = cross(right, forward);
    let rd = normalize(forward * 1.8 + right * ndc.x + up * ndc.y);

    var colour = vec3<f32>(0.02, 0.02, 0.03) * (1.0 - uv.y);
    let hits = box_hits(ro, rd);
    if hits.x < hits.y {
        let steps = 96;
        let dt = (hits.y - hits.x) / f32(steps);
        var transmittance = 1.0;
        var light = vec3<f32>(0.0);
        for (var i = 0; i < steps; i++) {
            let p = ro + rd * (hits.x + (f32(i) + 0.5) * dt);
            let voxel = textureSampleLevel(smoke, samp, p * 0.5 + 0.5, 0.0);
            let sigma = voxel.a * params.density;
            let absorbed = 1.0 - exp(-sigma * dt);
            light += transmittance * absorbed * voxel.rgb * 2.0;
            transmittance *= 1.0 - absorbed;
            if transmittance < 0.01 {
                break;
            }
        }
        colour = colour * transmittance + light;
    }
    return vec4<f32>(colour, 1.0);
}
