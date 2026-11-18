//! Per-camera management of intermediate pass-target images.
//!
//! Each wisp camera gets a [`WispPassTargets`] component with one slot per pass.
//! Targets are (re)allocated when the view size, target size expression, format or
//! feedback-ness changes. Self-feedback passes get a ping-pong image pair whose
//! roles swap each frame, so a pass can read its own previous output while
//! writing the next. Volume targets are 3d images, sized by their own
//! `width`/`height`/`depth` expressions.

use crate::asset::{Wisp, WispHandle};
use crate::render::supports_compute;
use crate::schema::{
    PassSchema, PassStage, TargetSchema, WispSchema, eval_size, eval_size_3d, requires_compute,
};
use bevy_asset::prelude::{Assets, Handle};
use bevy_camera::{Camera, RenderTarget};
use bevy_ecs::prelude::*;
use bevy_image::prelude::Image;
use bevy_log::warn_once;
use bevy_math::{UVec2, UVec3};
use bevy_render::RenderApp;
use bevy_render::extract_component::ExtractComponent;
use bevy_render::render_resource::{Extent3d, TextureDimension, TextureUsages};
use bevy_render::renderer::RenderAdapter;
use bevy_window::{PrimaryWindow, Window, WindowRef};

/// The intermediate target images for a wisp camera, parallel to the schema's
/// passes (`None` for the final pass).
#[derive(Component, ExtractComponent, Clone, Debug, Default)]
#[extract_app(RenderApp)]
pub struct WispPassTargets(pub Vec<Option<PassTarget>>);

#[derive(Clone, Debug)]
pub struct PassTarget {
    /// The target's size in pixels; `z` is the depth of a volume, 1 otherwise.
    pub size: UVec3,
    /// Clear the write image at the start of the pass (non-persistent targets).
    /// Compute targets are never cleared - the shader overwrites what it covers.
    pub clear: bool,
    /// Workgroup counts for compute passes (`None` for fragment passes).
    pub dispatch: Option<[u32; 3]>,
    /// `[a, b]` ping-pong pair for self-feedback passes; otherwise the same
    /// handle twice.
    images: [Handle<Image>; 2],
    flip: bool,
}

impl PassTarget {
    /// The image the pass writes this frame.
    pub fn write(&self) -> &Handle<Image> {
        &self.images[self.flip as usize]
    }

    /// The image holding the previous frame's contents.
    pub fn read(&self) -> &Handle<Image> {
        &self.images[1 - self.flip as usize]
    }
}

/// Keep every wisp camera's pass targets in sync with its schema and view size.
pub(crate) fn update_pass_targets(
    mut commands: Commands,
    // Absent when there is no render backend (e.g. headless); then there is no
    // GPU to allocate against, so the compute-support gate below is skipped.
    render_adapter: Option<Res<RenderAdapter>>,
    wisps: Res<Assets<Wisp>>,
    windows: Query<(&Window, Option<&PrimaryWindow>)>,
    mut images: ResMut<Assets<Image>>,
    mut cameras: Query<(
        Entity,
        &Camera,
        &RenderTarget,
        &WispHandle,
        Option<&mut WispPassTargets>,
    )>,
) {
    for (entity, camera, render_target, wisp, targets) in cameras.iter_mut() {
        let Some(wisp) = wisps.get(&**wisp) else {
            continue;
        };
        // A compute wisp needs a storage-texture target, which the device
        // cannot allocate without compute support (e.g. WebGL2). Skip it - the
        // render layer skips the wisp too and surfaces the reason on screen.
        if requires_compute(&wisp.schema)
            && render_adapter
                .as_ref()
                .is_some_and(|adapter| !supports_compute(adapter))
        {
            continue;
        }
        // A camera viewport (e.g. when sharing the window with UI panels)
        // bounds the view; targets and `$WIDTH`/`$HEIGHT` follow it.
        let viewport_size = camera.viewport.as_ref().map(|v| v.physical_size);
        let Some(view_size) =
            viewport_size.or_else(|| render_target_size(render_target, &windows, &images))
        else {
            warn_once!("wisp: unsupported camera render target; only windows and images work");
            continue;
        };
        match targets {
            Some(mut targets) => sync_targets(&mut targets, &wisp.schema, view_size, &mut images),
            None => {
                let mut targets = WispPassTargets::default();
                sync_targets(&mut targets, &wisp.schema, view_size, &mut images);
                commands.entity(entity).insert(targets);
            }
        }
    }
}

/// The pixel size of a camera's render target, if it is a window or image.
fn render_target_size(
    render_target: &RenderTarget,
    windows: &Query<(&Window, Option<&PrimaryWindow>)>,
    images: &Assets<Image>,
) -> Option<UVec2> {
    match render_target {
        RenderTarget::Window(WindowRef::Primary) => windows
            .iter()
            .find(|(_, primary)| primary.is_some())
            .map(|(window, _)| window.resolution.physical_size()),
        RenderTarget::Window(WindowRef::Entity(entity)) => windows
            .get(*entity)
            .ok()
            .map(|(window, _)| window.resolution.physical_size()),
        RenderTarget::Image(target) => images.get(&target.handle).map(|image| image.size()),
        RenderTarget::TextureView(_) | RenderTarget::None { .. } => None,
    }
}

fn sync_targets(
    targets: &mut WispPassTargets,
    schema: &WispSchema,
    view_size: UVec2,
    images: &mut Assets<Image>,
) {
    targets.0.resize(schema.passes.len(), None);
    for (index, pass) in schema.passes.iter().enumerate() {
        let slot = &mut targets.0[index];
        let Some(target) = &pass.target else {
            *slot = None;
            continue;
        };
        let size = target_size(target, view_size);
        let clear = !target.persistent;
        let volume = target.is_volume();
        let dispatch = match pass.stage {
            PassStage::Fragment => None,
            PassStage::Compute => Some(dispatch_size(pass, size)),
        };
        let up_to_date = slot.as_ref().is_some_and(|t| {
            t.size == size
                && t.clear == clear
                && t.dispatch == dispatch
                && (t.images[0] != t.images[1]) == pass.self_feedback
                && images.get(&t.images[0]).is_some_and(|image| {
                    image.texture_descriptor.format == target.format
                        && (image.texture_descriptor.dimension == TextureDimension::D3) == volume
                })
        });
        if up_to_date {
            // Keep the contents; just advance the ping-pong.
            if let Some(t) = slot
                && pass.self_feedback
            {
                t.flip = !t.flip;
            }
            continue;
        }
        let mut usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
        if !volume {
            usage |= TextureUsages::RENDER_ATTACHMENT;
        }
        if pass.stage == PassStage::Compute {
            usage |= TextureUsages::STORAGE_BINDING;
        }
        let mut new_image = || {
            let mut image = Image::default();
            image.texture_descriptor.format = target.format;
            image.texture_descriptor.usage = usage;
            let extent = Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: size.z,
            };
            if volume {
                // No CPU copy: a volume is filled on the GPU, and its size
                // (e.g. 256^3 at 8 bytes a voxel) is too large to mirror.
                image.texture_descriptor.dimension = TextureDimension::D3;
                image.texture_descriptor.size = extent;
                image.data = None;
            } else {
                image.resize(extent);
            }
            images.add(image)
        };
        let first = new_image();
        let second = match pass.self_feedback {
            true => new_image(),
            false => first.clone(),
        };
        *slot = Some(PassTarget {
            size,
            clear,
            dispatch,
            images: [first, second],
            flip: false,
        });
    }
}

/// A compute pass's workgroup counts: explicit `dispatch = ".."` expressions
/// evaluated against the *target* size, or derived as `ceil(target / workgroup)`.
fn dispatch_size(pass: &PassSchema, target_size: UVec3) -> [u32; 3] {
    match &pass.dispatch {
        Some(exprs) => {
            let eval = |expr: &String| {
                eval_size_3d(expr, target_size).unwrap_or_else(|err| {
                    warn_once!("wisp: {err}; dispatching a single workgroup");
                    1
                })
            };
            [eval(&exprs[0]), eval(&exprs[1]), eval(&exprs[2])]
        }
        None => {
            let [x, y, z] = pass.workgroup_size;
            [
                target_size.x.div_ceil(x.max(1)),
                target_size.y.div_ceil(y.max(1)),
                target_size.z.div_ceil(z.max(1)),
            ]
        }
    }
}

/// A target's pixel size for the given view size; `z` is the depth of a
/// volume, 1 otherwise.
///
/// Expressions were validated at load time; should one still fail for this
/// particular view size (e.g. going non-positive), fall back to the view size
/// (and a depth of 1).
fn target_size(target: &TargetSchema, view_size: UVec2) -> UVec3 {
    let dim = |expr: &Option<String>, base: u32| match expr {
        None => base,
        Some(expr) => eval_size(expr, view_size).unwrap_or_else(|err| {
            warn_once!("wisp: {err}; falling back to the view size");
            base
        }),
    };
    UVec3::new(
        dim(&target.width, view_size.x),
        dim(&target.height, view_size.y),
        dim(&target.depth, 1),
    )
}
