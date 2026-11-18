//! A wisp with a volume pass, Shadertoy style: a `@compute` pass bakes a 3d
//! target (reading its own previous frame), and the final fragment pass
//! raymarches it. Pass another shader's asset path as an argument to run that
//! instead.

use bevy::prelude::*;
use bevy_wisp::prelude::*;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: String::from("bevy_wisp - volume"),
                    ..default()
                }),
                ..default()
            }),
            WispPlugin,
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| String::from("wisp/test_volume.wgsl"));
    let wisp: Handle<Wisp> = asset_server.load(path);
    commands.spawn((Camera3d::default(), WispHandle(wisp)));
}
