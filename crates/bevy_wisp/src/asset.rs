//! The [`Wisp`] asset and its loader.
//!
//! Wisp shaders are plain `.wgsl` or `.wesl` files, so [`WispLoader`] registers
//! no extensions of its own: load one explicitly with
//! `asset_server.load::<Wisp>("path.wgsl")`, which resolves the loader by asset
//! type. Untyped `.wgsl` and `.wesl` loads (including bevy's own `Shader`
//! loading) are unaffected.
//!
//! A `.wesl` wisp is first linked to WGSL (see [`crate::link`]); the local
//! modules it imports are read as loader dependencies, so editing one reloads
//! the wisp.
//!
//! The compiled [`Shader`] is stored as a labeled sub-asset (`"shader"`) of the
//! `Wisp`, so the same source is never loaded through bevy's shader loader, and
//! replacing it on hot reload recompiles dependent pipelines automatically.

use crate::WispConfig;
use crate::link::{LinkError, StaticPackage, WeslSources};
use crate::reflect::{self, ReflectError};
use crate::schema::{self, SchemaError, WispSchema};
use bevy_asset::io::Reader;
use bevy_asset::prelude::*;
use bevy_asset::{AssetLoader, AssetPath, LoadContext};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::TypePath;
use bevy_render::RenderApp;
use bevy_render::extract_component::ExtractComponent;
use bevy_shader::Shader;
use thiserror::Error;

/// A loaded wisp shader: its reflected interface and compiled shader module.
#[derive(Asset, TypePath, Debug, Clone)]
pub struct Wisp {
    pub schema: WispSchema,
    pub shader: Handle<Shader>,
}

/// Renders the wisp to a camera: insert on a camera entity, pointing at a loaded
/// [`Wisp`]. Output goes wherever the camera renders - a window, or an `Image`
/// render target.
#[derive(Component, ExtractComponent, Deref, DerefMut, Debug, Default, Clone)]
#[extract_app(RenderApp)]
pub struct WispHandle(pub Handle<Wisp>);

/// Loads [`Wisp`]s, linking `.wesl` ones against the packages in
/// [`WispConfig::wesl_packages`].
#[derive(TypePath)]
pub struct WispLoader {
    packages: Vec<&'static StaticPackage>,
}

impl FromWorld for WispLoader {
    fn from_world(world: &mut World) -> Self {
        let config = world.get_resource::<WispConfig>();
        let packages = config.map(|config| config.wesl_packages.clone());
        Self {
            packages: packages.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Error)]
pub enum WispLoadError {
    #[error("failed to read wisp shader: {0}")]
    Io(#[from] std::io::Error),
    #[error("wisp shader is not valid UTF-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("failed to link WESL:\n{0}")]
    Link(#[from] LinkError),
    #[error(transparent)]
    Reflect(#[from] ReflectError),
    #[error(transparent)]
    Schema(#[from] SchemaError),
}

impl AssetLoader for WispLoader {
    type Asset = Wisp;
    type Settings = ();
    type Error = WispLoadError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let source = std::str::from_utf8(&bytes)?;
        let is_wesl = load_context.path().get_extension() == Some("wesl");
        let linked;
        let source = if is_wesl {
            linked = self.link(source, load_context).await?;
            linked.as_str()
        } else {
            source
        };
        let reflected = reflect::parse_and_validate(source)?;
        let schema = schema::schema_from_module(&reflected)?;
        let path = load_context.path().to_string();
        // Pad the uniform structs to a 16-byte multiple so their types are
        // accepted on devices without `BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED`
        // (e.g. WebGL2); the schema rounds its sizes to match.
        let padded = crate::align::pad_uniform_structs(source, &reflected.module);
        let shader = Shader::from_wgsl(padded, path);
        let shader = load_context.add_labeled_asset(String::from("shader"), shader);
        Ok(Wisp { schema, shader })
    }

    // No extensions: wisp shaders are plain `.wgsl` or `.wesl`, loaded by asset
    // type so that bevy's own `Shader` loader keeps owning the extensions.
    fn extensions(&self) -> &[&str] {
        &[]
    }
}

impl WispLoader {
    /// Link a `.wesl` wisp to WGSL, reading the local modules it imports from the
    /// wisp's own asset source as they are needed.
    async fn link(
        &self,
        source: &str,
        load_context: &mut LoadContext<'_>,
    ) -> Result<String, LinkError> {
        let asset_source = load_context.path().source().clone_owned();
        let mut sources = WeslSources::new(asset_path_string(load_context.path()), source);
        loop {
            let (module, message) = match sources.link(&self.packages) {
                Err(LinkError::MissingModule { module, message }) => (module, message),
                linked => return linked,
            };
            let mut found = None;
            for extension in ["wesl", "wgsl"] {
                let path = format!("{module}.{extension}");
                let asset_path = AssetPath::from(path.clone()).with_source(asset_source.clone());
                if let Ok(bytes) = load_context.read_asset_bytes(asset_path).await {
                    found = Some((path, bytes));
                    break;
                }
            }
            let Some((path, bytes)) = found else {
                return Err(LinkError::MissingModule { module, message });
            };
            let module_source = String::from_utf8(bytes)
                .map_err(|error| LinkError::Wesl(format!("{path} is not valid UTF-8: {error}")))?;
            sources.insert(path, module_source);
        }
    }
}

/// An asset's path within its source, `/`-separated on every platform. An
/// absolute path stays absolute, so its local modules are read beside it.
fn asset_path_string(path: &AssetPath<'_>) -> String {
    path.path().to_string_lossy().replace('\\', "/")
}
