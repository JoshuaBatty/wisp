//! Linking `.wesl` wisps to WGSL.
//!
//! A wisp may be written in [WESL](https://wesl-lang.dev): WGSL plus `import`s
//! and `@if` conditional compilation. Before reflection, a `.wesl` wisp is
//! linked into one plain WGSL module, and from there it is handled exactly like
//! a `.wgsl` wisp: the linked WGSL is what wisp reflects and what the GPU
//! compiles.
//!
//! Imports resolve two ways:
//!
//! - **Packages**, such as `import my_lib::noise::fbm;`, resolve to the
//!   [`StaticPackage`]s in [`WispConfig::wesl_packages`](crate::WispConfig::wesl_packages)
//!   (built with `wesl`'s `PackageBuilder` and `wesl_pkg!`).
//! - **Local modules**, such as `import super::common::palette;` or
//!   `import package::wisp::common::palette;`, resolve to other assets. The
//!   module path is the asset path without its extension (`wisp/common`), read
//!   as `wisp/common.wesl`, or `wisp/common.wgsl` if that does not exist.
//!   `super::` is relative to the wisp's own path, which may be absolute, and
//!   `package::` is the root of the asset source. Local modules are loader
//!   dependencies, so editing one hot-reloads every wisp that imports it.
//!
//! Every `@if` feature flag is off. Declarations in the wisp's own file are
//! never renamed or removed, and their `///` doc comments are carried into the
//! linked WGSL, so annotations work as they do in a `.wgsl` wisp. Imported
//! declarations get unique hashed names (such as `palette_1234`), and unused
//! ones are removed.

pub use wesl::package::{StaticPackage, StaticPackageModule};

use std::borrow::Cow;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::HashMap;
use thiserror::Error;
use wesl::error::ResolveError;
use wesl::resolver::PackageResolver;
use wesl::syntax::{GlobalDeclaration, ModulePath, PathOrigin, TranslationUnit};

/// The WESL sources of one wisp: its own file and the local modules it imports.
///
/// Link with [`WeslSources::link`]. Local modules are added on demand: when
/// linking reports a [`LinkError::MissingModule`], read that module, add it with
/// [`WeslSources::insert`] and link again.
#[derive(Clone, Debug)]
pub struct WeslSources {
    /// The module path of the wisp's own file.
    root: String,
    /// Each module's asset path and source, keyed by module path.
    modules: HashMap<String, (String, String)>,
}

/// Why a wisp could not be linked.
#[derive(Clone, Debug, Error)]
pub enum LinkError {
    /// The wisp imports a local module that has not been added yet. Read
    /// `<module>.wesl` (or `<module>.wgsl` if that does not exist), add it with
    /// [`WeslSources::insert`] and link again. If neither file exists, `message`
    /// is the error to report.
    #[error("{message}")]
    MissingModule {
        /// The module path: the asset path without its extension.
        module: String,
        message: String,
    },
    /// The WESL is invalid: a syntax error, an unknown package or declaration,
    /// and so on.
    #[error("{0}")]
    Wesl(String),
}

impl WeslSources {
    /// The sources of the wisp at asset path `path` (such as `wisp/scene.wesl`),
    /// whose file holds `source`.
    pub fn new(path: impl Into<String>, source: impl Into<String>) -> Self {
        let mut sources = Self {
            root: String::new(),
            modules: HashMap::new(),
        };
        sources.root = sources.insert(path, source);
        sources
    }

    /// Add the local module at asset path `path` (such as `wisp/common.wesl`),
    /// returning its module path.
    pub fn insert(&mut self, path: impl Into<String>, source: impl Into<String>) -> String {
        let path = path.into();
        let module = module_of(&path).to_string();
        self.modules.insert(module.clone(), (path, source.into()));
        module
    }

    /// Link like [`WeslSources::link`], reading each missing local module with
    /// `read` until none is missing.
    ///
    /// `read` gets a module path (such as `wisp/common`) and returns that
    /// module's asset path and source, or `None` if neither `<module>.wesl` nor
    /// `<module>.wgsl` exists. This suits callers that read synchronously, such
    /// as tools that reflect wisps from disk; the asset loader reads through the
    /// asset server instead.
    pub fn link_with(
        &mut self,
        packages: &[&'static StaticPackage],
        mut read: impl FnMut(&str) -> Option<(String, String)>,
    ) -> Result<String, LinkError> {
        loop {
            match self.link(packages) {
                Err(LinkError::MissingModule { module, message }) => {
                    let (path, source) =
                        read(&module).ok_or(LinkError::MissingModule { module, message })?;
                    self.insert(path, source);
                }
                linked => return linked,
            }
        }
    }

    /// Link the wisp and everything it imports into one WGSL module, resolving
    /// package imports against `packages`.
    pub fn link(&self, packages: &[&'static StaticPackage]) -> Result<String, LinkError> {
        let mut resolver = Resolver {
            sources: self,
            packages: PackageResolver::new(),
            missing: RefCell::new(None),
        };
        for package in packages {
            resolver.packages.add_package(package);
        }
        let options = wesl::CompileOptions {
            // Keep every declaration in the wisp's own file, even unused ones:
            // a `.wgsl` wisp keeps them too, and its params and bindings are
            // reflected whether or not a pass reads them.
            keep_main: true,
            // The linked WGSL is validated by naga during reflection.
            validate: false,
            // Imported names are made from module paths, which are asset paths
            // and may hold characters that are not valid in WGSL names. Hashed
            // names are always valid.
            mangler: wesl::ManglerKind::Hash,
            ..Default::default()
        };
        let root = module_path(&self.root);
        let linked = match wesl::compile(&root, &options, &resolver) {
            Ok(linked) => linked.to_string(),
            Err(error) => {
                let message = error.diagnostic().render_plain();
                return Err(match resolver.missing.into_inner() {
                    Some(module) => LinkError::MissingModule { module, message },
                    None => LinkError::Wesl(message),
                });
            }
        };
        let (_, source) = &self.modules[&self.root];
        carry_docs(source, linked)
    }
}

/// Resolves package modules from static packages and local modules from a
/// [`WeslSources`], recording the last local module it could not find.
struct Resolver<'a> {
    sources: &'a WeslSources,
    packages: PackageResolver,
    missing: RefCell<Option<String>>,
}

impl wesl::Resolver for Resolver<'_> {
    fn resolve_source<'a>(&'a self, path: &ModulePath) -> Result<Cow<'a, str>, ResolveError> {
        if let PathOrigin::Package(_) = path.origin {
            return self.packages.resolve_source(path);
        }
        let module = path.components.join("/");
        match self.sources.modules.get(&module) {
            Some((_, source)) => Ok(Cow::Borrowed(source)),
            None => {
                *self.missing.borrow_mut() = Some(module.clone());
                Err(ResolveError::ModuleNotFound(
                    path.clone(),
                    format!("no `{module}.wesl` or `{module}.wgsl` asset"),
                ))
            }
        }
    }

    fn display_name(&self, path: &ModulePath) -> Option<String> {
        if let PathOrigin::Package(_) = path.origin {
            return Some(path.to_string());
        }
        let module = path.components.join("/");
        let shown = self.sources.modules.get(&module).map(|(path, _)| path);
        Some(shown.cloned().unwrap_or(module))
    }
}

/// The module path of an asset path: the path without the file's extension.
fn module_of(path: &str) -> &str {
    let name_start = path.rfind('/').map_or(0, |slash| slash + 1);
    match path[name_start..].rfind('.') {
        Some(dot) => &path[..name_start + dot],
        None => path,
    }
}

/// A `/`-separated module path as a WESL module path.
fn module_path(module: &str) -> ModulePath {
    let components = module.split('/').map(String::from).collect();
    ModulePath::new(PathOrigin::Absolute, components)
}

/// A declaration that can carry doc comments: a global declaration, or a member
/// of the named struct.
type Item = (String, Option<String>);

/// Copy the doc comments of `source`'s declarations onto the same declarations
/// in `linked`, its linked WGSL.
///
/// WESL linking drops comments. Declarations in the wisp's own file keep their
/// names when linked, so each one's docs are found again by name.
fn carry_docs(source: &str, mut linked: String) -> Result<String, LinkError> {
    let parse = |text: &str| {
        text.parse::<TranslationUnit>()
            .map_err(|error| LinkError::Wesl(error.to_string()))
    };
    let docs: HashMap<Item, Vec<&str>> = item_starts(&parse(source)?)
        .into_iter()
        .filter_map(|(item, start)| {
            let docs = docs_before(source, start);
            (!docs.is_empty()).then_some((item, docs))
        })
        .collect();
    let mut inserts: Vec<(usize, String)> = item_starts(&parse(&linked)?)
        .into_iter()
        .filter_map(|(item, start)| {
            let docs = docs.get(&item)?;
            let line_start = linked[..start].rfind('\n').map_or(0, |newline| newline + 1);
            let indent = &linked[line_start..start];
            let (at, indent) = if indent.trim().is_empty() {
                (line_start, indent)
            } else {
                (start, "")
            };
            let text = docs.iter().map(|doc| format!("{indent}{doc}\n")).collect();
            Some((at, text))
        })
        .collect();
    // Insert from the end first so earlier offsets stay valid.
    inserts.sort_by_key(|&(at, _)| Reverse(at));
    for (at, text) in inserts {
        linked.insert_str(at, &text);
    }
    let module_docs: String = module_docs(source).map(|doc| format!("{doc}\n")).collect();
    linked.insert_str(0, &module_docs);
    Ok(linked)
}

/// Every declaration of `unit` that can carry doc comments, with the byte
/// offset where it starts (its first attribute, if it has any).
fn item_starts(unit: &TranslationUnit) -> Vec<(Item, usize)> {
    let mut items = Vec::new();
    for declaration in &unit.global_declarations {
        let name = match declaration.node() {
            GlobalDeclaration::Declaration(declaration) => declaration.ident.to_string(),
            GlobalDeclaration::TypeAlias(alias) => alias.ident.to_string(),
            GlobalDeclaration::Function(function) => function.ident.to_string(),
            GlobalDeclaration::Struct(structure) => {
                let name = structure.ident.to_string();
                for member in &structure.members {
                    let item = (name.clone(), Some(member.ident.to_string()));
                    items.push((item, member.span().start));
                }
                name
            }
            _ => continue,
        };
        items.push(((name, None), declaration.span().start));
    }
    items
}

/// The doc comments directly above byte offset `start`: `///` lines and
/// `/** */` blocks, in source order.
fn docs_before(source: &str, start: usize) -> Vec<&str> {
    let mut docs = Vec::new();
    let mut rest = source[..start].trim_end();
    loop {
        if rest.ends_with("*/") {
            match rest.rfind("/**") {
                Some(open) if !rest[open..].starts_with("/***") => {
                    docs.push(&rest[open..]);
                    rest = rest[..open].trim_end();
                    continue;
                }
                _ => break,
            }
        }
        let line_start = rest.rfind('\n').map_or(0, |newline| newline + 1);
        let line = rest[line_start..].trim_start();
        if !line.starts_with("///") || line.starts_with("////") {
            break;
        }
        docs.push(line);
        rest = rest[..line_start].trim_end();
    }
    docs.reverse();
    docs
}

/// The module's `//!` doc comment lines, which open the file.
fn module_docs(source: &str) -> impl Iterator<Item = &str> {
    source
        .lines()
        .map(str::trim)
        .take_while(|line| line.is_empty() || line.starts_with("//"))
        .filter(|line| line.starts_with("//!"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reflect::parse_and_validate;
    use crate::schema::{UiHints, schema_from_module};

    /// Link `sources`, reading local modules from `files` as they are needed.
    fn link_with(
        mut sources: WeslSources,
        files: &[(&str, &str)],
        packages: &[&'static StaticPackage],
    ) -> Result<String, LinkError> {
        sources.link_with(packages, |module| {
            let wesl = format!("{module}.wesl");
            let (path, source) = files.iter().find(|(path, _)| *path == wesl)?;
            Some((path.to_string(), source.to_string()))
        })
    }

    const SCENE: &str = "//! A scene that tints a gradient.
import super::lib::palette::tint;

struct Params {
    /// How strong the tint is.
    /// @min(0.0) @max(2.0) @default(0.5)
    level: f32,
    /** @color @default(1.0, 0.5, 0.0, 1.0) */
    colour: vec4<f32>,
    unused: f32,
}
@group(1) @binding(0) var<uniform> params: Params;

/// @pass(target = \"gradient\")
@fragment
fn draw(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(tint(uv.x, params.level), 1.0);
}

@group(1) @binding(1) var gradient: texture_2d<f32>;
@group(1) @binding(2) var gradient_sampler: sampler;

@fragment
fn fragment(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(gradient, gradient_sampler, uv) * params.colour;
}
";

    const PALETTE: &str = "
fn tint(x: f32, level: f32) -> vec3<f32> {
    return vec3<f32>(x * level, ramp(x), 0.5);
}
fn ramp(x: f32) -> f32 { return x * x; }
fn never_used() {}
";

    #[test]
    fn links_local_modules_and_keeps_annotations() {
        let sources = WeslSources::new("wisp/scene.wesl", SCENE);
        let wgsl = link_with(sources, &[("wisp/lib/palette.wesl", PALETTE)], &[]).unwrap();
        assert!(
            !wgsl.contains("never_used"),
            "unused imports are removed:\n{wgsl}"
        );
        let schema = schema_from_module(&parse_and_validate(&wgsl).unwrap()).unwrap();
        assert_eq!(schema.description, "A scene that tints a gradient.");
        let params = schema.params.unwrap();
        let names: Vec<_> = params
            .fields
            .iter()
            .map(|field| field.name.as_str())
            .collect();
        assert_eq!(names, ["level", "colour", "unused"]);
        let level = &params.fields[0].ui;
        assert_eq!(level.description, "How strong the tint is.");
        assert_eq!((level.min, level.max), (Some(0.0), Some(2.0)));
        assert_eq!(level.default, Some(vec![0.5]));
        assert!(params.fields[1].ui.color);
        assert_eq!(params.fields[2].ui, UiHints::default());
        assert_eq!(schema.passes.len(), 2);
        let target = schema.passes[0]
            .target
            .as_ref()
            .map(|target| target.name.as_str());
        assert_eq!(target, Some("gradient"));
    }

    /// A wisp loaded by an absolute path, with characters that are not valid in
    /// WGSL names, still finds its local modules beside it.
    #[test]
    fn links_wisps_at_absolute_paths() {
        let sources = WeslSources::new("/my shaders/Study 008/scene.wesl", SCENE);
        let files = [("/my shaders/Study 008/lib/palette.wesl", PALETTE)];
        let wgsl = link_with(sources, &files, &[]).unwrap();
        parse_and_validate(&wgsl).unwrap();
    }

    #[test]
    fn reports_each_missing_local_module() {
        let sources = WeslSources::new("wisp/scene.wesl", SCENE);
        let Err(LinkError::MissingModule { module, message }) = link_with(sources, &[], &[]) else {
            panic!("expected a missing module");
        };
        assert_eq!(module, "wisp/lib/palette");
        assert!(message.contains("wisp/lib/palette"), "{message}");
    }

    static SHAPES: StaticPackage = StaticPackage {
        crate_name: "shapes",
        root: &StaticPackageModule {
            name: "shapes",
            source: "",
            submodules: &[&StaticPackageModule {
                name: "circle",
                source: "public fn circle(p: vec2<f32>, r: f32) -> f32 { return length(p) - r; }",
                submodules: &[],
            }],
        },
        dependencies: &[],
    };

    const USES_SHAPES: &str = "import shapes::circle::circle;
@fragment
fn fragment(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(step(circle(uv - 0.5, 0.25), 0.0));
}
";

    #[test]
    fn links_static_packages() {
        let sources = WeslSources::new("circle.wesl", USES_SHAPES);
        let wgsl = sources.link(&[&SHAPES]).unwrap();
        parse_and_validate(&wgsl).unwrap();
    }

    /// The bundled `wesl` example, with the same `shapes` package it registers.
    #[test]
    fn bundled_wesl_example_links() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/assets");
        let read = |path: &str| std::fs::read_to_string(format!("{dir}/{path}")).unwrap();
        let sources = WeslSources::new("wisp/test_wesl.wesl", read("wisp/test_wesl.wesl"));
        let palette = read("wisp/lib/palette.wesl");
        let files = [("wisp/lib/palette.wesl", palette.as_str())];
        let wgsl = link_with(sources, &files, &[&SHAPES]).unwrap();
        let schema = schema_from_module(&parse_and_validate(&wgsl).unwrap()).unwrap();
        let params = schema.params.unwrap();
        let defaults: Vec<_> = params
            .fields
            .iter()
            .map(|field| field.ui.default.clone())
            .collect();
        assert_eq!(defaults, [Some(vec![0.3]), Some(vec![12.0])]);
    }

    #[test]
    fn unknown_packages_are_errors() {
        let sources = WeslSources::new("circle.wesl", USES_SHAPES);
        let Err(LinkError::Wesl(message)) = sources.link(&[]) else {
            panic!("expected a WESL error");
        };
        assert!(message.contains("shapes"), "{message}");
    }

    /// Every bundled `.wgsl` example is also valid WESL; linked as WESL it must
    /// reflect to exactly the same interface, annotations included.
    #[test]
    fn wgsl_examples_reflect_the_same_when_linked() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/wisp");
        let mut checked = 0;
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|extension| extension != "wgsl") {
                continue;
            }
            let source = std::fs::read_to_string(&path).unwrap();
            let wgsl = schema_from_module(&parse_and_validate(&source).unwrap()).unwrap();
            let linked = WeslSources::new("wisp/example.wgsl", source.as_str())
                .link(&[])
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            let wesl = schema_from_module(&parse_and_validate(&linked).unwrap()).unwrap();
            assert_eq!(wgsl, wesl, "{}", path.display());
            checked += 1;
        }
        assert!(checked > 0);
    }

    #[test]
    fn docs_before_reads_line_and_block_comments() {
        let source = "fn a() {}\n// plain\n/// one\n/** two */\n  /// three\n@fragment fn b() {}";
        let start = source.find("@fragment").unwrap();
        assert_eq!(
            docs_before(source, start),
            ["/// one", "/** two */", "/// three"]
        );
        assert!(docs_before(source, source.find("fn a").unwrap()).is_empty());
    }

    /// The loader reads local modules through the asset server and links
    /// packages from `WispConfig`.
    #[test]
    fn loader_links_the_bundled_wesl_example() {
        use crate::WispConfig;
        use crate::asset::{Wisp, WispLoader};
        use bevy::asset::LoadState;
        use bevy::prelude::*;
        use bevy::shader::Shader;

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .insert_resource(WispConfig {
                wesl_packages: vec![&SHAPES],
                ..default()
            })
            .init_asset::<Shader>()
            .init_asset::<Wisp>()
            .init_asset_loader::<WispLoader>();
        let wisp: Handle<Wisp> = app
            .world()
            .resource::<AssetServer>()
            .load("wisp/test_wesl.wesl");
        for _ in 0..500 {
            app.update();
            match app.world().resource::<AssetServer>().load_state(&wisp) {
                LoadState::Loaded => break,
                LoadState::Failed(error) => panic!("{error}"),
                _ => std::thread::sleep(std::time::Duration::from_millis(10)),
            }
        }
        let wisps = app.world().resource::<Assets<Wisp>>();
        let params = wisps
            .get(&wisp)
            .expect("loaded")
            .schema
            .params
            .clone()
            .unwrap();
        let names: Vec<_> = params
            .fields
            .iter()
            .map(|field| field.name.as_str())
            .collect();
        assert_eq!(names, ["radius", "rings"]);
    }

    #[test]
    fn module_of_strips_only_the_extension() {
        assert_eq!(module_of("wisp/scene.wesl"), "wisp/scene");
        assert_eq!(module_of("my.shaders/scene.wesl"), "my.shaders/scene");
        assert_eq!(module_of("scene"), "scene");
    }
}
