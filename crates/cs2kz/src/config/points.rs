use serde::Deserialize;

/// Configuration for the points system.
///
/// All distribution fitting and recalculation is now done in pure Rust
/// (see `crate::points::calculator`), so no external script paths are needed.
#[derive(Debug, Default, Deserialize)]
#[serde(default, rename_all = "kebab-case", deny_unknown_fields)]
pub struct PointsConfig {
    // (no fields needed — kept for future extensibility)
}
