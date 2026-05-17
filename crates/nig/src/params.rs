use serde::Serialize;

/// NIG distribution parameters (scipy loc-scale parameterization).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct NigParams {
    pub a: f64,
    pub b: f64,
    pub loc: f64,
    pub scale: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NigParamsReparametrized {
    pub log_a: f64,
    pub skew_raw: f64,
    pub loc: f64,
    pub log_scale: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct NigData {
    pub a: f64,
    pub b: f64,
    pub loc: f64,
    pub scale: f64,
    pub top_scale: f64,
}

impl NigData {
    pub fn params(&self) -> NigParams {
        NigParams { a: self.a, b: self.b, loc: self.loc, scale: self.scale }
    }
}
