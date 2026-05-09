use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct NigParams {
    pub a: f64, // alpha
    pub b: f64, // beta 
    pub loc: f64, // mu
    pub scale: f64, // delta
    pub top_scale: f64, // used to normalize the pdf
}

#[derive(Debug, Clone)]
pub struct FitResult {
    pub params: NigParams,
    pub valid: bool,
}
