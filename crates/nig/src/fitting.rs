use crate::bessel::bessel_k1e;
use crate::distribution::nig_survival;
use crate::params::NigParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OptimizeNigError {
    MaxIterationsReached,
    StepSizeExhausted,
}

fn central_diff(mut f: impl FnMut(f64) -> f64, x0: f64) -> f64 {
    let eps = f64::EPSILON;
    let h = eps.cbrt() * x0.abs().max(1.0); // https://docs.sciml.ai/FiniteDiff/dev/epsilons/
    let x_plus = x0 + h;
    let x_minus = x0 - h;
    let denom = x_plus - x_minus; // representable floating point nb (not same as 2h)
    (f(x_plus) - f(x_minus)) / denom
}

pub fn fit_nig(times: &[f64], prev_params: Option<&NigParams>) -> Option<NigParams> {
    let (a, b, loc, scale);

    if let Some(prev) = prev_params.filter(|p| p.a > 0.0) {
        a = prev.a;
        b = prev.b;
        loc = prev.loc;
        scale = prev.scale;
    } else {
        loc = estimate_location(times);
        scale = estimate_scale(times, loc);
        a = estimate_alpha(times, loc, scale);
        b = estimate_beta(times, loc, scale, a);
    }

    let (a, b, loc, scale) = match optimize_nig(times, a, b, loc, scale) {
        Ok(params) => params,
        Err(err) => {
            tracing::warn!(?err, samples = times.len(), "NIG optimization failed; using initial estimates");
            (a, b, loc, scale)
        }
    };

    let top_scale = {
        let sf = nig_survival(a, b, loc, scale, times[0]);
        if sf <= 0.0 { 1.0 } else { sf }
    };

    Some(NigParams { a, b, loc, scale, top_scale })
}

/// Median-based location estimate.
fn estimate_location(times: &[f64]) -> f64 {
    let n = times.len();
    if n % 2 == 0 {
        (times[n / 2 - 1] + times[n / 2]) / 2.0
    } else {
        times[n / 2]
    }
}

/// Standard-deviation-based scale estimate.
fn estimate_scale(times: &[f64], loc: f64) -> f64 {
    let n = times.len() as f64;
    let variance = times.iter().map(|&t| (t - loc).powi(2)).sum::<f64>() / n;
    variance.max(1e-10).sqrt()
}

/// Kurtosis-based estimate
fn estimate_alpha(times: &[f64], loc: f64, _scale: f64) -> f64 {
    let n = times.len() as f64;
    let (mut m2, mut m4) = (0.0, 0.0);
    for &t in times {
        let d = t - loc;
        m2 += d * d;
        m4 += d * d * d * d;
    }
    m2 /= n;
    m4 /= n;

    if m2 < 1e-10 {
        return 1.0;
    }

    let kurtosis = m4 / (m2 * m2);
    1.0_f64.max(3.0 / (kurtosis - 3.0).max(0.1))
}

/// Skewness-based estimate
fn estimate_beta(times: &[f64], loc: f64, _scale: f64, alpha: f64) -> f64 {
    let n = times.len() as f64;
    let (mut m2, mut m3) = (0.0, 0.0);
    for &t in times {
        let d = t - loc;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n;
    m3 /= n;

    if m2 < 1e-10 {
        return 0.0;
    }

    let skewness = m3 / m2.powf(1.5);
    let mut beta = skewness * alpha * 0.5;
    if beta.abs() >= alpha {
        beta = 0.9 * alpha * beta.signum();
    }
    beta
}

fn neg_log_likelihood(times: &[f64], a: f64, b: f64, loc: f64, scale: f64) -> f64 {
    if a <= 0.0 || scale <= 0.0 || b.abs() >= a {
        return f64::INFINITY;
    }

    let delta = scale;
    let mu = loc;
    let gamma = (a * a - b * b).sqrt();

    let mut nll = 0.0;
    for &x in times {
        let z = (x - mu) / delta;
        let sqrt_z2p1 = (z * z + 1.0).sqrt();
        let y = a * sqrt_z2p1;
        let scaled_bessel = bessel_k1e(y);
        if scaled_bessel <= 0.0 {
            return f64::INFINITY;
        }

        let log_pdf = a.ln() - std::f64::consts::PI.ln() - delta.ln() - sqrt_z2p1.ln()
            + gamma
            + b * z
            - y
            + scaled_bessel.ln();
        nll -= log_pdf;
    }
    nll
}

fn optimize_nig(
    times: &[f64],
    mut a: f64,
    mut b: f64,
    mut loc: f64,
    mut scale: f64,
) -> Result<(f64, f64, f64, f64), OptimizeNigError> {
    const MAX_ITER: usize = 200;
    const TOL: f64 = 1e-8;
    const INIT_LR: f64 = 0.01;

    let mut lr = INIT_LR;
    let mut best_ll = neg_log_likelihood(times, a, b, loc, scale);

    for _iter in 0..MAX_ITER {
        // Compute gradients via central differences on each parameter
        let da = central_diff(|x| neg_log_likelihood(times, x, b, loc, scale), a);
        let db = central_diff(|x| neg_log_likelihood(times, a, x, loc, scale), b);
        let dloc = central_diff(|x| neg_log_likelihood(times, a, b, x, scale), loc);
        let dscale = central_diff(|x| neg_log_likelihood(times, a, b, loc, x), scale);

        let mut new_a = a - lr * da;
        let mut new_b = b - lr * db;
        let new_loc = loc - lr * dloc;
        let mut new_scale = scale - lr * dscale;

        // Enforce constraints
        if new_a < 0.01 {
            new_a = 0.01;
        }
        if new_b.abs() >= new_a {
            new_b = 0.99 * new_a * new_b.signum();
        }
        if new_scale < 1e-6 {
            new_scale = 1e-6;
        }

        let new_ll = neg_log_likelihood(times, new_a, new_b, new_loc, new_scale);
        if new_ll < best_ll {
            if best_ll - new_ll < TOL {
                a = new_a;
                b = new_b;
                loc = new_loc;
                scale = new_scale;
                return Ok((a, b, loc, scale));
            }
            a = new_a;
            b = new_b;
            loc = new_loc;
            scale = new_scale;
            best_ll = new_ll;
            // Gradually restore learning rate on success
            if lr < INIT_LR {
                lr *= 1.2;
                if lr > INIT_LR {
                    lr = INIT_LR;
                }
            }
        } else {
            lr *= 0.5;
            if lr < 1e-10 {
                return Err(OptimizeNigError::StepSizeExhausted);
            }
        }
    }

    Err(OptimizeNigError::MaxIterationsReached)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_abs_close(actual: f64, expected: f64, tolerance: f64) {
        let abs_error = (actual - expected).abs();
        assert!(
            abs_error <= tolerance,
            "expected {expected:.15e}, got {actual:.15e}, abs error {abs_error:.2e}",
        );
    }

    #[test]
    fn neg_log_likelihood_returns_inf_for_invalid_params() {
        let times = [7.0, 8.0, 9.0, 10.0];
        assert!(neg_log_likelihood(&times, 0.0, 1.0, 5.0, 1.0).is_infinite());
        assert!(neg_log_likelihood(&times, 2.0, 0.0, 5.0, 0.0).is_infinite());
        assert!(neg_log_likelihood(&times, 1.0, 1.0, 5.0, 1.0).is_infinite());
        assert!(neg_log_likelihood(&times, 1.0, -1.1, 5.0, 1.0).is_infinite());
    }

    #[test]
    fn neg_log_likelihood_returns_finite_for_valid_params() {
        let times = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0];
        let nll = neg_log_likelihood(&times, 5.0, 2.0, 8.0, 1.0);
        assert!(nll.is_finite());
        assert!(nll > 0.0);
    }

    #[test]
    fn neg_log_likelihood_matches_scipy_reference_value() {
        let times: Vec<f64> = (0..200)
            .map(|i| 7.0 + (i as f64).powf(1.5) * 0.005)
            .collect();
        let nll = neg_log_likelihood(
            &times,
            86.72396846356486,
            86.68319372270773,
            4.487105095426718,
            0.24914444085073106,
        );

        assert_abs_close(nll, 559.6276051205211, 1e-6);
    }

    #[test]
    fn estimate_location_is_median() {
        let mut times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        times.sort_by(f64::total_cmp);
        assert_abs_close(estimate_location(&times), 3.0, 1e-12);
    }

    #[test]
    fn estimate_scale_is_positive() {
        let mut times = vec![8.0, 8.5, 9.0, 9.5, 10.0];
        times.sort_by(f64::total_cmp);
        let loc = estimate_location(&times);
        let scale = estimate_scale(&times, loc);
        assert!(scale > 0.0);
    }

    #[test]
    fn estimate_alpha_is_at_least_one() {
        let mut times: Vec<f64> = (0..100).map(|i| 7.0 + i as f64 * 0.01).collect();
        times.sort_by(f64::total_cmp);
        let loc = estimate_location(&times);
        let scale = estimate_scale(&times, loc);
        let alpha = estimate_alpha(&times, loc, scale);
        assert!(alpha >= 1.0);
    }

    #[test]
    fn estimate_beta_constrained_by_alpha() {
        let mut times: Vec<f64> = (0..100).map(|i| 7.0 + i as f64 * 0.01).collect();
        times.sort_by(f64::total_cmp);
        let loc = estimate_location(&times);
        let scale = estimate_scale(&times, loc);
        let alpha = estimate_alpha(&times, loc, scale);
        let beta = estimate_beta(&times, loc, scale, alpha);
        assert!(beta.abs() < alpha);
    }

    #[test]
    fn fit_nig_converges_on_synthetic_data() {
        let times: Vec<f64> = (0..200)
            .map(|i| 7.0 + (i as f64).powf(1.5) * 0.005)
            .collect();
        let result = fit_nig(&times, None).expect("expected fit to converge"); 
        assert!(result.a > 0.01);
        assert!(result.b.abs() < result.a);
        assert!(result.scale > 1e-6);
        assert!(result.top_scale > 0.0);
        assert!(result.top_scale <= 1.01);
    }

    #[test]
    fn fit_nig_warm_start_works() {
        let times: Vec<f64> = (0..200)
            .map(|i| 7.0 + (i as f64).powf(1.5) * 0.005)
            .collect();
        let cold_result = fit_nig(&times, None).expect("expected cold fit to converge");
        let warm_result = fit_nig(&times, Some(&cold_result)).expect("expected warm fit to converge");
        assert_abs_close(warm_result.a, cold_result.a, 1.0);
    }

    #[test]
    fn fit_nig_improves_initial_estimates_on_synthetic_data() {
        let times: Vec<f64> = (0..200)
            .map(|i| 7.0 + (i as f64).powf(1.5) * 0.005)
            .collect();
        let loc = estimate_location(&times);
        let scale = estimate_scale(&times, loc);
        let a = estimate_alpha(&times, loc, scale);
        let b = estimate_beta(&times, loc, scale, a);
        let initial_nll = neg_log_likelihood(&times, a, b, loc, scale);

        let fitted = fit_nig(&times, None).expect("expected fit to converge");
        let fitted_nll = neg_log_likelihood(&times, fitted.a, fitted.b, fitted.loc, fitted.scale);

        assert!(fitted_nll.is_finite());
        assert!(
            fitted_nll <= initial_nll,
            "expected fitted params to improve NLL; initial={initial_nll:.6}, fitted={fitted_nll:.6}",
        );
    }
}
