use crate::bessel::bessel_k1_scaled;


pub(crate) fn nig_pdf(a: f64, b: f64, mu: f64, delta: f64, x: f64) -> f64 {
    if a <= 0.0 || delta <= 0.0 || b.abs() >= a {
        return 0.0;
    }

    let gamma = (a * a - b * b).sqrt();
    let z = (x - mu) / delta;
    let sqrt_z2p1 = (z * z + 1.0).sqrt();
    let y = a * sqrt_z2p1;
    let scaled_bessel = bessel_k1_scaled(y);

    if scaled_bessel <= 0.0 {
        return 0.0;
    }

    let net_exp = gamma + b * z - y;
    let log_pdf = a.ln()
        - std::f64::consts::PI.ln()
        - delta.ln()
        - sqrt_z2p1.ln()
        + net_exp
        + scaled_bessel.ln();

    if log_pdf < -745.0 {
        // exp(-745) underflows to 0 in f64
        return 0.0;
    }

    log_pdf.exp()
}

// Adapted from C code from https://en.wikipedia.org/wiki/Adaptive_Simpson%27s_method
pub(crate) fn adaptive_simpson(f: impl Fn(f64) -> f64 + Copy, a: f64, b: f64, eps: f64, max_depth: u32) -> Option<f64> {
    if a == b {
        return Some(0.0);
    }

    let h = b - a;
    let fa = f(a);
    let fb = f(b);
    let fm = f((a + b) / 2.0);
    let whole = (h / 6.0) * (fa + 4.0 * fm + fb);
    adaptive_simpson_rec(f, a, b, eps, whole, fa, fb, fm, max_depth)
}

fn adaptive_simpson_rec(
    f: impl Fn(f64) -> f64 + Copy,
    a: f64,
    b: f64,
    eps: f64,
    whole: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    depth: u32,
) -> Option<f64> {
    let m = (a + b) / 2.0;
    let lm = (a + m) / 2.0;
    let rm = (m + b) / 2.0;

    // Numerical trouble: epsilon underflow or interval collapsed
    if eps / 2.0 == eps || a == lm {
        return None;
    }

    let flm = f(lm);
    let frm = f(rm);
    let h = (b - a) / 2.0;
    let left = (h / 6.0) * (fa + 4.0 * flm + fm);
    let right = (h / 6.0) * (fm + 4.0 * frm + fb);
    let delta = left + right - whole;

    if depth == 0 || delta.abs() <= 15.0 * eps {
        return Some(left + right + delta / 15.0);
    }

    Some(
        adaptive_simpson_rec(f, a, m, eps / 2.0, left, fa, fm, flm, depth - 1)?
        + adaptive_simpson_rec(f, m, b, eps / 2.0, right, fm, fb, frm, depth - 1)?,
    )
}

pub fn nig_survival(a: f64, b: f64, loc: f64, scale: f64, x: f64) -> f64 {
    if a <= 0.0 || scale <= 0.0 || b.abs() >= a {
        return 0.0;
    }

    let gamma = (a * a - b * b).sqrt().max(1e-10);
    let mean = loc + scale * b / gamma;
    let stddev = (scale * a * a / (gamma * gamma * gamma)).sqrt();

    let mut upper = mean + 20.0 * stddev;
    if upper < x + scale {
        upper = x + 10.0 * scale;
    }

    adaptive_simpson(|t| nig_pdf(a, b, loc, scale, t), x, upper, 1e-12, 64)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_rel_close(actual: f64, expected: f64, tolerance: f64) {
        let rel_error = if expected == 0.0 {
            actual.abs()
        } else {
            (actual - expected).abs() / expected.abs()
        };
        assert!(
            rel_error <= tolerance,
            "expected {expected:.15e}, got {actual:.15e}, rel error {rel_error:.2e}",
        );
    }

    fn assert_abs_close(actual: f64, expected: f64, tolerance: f64) {
        let abs_error = (actual - expected).abs();
        assert!(
            abs_error <= tolerance,
            "expected {expected:.15e}, got {actual:.15e}, abs error {abs_error:.2e}",
        );
    }

    #[test]
    fn nig_pdf_matches_reference_values() {
        let (a, b, loc, scale) =
            (33.53900289787477, 33.52140111667502, 6.3663207368487065, 0.4480388195262859);

        for (x, expected) in [
            (7.648, 9.314339782198335e-03),
            (8.0, 2.138356268395934e-02),
            (10.0, 7.240000069597700e-02),
            (20.0, 3.070336727949191e-02),
        ] {
            assert_rel_close(nig_pdf(a, b, loc, scale, x), expected, 1e-10);
        }
    }

    #[test]
    fn nig_survival_matches_reference_values() {
        let (a, b, loc, scale) =
            (33.53900289787477, 33.52140111667502, 6.3663207368487065, 0.4480388195262859);

        for (x, expected) in [
            (7.0, 9.999892785756547e-01),
            (7.648, 9.979326056403205e-01),
            (10.0, 8.873317615160712e-01),
            (20.0, 3.429376452167427e-01),
        ] {
            assert_abs_close(nig_survival(a, b, loc, scale, x), expected, 1e-4);
        }
    }
}
