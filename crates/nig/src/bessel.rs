/// Evaluates the rational function P(z)/Q(z) using Horner's method.
/// <https://github.com/boostorg/math/blob/develop/include/boost/math/tools/rational.hpp>
pub(crate) fn evaluate_rational(p: &[f64], q: &[f64], z: f64) -> f64 {
    if p.is_empty() || q.is_empty() {
        return 0.0;
    }

    let mut pn = p[p.len() - 1];
    for i in (0..p.len() - 1).rev() {
        pn = pn * z + p[i];
    }

    let mut qn = q[q.len() - 1];
    for i in (0..q.len() - 1).rev() {
        qn = qn * z + q[i];
    }

    pn / qn
}

/// Bessel K1 functions adapted from Boost
/// https://github.com/boostorg/math/blob/develop/include/boost/math/special_functions/detail/bessel_k1.hpp
pub(crate) fn bessel_k1(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    if x <= 1.0 {
        bessel_k1_small(x)
    } else {
        bessel_k1_large(x)
    }
}

fn bessel_k1_small(x: f64) -> f64 {
    const Y: f64 = 8.69547128677368164e-02;

    const P1: [f64; 4] = [
        -3.62137953440350228e-03,
        7.11842087490330300e-03,
        1.00302560256614306e-05,
        1.77231085381040811e-06,
    ];
    const Q1: [f64; 4] = [
        1.00000000000000000e+00,
        -4.80414794429043831e-02,
        9.85972641934416525e-04,
        -8.91196859397070326e-06,
    ];

    const P2: [f64; 4] = [
        -3.07965757829206184e-01,
        -7.80929703673074907e-02,
        -2.70619343754051620e-03,
        -2.49549522229072008e-05,
    ];
    const Q2: [f64; 4] = [
        1.00000000000000000e+00,
        -2.36316836412163098e-02,
        2.64524577525962719e-04,
        -1.49749618004162787e-06,
    ];

    let a = x * x / 4.0;
    let log_term = ((evaluate_rational(&P1, &Q1, a) + Y) * a * a + a / 2.0 + 1.0) * x / 2.0;

    evaluate_rational(&P2, &Q2, x * x) * x + 1.0 / x + x.ln() * log_term
}

fn bessel_k1_large(x: f64) -> f64 {
    const Y: f64 = 1.45034217834472656e+00;

    const P: [f64; 9] = [
        -1.97028041029226295e-01,
        -2.32408961548087617e+00,
        -7.98269784507699938e+00,
        -2.39968410774221632e+00,
        3.28314043780858713e+01,
        5.67713761158496058e+01,
        3.30907788466509823e+01,
        6.62582288933739787e+00,
        3.08851840645286691e-01,
    ];
    const Q: [f64; 9] = [
        1.00000000000000000e+00,
        1.41811409298826118e+01,
        7.35979466317556420e+01,
        1.77821793937080859e+02,
        2.11014501598705982e+02,
        1.19425262951064454e+02,
        2.88448064302447607e+01,
        2.27912927104139732e+00,
        2.50358186953478678e-02,
    ];

    let rat = evaluate_rational(&P, &Q, 1.0 / x) + Y;
    if x < 709.0 {
        return rat * (-x).exp() / x.sqrt();
    }

    let exp_half = (-x / 2.0).exp();
    rat * exp_half / x.sqrt() * exp_half
}

pub(crate) fn bessel_k1_scaled(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    if x <= 1.0 {
        return bessel_k1(x) * x.exp();
    }

    const Y: f64 = 1.45034217834472656e+00;

    const P: [f64; 9] = [
        -1.97028041029226295e-01,
        -2.32408961548087617e+00,
        -7.98269784507699938e+00,
        -2.39968410774221632e+00,
        3.28314043780858713e+01,
        5.67713761158496058e+01,
        3.30907788466509823e+01,
        6.62582288933739787e+00,
        3.08851840645286691e-01,
    ];
    const Q: [f64; 9] = [
        1.00000000000000000e+00,
        1.41811409298826118e+01,
        7.35979466317556420e+01,
        1.77821793937080859e+02,
        2.11014501598705982e+02,
        1.19425262951064454e+02,
        2.88448064302447607e+01,
        2.27912927104139732e+00,
        2.50358186953478678e-02,
    ];

    (evaluate_rational(&P, &Q, 1.0 / x) + Y) / x.sqrt()
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

    #[test]
    fn bessel_k1_matches_reference_values() {
        for (x, expected) in [
            (0.001, 9.999962381560855e+02),
            (0.1, 9.853844780870606e+00),
            (1.0, 6.019072301972346e-01),
            (2.0, 1.398658818165225e-01),
            (5.0, 4.044613445452163e-03),
            (20.0, 5.883057969557038e-10),
            (100.0, 4.679853735636910e-45),
            (709.0, 5.730317612554602e-310),
        ] {
            assert_rel_close(bessel_k1(x), expected, 1e-14);
        }
    }
}
