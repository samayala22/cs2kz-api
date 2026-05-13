use crate::maps::courses::Tier;
use crate::points::{self, NigParams};
use crate::records::RecordId;

#[derive(Debug, Clone, serde::Serialize)]
pub struct LeaderboardData {
    pub dist_params: Option<NigParams>,
    #[serde(serialize_with = "Tier::serialize_as_integer")]
    pub tier: Tier,
    pub leaderboard_size: u64,
    #[serde(rename = "wr")]
    pub top_time: f64,
}

/// Input record data for batch recalculation.
#[derive(Debug, Clone)]
pub struct RecordTime {
    pub record_id: RecordId,
    pub time: f64,
}

/// Output record with recalculated distribution points fraction.
#[derive(Debug, Clone)]
pub struct RecordPoints {
    pub record_id: RecordId,
    pub points: f64,
}

/// Result of a leaderboard recalculation.
#[derive(Debug, Clone)]
pub struct RecalculatedLeaderboard {
    pub leaderboard: LeaderboardData,
    pub records: Vec<RecordPoints>,
    pub params: NigParams,
    pub fitted: bool,
}

pub fn calculate_fraction(time: f64, leaderboard: &LeaderboardData) -> f64 {
    if leaderboard.leaderboard_size < points::SMALL_LEADERBOARD_THRESHOLD {
        return points::for_small_leaderboard(leaderboard.tier, leaderboard.top_time, time);
    }

    let Some(dist) = leaderboard.dist_params else {
        return points::for_small_leaderboard(leaderboard.tier, leaderboard.top_time, time);
    };

    let top_scale = if dist.top_scale > 0.0 { dist.top_scale } else { 1.0 };
    (nig::nig_survival(dist.a, dist.b, dist.loc, dist.scale, time) / top_scale).clamp(0.0, 1.0)
}

/// Recompute point fractions for a single leaderboard.
pub fn recalculate_leaderboard(
    records: &[RecordTime],
    tier: Tier,
    prev_params: Option<&NigParams>,
) -> RecalculatedLeaderboard {
    let times: Vec<f64> = records.iter().map(|record| record.time).collect();
    let (params, fitted) = fit_distribution(&times, prev_params);

    let leaderboard = LeaderboardData {
        dist_params: fitted.then_some(params),
        tier,
        leaderboard_size: records.len() as u64,
        top_time: times.first().copied().unwrap_or(0.0),
    };

    let recalculated_records = records
        .iter()
        .map(|record| RecordPoints {
            record_id: record.record_id,
            points: calculate_fraction(record.time, &leaderboard),
        })
        .collect();

    RecalculatedLeaderboard {
        leaderboard,
        records: recalculated_records,
        params,
        fitted,
    }
}

fn fit_distribution(times: &[f64], prev_params: Option<&NigParams>) -> (NigParams, bool) {
    let zero_params = NigParams {
        a: 0.0,
        b: 0.0,
        loc: 0.0,
        scale: 0.0,
        top_scale: 0.0,
    };

    if times.len() < 50 {
        return (zero_params, false);
    }

    let result = nig::fit_nig(times, prev_params);
    if result.valid {
        (result.params, true)
    } else {
        (zero_params, false)
    }
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
    fn calculate_fraction_matches_python_example() {
        let time = 8.609375;
        let nub_data = LeaderboardData {
            dist_params: Some(NigParams {
                a: 33.53900289787477,
                b: 33.52140111667502,
                loc: 6.3663207368487065,
                scale: 0.4480388195262859,
                top_scale: 0.9979285278452101,
            }),
            tier: Tier::VeryEasy,
            leaderboard_size: 224,
            top_time: 7.6484375,
        };
        let pro_data = LeaderboardData {
            dist_params: Some(NigParams {
                a: 2.6294814553333743,
                b: 2.511121972118702,
                loc: 8.713014153227697,
                scale: 2.2226724397990805,
                top_scale: 0.9952929135343108,
            }),
            tier: Tier::VeryEasy,
            leaderboard_size: 165,
            top_time: 7.6484375,
        };

        assert_abs_close(calculate_fraction(time, &nub_data), 0.9745534941686896, 1e-3);
        assert_abs_close(calculate_fraction(time, &pro_data), 0.9760910013054752, 1e-3);
    }

    #[test]
    fn calculate_fraction_falls_back_to_small_leaderboard_formula() {
        let leaderboard = LeaderboardData {
            dist_params: None,
            tier: Tier::VeryEasy,
            leaderboard_size: 30,
            top_time: 7.0,
        };

        assert_abs_close(
            calculate_fraction(8.0, &leaderboard),
            points::for_small_leaderboard(leaderboard.tier, leaderboard.top_time, 8.0),
            1e-12,
        );
    }
}
