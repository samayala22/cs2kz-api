use crate::maps::courses::Tier;
use crate::points::{self, NigParams};
use crate::records::RecordId;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Request {
    pub time: f64,
    pub nub_data: LeaderboardData,
    pub pro_data: Option<LeaderboardData>,
}

#[derive(Debug, serde::Deserialize)]
pub struct Response {
    pub nub_fraction: f64,
    pub pro_fraction: Option<f64>,
}

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
pub struct BestRecordData {
    pub record_id: RecordId,
    pub time: f64,
}

/// Output record with recalculated distribution points fraction.
#[derive(Debug, Clone)]
pub struct RecordPoints {
    pub record_id: RecordId,
    pub points: f64,
}

/// Result of a batch filter recalculation.
#[derive(Debug, Clone)]
pub struct RecalculateFilterResult {
    pub nub_records: Vec<RecordPoints>,
    pub pro_records: Vec<RecordPoints>,
    pub nub_params: NigParams,
    pub pro_params: NigParams,
    pub nub_fitted: bool,
    pub pro_fitted: bool,
}

pub fn calculate(request: &Request) -> Response {
    let nub_fraction = dist_points_portion(request.time, &request.nub_data);

    let pro_fraction = request.pro_data.as_ref().map(|data| {
        let pro_fraction = dist_points_portion(request.time, data);
        f64::max(pro_fraction, nub_fraction)
    });

    Response { nub_fraction, pro_fraction }
}

fn dist_points_portion(time: f64, data: &LeaderboardData) -> f64 {
    if data.leaderboard_size < points::SMALL_LEADERBOARD_THRESHOLD {
        return points::for_small_leaderboard(data.tier, data.top_time, time);
    }

    let Some(dist) = data.dist_params else {
        return points::for_small_leaderboard(data.tier, data.top_time, time);
    };
    let top_scale = if dist.top_scale > 0.0 { dist.top_scale } else { 1.0 };
    (nig::nig_survival(dist.a, dist.b, dist.loc, dist.scale, time) / top_scale).clamp(0.0, 1.0)
}

/// Recompute point fractions for all records in a filter
pub fn recalculate_filter(
    nub_records: &[BestRecordData],
    pro_records: &[BestRecordData],
    nub_tier: Tier,
    pro_tier: Tier,
    prev_nub_params: Option<&NigParams>,
    prev_pro_params: Option<&NigParams>,
) -> RecalculateFilterResult {
    let zero_params =
        NigParams { a: 0.0, b: 0.0, loc: 0.0, scale: 0.0, top_scale: 0.0 };

    let nub_times: Vec<f64> = nub_records.iter().map(|r| r.time).collect();
    let (nub_params, nub_fitted) = if nub_times.len() >= 50 {
        let prev = prev_nub_params.copied();
        let result = nig::fit_nig(&nub_times, prev.as_ref());
        if result.valid {
            (result.params, true)
        } else {
            (zero_params, false)
        }
    } else {
        (zero_params, false)
    };

    let nub_wr = nub_times.first().copied().unwrap_or(0.0);
    let nub_leaderboard = LeaderboardData {
        dist_params: nub_fitted.then_some(nub_params),
        tier: nub_tier,
        leaderboard_size: nub_records.len() as u64,
        top_time: nub_wr,
    };

    let new_nub_records: Vec<RecordPoints> = nub_records
        .iter()
        .map(|r| RecordPoints {
            record_id: r.record_id,
            points: dist_points_portion(r.time, &nub_leaderboard),
        })
        .collect();

    let pro_times: Vec<f64> = pro_records.iter().map(|r| r.time).collect();
    let (pro_params, pro_fitted) = if pro_times.len() >= 50 {
        let prev = prev_pro_params.copied();
        let result = nig::fit_nig(&pro_times, prev.as_ref());
        if result.valid {
            (result.params, true)
        } else {
            (zero_params, false)
        }
    } else {
        (zero_params, false)
    };

    let pro_wr = pro_times.first().copied().unwrap_or(0.0);
    let pro_leaderboard = LeaderboardData {
        dist_params: pro_fitted.then_some(pro_params),
        tier: pro_tier,
        leaderboard_size: pro_records.len() as u64,
        top_time: pro_wr,
    };

    let new_pro_records: Vec<RecordPoints> = pro_records
        .iter()
        .map(|r| {
            let pro_fraction = dist_points_portion(r.time, &pro_leaderboard);
            let nub_fraction = dist_points_portion(r.time, &nub_leaderboard);
            RecordPoints {
                record_id: r.record_id,
                points: f64::max(pro_fraction, nub_fraction),
            }
        })
        .collect();

    RecalculateFilterResult {
        nub_records: new_nub_records,
        pro_records: new_pro_records,
        nub_params,
        pro_params,
        nub_fitted,
        pro_fitted,
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
    fn calculate_points_matches_python_example() {
        let request = Request {
            time: 8.609375,
            nub_data: LeaderboardData {
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
            },
            pro_data: Some(LeaderboardData {
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
            }),
        };

        let response = calculate(&request);
        assert_abs_close(response.nub_fraction, 0.9745534941686896, 1e-3);
        assert_abs_close(response.pro_fraction.expect("pro fraction"), 0.9760910013054752, 1e-3);
    }

    #[test]
    fn pro_fraction_is_never_less_than_nub_fraction() {
        let request = Request {
            time: 8.0,
            nub_data: LeaderboardData {
                dist_params: None,
                tier: Tier::VeryEasy,
                leaderboard_size: 30,
                top_time: 7.0,
            },
            pro_data: Some(LeaderboardData {
                dist_params: None,
                tier: Tier::Death,
                leaderboard_size: 30,
                top_time: 7.0,
            }),
        };

        let response = calculate(&request);
        assert!(response.pro_fraction.is_some());
        assert!(response.pro_fraction.unwrap() >= response.nub_fraction);
    }
}
