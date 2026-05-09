use std::sync::Arc;
use std::time::Duration;

use futures_util::TryFutureExt as _;
use sqlx::Row as _;
use tokio::sync::Notify;
use tokio::time::interval;
use tokio_util::sync::CancellationToken;

use crate::maps::CourseFilterId;
use crate::mode::Mode;
use crate::points::calculator;
use crate::points::NigParams;
use crate::{Context, database, players};

#[derive(Debug, Clone)]
pub struct PointsDaemonHandle {
    notifications: Arc<Notifications>,
}

impl PointsDaemonHandle {
    #[expect(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            notifications: Arc::new(Notifications { record_submitted: Notify::new() }),
        }
    }

    pub fn notify_record_submitted(&self) {
        self.notifications.record_submitted.notify_waiters();
    }
}

#[derive(Debug)]
struct Notifications {
    record_submitted: Notify,
}

#[derive(Debug, Display, Error, From)]
pub enum Error {
    DetermineFilterToRecalculate(DetermineFilterToRecalculateError),
    ProcessFilter(database::Error),
}

#[derive(Debug, Display, Error, From)]
#[display("failed to determine next filter to recalculate: {_0}")]
#[from(forward)]
pub struct DetermineFilterToRecalculateError(database::Error);

#[tracing::instrument(skip_all, err)]
pub async fn run(cx: Context, cancellation_token: CancellationToken) -> Result<(), Error> {
    let mut recalc_ratings_interval = interval(Duration::from_secs(10));

    loop {
        select! {
            () = cancellation_token.cancelled() => {
                tracing::debug!("cancelled");
                break Ok(());
            },

            _ = recalc_ratings_interval.tick() => {
                tracing::debug!("recalculating ratings");
                recalculate_ratings(&cx).await;
            },

            res = determine_filter_to_recalculate(&cx) => {
                let (filter_id, priority) = res?;
                process_filter(&cx, &cancellation_token, filter_id).await?;
                update_filters_to_recalculate(&cx, filter_id, priority).await;
            },
        };
    }
}

#[tracing::instrument(skip(cx))]
async fn recalculate_ratings(cx: &Context) {
    use players::update_ratings;

    for mode in [Mode::Vanilla, Mode::Classic] {
        if let Err(err) = update_ratings(cx, mode).await {
            tracing::error!(%err, ?mode, "failed to recalculate ratings");
        }
    }
}

#[tracing::instrument(skip(cx))]
async fn determine_filter_to_recalculate(
    cx: &Context,
) -> Result<(CourseFilterId, u64), DetermineFilterToRecalculateError> {
    loop {
        if let Some(data) = sqlx::query!(
            "SELECT
               filter_id AS `filter_id: CourseFilterId`,
               priority
             FROM FiltersToRecalculate
             WHERE priority > 0
             ORDER BY priority DESC
             LIMIT 1",
        )
        .fetch_optional(cx.database().as_ref())
        .map_ok(|maybe_row| maybe_row.map(|row| (row.filter_id, row.priority)))
        .await?
        {
            break Ok(data);
        }

        () = cx
            .points_daemon()
            .notifications
            .record_submitted
            .notified()
            .await;

        tracing::trace!("received notification about submitted record");
    }
}

#[tracing::instrument(skip(cx))]
async fn update_filters_to_recalculate(
    cx: &Context,
    filter_id: CourseFilterId,
    prev_priority: u64,
) {
    if let Err(err) = sqlx::query!(
        "UPDATE FiltersToRecalculate
         SET priority = (priority - ?)
         WHERE filter_id = ?",
        prev_priority,
        filter_id,
    )
    .execute(cx.database().as_ref())
    .await
    {
        tracing::warn!(%err, %filter_id, prev_priority, "failed to update FiltersToRecalculate");
    }
}

/// Holds DB-fetched record data for native recalculation.
#[derive(Debug)]
struct DbRecord {
    record_id: [u8; 16],
    time: f64,
}

#[tracing::instrument(skip(cx, cancellation_token))]
async fn process_filter(
    cx: &Context,
    cancellation_token: &CancellationToken,
    filter_id: CourseFilterId,
) -> Result<(), database::Error> {
    tracing::debug!(%filter_id, "recalculating filter natively");

    let db = cx.database().as_ref();

    // Nub records (sorted by time ASC)
    let nub_rows = sqlx::query(
        "SELECT record_id, time FROM BestNubRecords WHERE filter_id = ? ORDER BY time ASC"
    )
    .bind(filter_id)
    .fetch_all(db)
    .await?;

    // Pro records (sorted by time ASC)
    let pro_rows = sqlx::query(
        "SELECT record_id, time FROM BestProRecords WHERE filter_id = ? ORDER BY time ASC"
    )
    .bind(filter_id)
    .fetch_all(db)
    .await?;

    // Filter tiers
    let tiers_row = sqlx::query(
        "SELECT nub_tier, pro_tier FROM CourseFilters WHERE id = ?"
    )
    .bind(filter_id)
    .fetch_optional(db)
    .await?;

    let Some(tiers_row) = tiers_row else {
        tracing::warn!(%filter_id, "filter not found in CourseFilters");
        return Ok(());
    };

    let nub_tier: crate::maps::courses::Tier = tiers_row.try_get(0)?;
    let pro_tier: crate::maps::courses::Tier = tiers_row.try_get(1)?;

    // Previous distribution parameters for warm start
    let prev_nub_row = sqlx::query(
        "SELECT a, b, loc, scale, top_scale
         FROM PointDistributionData
         WHERE filter_id = ? AND (NOT is_pro_leaderboard)"
    )
    .bind(filter_id)
    .fetch_optional(db)
    .await?;

    let prev_nub_params = prev_nub_row.map(|row| NigParams {
        a: row.get(0),
        b: row.get(1),
        loc: row.get(2),
        scale: row.get(3),
        top_scale: row.get(4),
    });

    let prev_pro_row = sqlx::query(
        "SELECT a, b, loc, scale, top_scale
         FROM PointDistributionData
         WHERE filter_id = ? AND is_pro_leaderboard"
    )
    .bind(filter_id)
    .fetch_optional(db)
    .await?;

    let prev_pro_params = prev_pro_row.map(|row| NigParams {
        a: row.get(0),
        b: row.get(1),
        loc: row.get(2),
        scale: row.get(3),
        top_scale: row.get(4),
    });

    let nub_records: Vec<DbRecord> = nub_rows.iter().map(|row| {
        let bytes: &[u8] = row.get(0);
        let mut id = [0u8; 16];
        id.copy_from_slice(bytes);
        DbRecord { record_id: id, time: row.get(1) }
    }).collect();
    let pro_records: Vec<DbRecord> = pro_rows.iter().map(|row| {
        let bytes: &[u8] = row.get(0);
        let mut id = [0u8; 16];
        id.copy_from_slice(bytes);
        DbRecord { record_id: id, time: row.get(1) }
    }).collect();

    let nub_recs: Vec<calculator::BestRecordData> = nub_records.iter().map(|r|
        calculator::BestRecordData { record_id: r.record_id, time: r.time }
    ).collect();
    let pro_recs: Vec<calculator::BestRecordData> = pro_records.iter().map(|r|
        calculator::BestRecordData { record_id: r.record_id, time: r.time }
    ).collect();

    // heavy calcs on blocking thread
    let result = tokio::task::spawn_blocking(move || {
        calculator::recalculate_filter(
            &nub_recs,
            &pro_recs,
            nub_tier,
            pro_tier,
            prev_nub_params.as_ref(),
            prev_pro_params.as_ref(),
        )
    });

    let result = cancellation_token
        .run_until_cancelled(result)
        .await
        .ok_or_else(|| {
            database::Error::decode(std::io::Error::other("points recalculation cancelled"))
        })
        .and_then(|res| res.map_err(database::Error::decode))?;

    tracing::debug!(
        %filter_id,
        nub_fitted = result.nub_fitted,
        pro_fitted = result.pro_fitted,
        "recalculation complete, writing to DB"
    );

    for record in &result.nub_records {
        sqlx::query("UPDATE BestNubRecords SET points = ? WHERE record_id = ?")
            .bind(record.points)
            .bind(record.record_id.as_slice())
            .execute(db)
            .await?;
    }

    for record in &result.pro_records {
        sqlx::query("UPDATE BestProRecords SET points = ? WHERE record_id = ?")
            .bind(record.points)
            .bind(record.record_id.as_slice())
            .execute(db)
            .await?;
    }

    if result.nub_fitted {
        sqlx::query(
            "INSERT INTO PointDistributionData (
                filter_id, is_pro_leaderboard, a, b, loc, scale, top_scale
             )
             VALUES (?, FALSE, ?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                a = VALUES(a),
                b = VALUES(b),
                loc = VALUES(loc),
                scale = VALUES(scale),
                top_scale = VALUES(top_scale)"
        )
        .bind(filter_id)
        .bind(result.nub_params.a)
        .bind(result.nub_params.b)
        .bind(result.nub_params.loc)
        .bind(result.nub_params.scale)
        .bind(result.nub_params.top_scale)
        .execute(db)
        .await?;
    }

    if result.pro_fitted {
        sqlx::query(
            "INSERT INTO PointDistributionData (
                filter_id, is_pro_leaderboard, a, b, loc, scale, top_scale
             )
             VALUES (?, TRUE, ?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                a = VALUES(a),
                b = VALUES(b),
                loc = VALUES(loc),
                scale = VALUES(scale),
                top_scale = VALUES(top_scale)"
        )
        .bind(filter_id)
        .bind(result.pro_params.a)
        .bind(result.pro_params.b)
        .bind(result.pro_params.loc)
        .bind(result.pro_params.scale)
        .bind(result.pro_params.top_scale)
        .execute(db)
        .await?;
    }

    Ok(())
}
