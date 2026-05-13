use std::sync::Arc;
use std::thread;
use std::time::Duration;

use futures_util::TryFutureExt as _;
use tokio::sync::Notify;
use tokio::time::interval;
use tokio_util::sync::CancellationToken;

use crate::maps::CourseFilterId;
use crate::maps::courses::Tier;
use crate::mode::Mode;
use crate::players::PlayerId;
use crate::points::{self, NigParams};
use crate::records::RecordId;
use crate::{Context, database, players};

const UPSERT_CHUNK_SIZE: usize = 5_000; // should prob put this somewhe

#[derive(Debug, Clone, Copy)]
struct BestRecordRow {
    filter_id: CourseFilterId,
    player_id: PlayerId,
    record_id: RecordId,
    time: f64,
}

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
                process_filter(&cx, filter_id).await?;
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

#[tracing::instrument(skip(cx))]
async fn process_filter(cx: &Context, filter_id: CourseFilterId) -> Result<(), database::Error> {
    tracing::debug!(%filter_id, "recalculating filter");

    let db = cx.database().as_ref();

    let nub_rows = sqlx::query_as!(
        BestRecordRow,
        "SELECT
           filter_id AS `filter_id: CourseFilterId`,
           player_id AS `player_id: PlayerId`,
           record_id AS `record_id: RecordId`,
           time
         FROM BestNubRecords
         WHERE filter_id = ?
         ORDER BY time ASC",
        filter_id,
    )
    .fetch_all(db)
    .await?;

    let nub_recs = nub_rows
        .iter()
        .map(|row| points::RecordTime { record_id: row.record_id, time: row.time })
        .collect::<Vec<_>>();

    // Pro records (sorted by time ASC)
    let pro_rows = sqlx::query_as!(
        BestRecordRow,
        "SELECT
           filter_id AS `filter_id: CourseFilterId`,
           player_id AS `player_id: PlayerId`,
           record_id AS `record_id: RecordId`,
           time
         FROM BestProRecords
         WHERE filter_id = ?
         ORDER BY time ASC",
        filter_id,
    )
    .fetch_all(db)
    .await?;

    let pro_recs = pro_rows
        .iter()
        .map(|row| points::RecordTime { record_id: row.record_id, time: row.time })
        .collect::<Vec<_>>();

    // Filter tiers
    let tiers_row = sqlx::query!(
        "SELECT
           nub_tier AS `nub_tier: Tier`,
           pro_tier AS `pro_tier: Tier`
         FROM CourseFilters
         WHERE id = ?",
        filter_id,
    )
    .fetch_optional(db)
    .await?;

    let Some(tiers_row) = tiers_row else {
        tracing::warn!(%filter_id, "filter not found in CourseFilters");
        return Ok(());
    };

    let nub_tier = tiers_row.nub_tier;
    let pro_tier = tiers_row.pro_tier;

    // Previous distribution parameters for warm start
    let prev_nub_params = sqlx::query_as!(
        NigParams,
        "SELECT a, b, loc, scale, top_scale
         FROM PointDistributionData
         WHERE filter_id = ? AND (NOT is_pro_leaderboard)",
        filter_id,
    )
    .fetch_optional(db)
    .await?;

    let prev_pro_params = sqlx::query_as!(
        NigParams,
        "SELECT a, b, loc, scale, top_scale
         FROM PointDistributionData
         WHERE filter_id = ? AND is_pro_leaderboard",
        filter_id,
    )
    .fetch_optional(db)
    .await?;

    // heavy calcs on dedicated thread
    let (tx, rx) = tokio::sync::oneshot::channel();

    thread::spawn({
        let nub_recs = nub_recs.clone();
        let pro_recs = pro_recs.clone();
        let prev_nub_params = prev_nub_params;
        let prev_pro_params = prev_pro_params;

        move || {
            let nub_result =
                points::recalculate_leaderboard(&nub_recs, nub_tier, prev_nub_params.as_ref());

            let mut pro_result =
                points::recalculate_leaderboard(&pro_recs, pro_tier, prev_pro_params.as_ref());

            for (record, recalculated_record) in pro_recs.iter().zip(pro_result.records.iter_mut())
            {
                let nub_fraction = points::calculate_fraction(record.time, &nub_result.leaderboard);
                recalculated_record.points = recalculated_record.points.max(nub_fraction);
            }

            let _ = tx.send((nub_result, pro_result));
        }
    });

    let (nub_result, pro_result) = rx.await.map_err(|_| {
        database::Error::decode(std::io::Error::other("points recalculation thread panicked"))
    })?;

    tracing::debug!(
        %filter_id,
        nub_fitted = nub_result.fitted,
        pro_fitted = pro_result.fitted,
        "recalculation complete, writing to DB"
    );

    cx.database_transaction(async move |conn| -> Result<_, database::Error> {
        upsert_best_records(
            conn,
            "INSERT INTO BestNubRecords (filter_id, player_id, record_id, points, time)",
            &nub_rows,
            &nub_result.records,
        )
        .await?;

        upsert_best_records(
            conn,
            "INSERT INTO BestProRecords (filter_id, player_id, record_id, points, time)",
            &pro_rows,
            &pro_result.records,
        )
        .await?;

        if nub_result.fitted {
            sqlx::query!(
                "INSERT INTO PointDistributionData (
                    filter_id, is_pro_leaderboard, a, b, loc, scale, top_scale
                 )
                 VALUES (?, FALSE, ?, ?, ?, ?, ?)
                 ON DUPLICATE KEY UPDATE
                    a = VALUES(a),
                    b = VALUES(b),
                    loc = VALUES(loc),
                    scale = VALUES(scale),
                    top_scale = VALUES(top_scale)",
                filter_id,
                nub_result.params.a,
                nub_result.params.b,
                nub_result.params.loc,
                nub_result.params.scale,
                nub_result.params.top_scale,
            )
            .execute(&mut *conn)
            .await?;
        }

        if pro_result.fitted {
            sqlx::query!(
                "INSERT INTO PointDistributionData (
                    filter_id, is_pro_leaderboard, a, b, loc, scale, top_scale
                 )
                 VALUES (?, TRUE, ?, ?, ?, ?, ?)
                 ON DUPLICATE KEY UPDATE
                    a = VALUES(a),
                    b = VALUES(b),
                    loc = VALUES(loc),
                    scale = VALUES(scale),
                    top_scale = VALUES(top_scale)",
                filter_id,
                pro_result.params.a,
                pro_result.params.b,
                pro_result.params.loc,
                pro_result.params.scale,
                pro_result.params.top_scale,
            )
            .execute(&mut *conn)
            .await?;
        }

        Ok(())
    })
    .await?;

    Ok(())
}

async fn upsert_best_records(
    conn: &mut database::Connection,
    insert_prefix: &'static str,
    rows: &[BestRecordRow],
    recalculated_records: &[points::RecordPoints],
) -> Result<(), database::Error> {
    if rows.len() != recalculated_records.len() {
        return Err(database::Error::decode(std::io::Error::other(
            "recalculated record count does not match fetched best record rows",
        )));
    }

    for (row_chunk, recalculated_chunk) in rows
        .chunks(UPSERT_CHUNK_SIZE)
        .zip(recalculated_records.chunks(UPSERT_CHUNK_SIZE))
    {
        for (row, recalculated_record) in row_chunk.iter().zip(recalculated_chunk.iter()) {
            if row.record_id != recalculated_record.record_id {
                return Err(database::Error::decode(std::io::Error::other(
                    "recalculated record order no longer matches fetched best record rows",
                )));
            }
        }

        let mut query = database::QueryBuilder::new(insert_prefix);

        query.push_values(
            row_chunk.iter().zip(recalculated_chunk.iter()),
            |mut query, (row, recalculated_record)| {
                query.push_bind(row.filter_id);
                query.push_bind(row.player_id);
                query.push_bind(row.record_id);
                query.push_bind(recalculated_record.points);
                query.push_bind(row.time);
            },
        );

        query.push(" ON DUPLICATE KEY UPDATE points = VALUES(points)");
        query.build().execute(&mut *conn).await?;
    }

    Ok(())
}
