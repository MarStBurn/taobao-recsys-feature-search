#!/usr/bin/env python3
"""Compare fixed RecSys aggregates with a deterministic search over time windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

DAY = 86_400
RELATIONS = {
    "user": ("user_id",),
    "item": ("item_id",),
    "pair": ("user_id", "item_id"),
    "user_category": ("user_id", "category_id"),
    "category": ("category_id",),
}
AUTO_WINDOWS = (3_600, 21_600, DAY, 3 * DAY, 7 * DAY, None)
MANUAL_WINDOWS = (DAY, 7 * DAY, None)
WINDOW_NAMES = {
    3_600: "1h",
    21_600: "6h",
    DAY: "1d",
    3 * DAY: "3d",
    7 * DAY: "7d",
    None: "all",
}


def quote_identifier(identifier: str) -> str:
    if not identifier.replace("_", "").isalnum():
        raise ValueError(f"Unexpected SQL identifier: {identifier}")
    return f'"{identifier}"'


def windows_sql(windows: tuple[int | None, ...]) -> str:
    expressions = []
    for seconds in windows:
        suffix = WINDOW_NAMES[seconds]
        if seconds is None:
            frame = "ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING"
        else:
            frame = f"RANGE BETWEEN {seconds} PRECEDING AND 1 PRECEDING"
        event_count = f"sum(event_count) OVER (PARTITION BY {{keys}} ORDER BY timestamp {frame})"
        buy_count = f"sum(buy_count) OVER (PARTITION BY {{keys}} ORDER BY timestamp {frame})"
        expressions.extend(
            [
                f"{event_count} AS count_{suffix}",
                f"{buy_count} / nullif({event_count}, 0) AS buy_rate_{suffix}",
            ]
        )
    return ",\n                ".join(expressions)


def create_event_tables(connection: duckdb.DuckDBPyConnection, csv_path: Path) -> tuple[int, int]:
    """Read the standard headerless 5-column UserBehavior CSV without loading it into Python."""
    source = str(csv_path.resolve()).replace("'", "''")
    connection.execute(
        f"""
        CREATE TEMP TABLE events AS
        SELECT
            row_number() OVER ()::BIGINT AS row_id,
            user_id::BIGINT AS user_id,
            item_id::BIGINT AS item_id,
            category_id::BIGINT AS category_id,
            lower(trim(behavior_type)) AS behavior_type,
            timestamp::BIGINT AS timestamp,
            (lower(trim(behavior_type)) = 'buy')::UTINYINT AS label
        FROM read_csv(
            '{source}',
            delim=',',
            header=false,
            columns={{
                'user_id': 'BIGINT',
                'item_id': 'BIGINT',
                'category_id': 'BIGINT',
                'behavior_type': 'VARCHAR',
                'timestamp': 'BIGINT'
            }}
        )
        """
    )
    total = connection.execute("SELECT count(*) FROM events").fetchone()[0]
    if not total:
        raise ValueError(f"No interactions found in {csv_path}")
    invalid = connection.execute(
        "SELECT count(*) FROM events WHERE behavior_type NOT IN ('pv', 'view', 'buy', 'cart', 'fav')"
    ).fetchone()[0]
    if invalid:
        examples = connection.execute(
            "SELECT DISTINCT behavior_type FROM events "
            "WHERE behavior_type NOT IN ('pv', 'view', 'buy', 'cart', 'fav') LIMIT 5"
        ).fetchall()
        raise ValueError(f"Unexpected behavior_type values: {examples}")

    train_end, valid_end = connection.execute(
        "SELECT approx_quantile(timestamp, 0.8), approx_quantile(timestamp, 0.9) FROM events"
    ).fetchone()
    connection.execute(
        """
        CREATE TEMP TABLE labeled_events AS
        SELECT *,
            CASE
                WHEN timestamp < ? THEN 'train'
                WHEN timestamp < ? THEN 'validation'
                ELSE 'test'
            END AS split
        FROM events
        """,
        [train_end, valid_end],
    )
    split_counts = connection.execute(
        "SELECT split, label, count(*) FROM labeled_events GROUP BY split, label"
    ).fetchall()
    if len({row[0] for row in split_counts}) < 3:
        raise ValueError("Temporal split produced an empty partition.")
    return int(total), int(invalid)


def create_sampled_candidates(
    connection: duckdb.DuckDBPyConnection, max_per_class: int, seed: int
) -> dict[str, int]:
    connection.execute(
        f"""
        CREATE TEMP TABLE candidates AS
        WITH ranked AS (
            SELECT *,
                row_number() OVER (
                    PARTITION BY split, label
                    ORDER BY hash(row_id + {int(seed)})
                ) AS sample_rank,
                count(*) OVER (PARTITION BY split, label) AS population_count
            FROM labeled_events
        )
        SELECT row_id, user_id, item_id, category_id, timestamp, label, split,
            population_count / least(population_count, {int(max_per_class)})::DOUBLE
                AS sample_weight,
            hour(to_timestamp(timestamp)) AS hour_of_day,
            dayofweek(to_timestamp(timestamp)) AS day_of_week
        FROM ranked
        WHERE sample_rank <= {int(max_per_class)}
        """
    )
    counts = connection.execute(
        "SELECT split, count(*) FROM candidates GROUP BY split"
    ).fetchall()
    return {str(split): int(count) for split, count in counts}


def create_relation_features(
    connection: duckdb.DuckDBPyConnection,
    relation: str,
    key_columns: tuple[str, ...],
    windows: tuple[int | None, ...],
) -> None:
    keys = ", ".join(quote_identifier(key) for key in key_columns)
    key_partition = ", ".join(quote_identifier(key) for key in key_columns)
    window_exprs = windows_sql(windows).replace("{keys}", key_partition)
    connection.execute(
        f"""
        CREATE TEMP TABLE feat_{relation} AS
        WITH aggregates AS (
            SELECT {keys}, timestamp, count(*)::BIGINT AS event_count,
                sum(label)::BIGINT AS buy_count
            FROM labeled_events
            GROUP BY {keys}, timestamp
        )
        SELECT {keys}, timestamp,
            {window_exprs}
        FROM aggregates
        """
    )


def fetch_feature_frame(
    connection: duckdb.DuckDBPyConnection,
    windows: tuple[int | None, ...],
) -> pd.DataFrame:
    joins = []
    selected = [
        "c.label", "c.split", "c.sample_weight", "c.user_id", "c.item_id",
        "c.category_id", "c.hour_of_day", "c.day_of_week",
    ]
    for relation, keys in RELATIONS.items():
        alias = f"f_{relation}"
        conditions = [f"{alias}.{quote_identifier(key)} = c.{quote_identifier(key)}" for key in keys]
        conditions.append(f"{alias}.timestamp = c.timestamp")
        joins.append(
            f"LEFT JOIN feat_{relation} AS {alias} ON " + " AND ".join(conditions)
        )
        for window in windows:
            suffix = WINDOW_NAMES[window]
            selected.extend(
                [
                    f"{alias}.count_{suffix} AS {relation}_count_{suffix}",
                    f"{alias}.buy_rate_{suffix} AS {relation}_buy_rate_{suffix}",
                ]
            )
    query = (
        "SELECT " + ", ".join(selected) + " FROM candidates AS c "
        + " ".join(joins)
    )
    frame = connection.execute(query).fetchdf()
    for column in ("user_id", "item_id", "category_id", "hour_of_day", "day_of_week"):
        frame[column] = frame[column].astype("category")
    return frame


def ranking_metrics(frame: pd.DataFrame, probabilities: np.ndarray, k: int = 10) -> dict[str, float]:
    labels = frame["label"].to_numpy()
    users = frame["user_id"].astype("int64").to_numpy()
    ndcgs: list[float] = []
    recalls: list[float] = []
    for user_id in np.unique(users):
        mask = users == user_id
        truth = labels[mask]
        order = np.argsort(-probabilities[mask], kind="stable")[:k]
        gains = truth[order]
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        dcg = float(np.sum(gains * discounts))
        ideal = np.sort(truth)[::-1][:k]
        idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        positives = int(truth.sum())
        if positives:
            recalls.append(float(gains.sum()) / min(positives, k))
    return {
        "ndcg_at_10": float(np.mean(ndcgs)),
        "recall_at_10": float(np.mean(recalls)) if recalls else 0.0,
    }


def metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    labels = frame["label"].to_numpy()
    weights = frame["sample_weight"].to_numpy()
    return {
        "weighted_log_loss": float(log_loss(labels, probabilities, sample_weight=weights)),
        "weighted_auc": float(roc_auc_score(labels, probabilities, sample_weight=weights)),
        **ranking_metrics(frame, probabilities),
    }


def evaluate_variant(
    name: str, frame: pd.DataFrame, window_set: tuple[int | None, ...]
) -> dict[str, Any]:
    feature_columns = [
        column for column in frame.columns
        if column not in {"label", "split", "sample_weight"}
        and (
            not any(column.endswith(f"_{WINDOW_NAMES[w]}") for w in AUTO_WINDOWS)
            or any(column.endswith(f"_{WINDOW_NAMES[w]}") for w in window_set)
        )
    ]
    categorical = ["user_id", "item_id", "category_id", "hour_of_day", "day_of_week"]
    train = frame["split"].eq("train").to_numpy()
    validation = frame["split"].eq("validation").to_numpy()
    test = frame["split"].eq("test").to_numpy()
    if any(not mask.any() for mask in (train, validation, test)):
        raise ValueError("A sampled train/validation/test partition is empty.")
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=50,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        frame.loc[train, feature_columns],
        frame.loc[train, "label"],
        sample_weight=frame.loc[train, "sample_weight"],
        categorical_feature=categorical,
        eval_set=[(frame.loc[validation, feature_columns], frame.loc[validation, "label"])],
        eval_sample_weight=[frame.loc[validation, "sample_weight"]],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    output: dict[str, Any] = {
        "feature_count": len(feature_columns),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
    }
    for split_name, mask in (("validation", validation), ("test", test)):
        subset = frame.loc[mask]
        probability = model.predict_proba(subset[feature_columns])[:, 1]
        output[split_name] = metrics(subset, probability)
    output["top_features"] = [
        {"feature": str(column), "gain": float(gain)}
        for column, gain in sorted(
            zip(feature_columns, model.booster_.feature_importance(importance_type="gain")),
            key=lambda pair: pair[1],
            reverse=True,
        )[:15]
    ]
    return output


def write_report(
    results: dict[str, Any], output_path: Path, max_per_class: int
) -> None:
    manual, automatic = results["manual"], results["automatic"]
    lines = [
        "| Split | Metric | Manual | Auto search |",
        "|---|---|---:|---:|",
    ]
    for split in ("validation", "test"):
        for key in ("weighted_log_loss", "weighted_auc", "ndcg_at_10", "recall_at_10"):
            lines.append(
                f"| {split} | {key} | {manual[split][key]:.5f} | {automatic[split][key]:.5f} |"
            )
    report = f"""# Taobao UserBehavior: automated RecSys feature search

## Experiment

The candidate set is a compact feature grammar: five relations (user, item,
user-item, user-category, category) × time window × aggregate (event count,
purchase rate). The automatic model receives windows of 1 hour, 6 hours, 1 day,
3 days, 7 days, and all history. The manual baseline receives 1 day, 7 days,
and all history. LightGBM performs the feature selection implicitly through
regularized splits; both models use the same parameters and validation-based
early stopping.

## Data and target

- Input: Alibaba Taobao UserBehavior, standard headerless CSV columns
  `user_id,item_id,category_id,behavior_type,timestamp`.
- Target: whether the current logged action is `buy`; `pv`/`view`, `cart`, and
  `fav` are negative examples. This predicts purchase actions among recorded
  behaviors, not conversion among known ad impressions.
- Temporal split: 80%/10%/10% by timestamp quantiles. Equal timestamps are
  assigned to one partition.
- Up to {max_per_class:,} examples per class and split are sampled uniformly;
  inverse inclusion weights correct class-wise sampling for weighted log loss
  and AUC.

## Point-in-time safety and scale

For each relation, the complete event log is first aggregated by entity key and
timestamp in DuckDB. SQL `RANGE` windows end one second before the current
timestamp; equal-time events are aggregated together and never expose each
other. `all` uses the preceding timestamp groups only. Events in validation/test
are still added to history after their own timestamp, modeling an online
prediction stream. DuckDB performs aggregation and joins without reading the
100M raw rows into pandas; only the capped candidate sample and its features
are materialized for LightGBM.

## Results

{chr(10).join(lines)}

Sampled candidate counts: {json.dumps(results["sample_counts"], ensure_ascii=False)}.

Ranking metrics are calculated among the sampled logged events per user, not
against the full item catalog. Treat them as diagnostics rather than a complete
top-K recommendation evaluation. The dataset has no impression/exposure field,
so unobserved items cannot safely be labeled negative. Repeat the comparison on
multiple chronological cutoffs before claiming generalization.

## Reproduction

Place the official CSV at `data/UserBehavior.csv` (unzip it if downloaded as an
archive), install `requirements.txt`, then run:

```bash
python src/run_experiment.py --data data/UserBehavior.csv
```

Raw data is not bundled. `reports/results.json` contains the full experiment
output.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def run(data_path: Path, results_path: Path, report_path: Path, max_per_class: int, seed: int):
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Taobao UserBehavior CSV not found: {data_path}. "
            "Download the official dataset and pass its path with --data."
        )
    connection = duckdb.connect()
    try:
        total_rows, _ = create_event_tables(connection, data_path)
        sample_counts = create_sampled_candidates(connection, max_per_class, seed)
        print(f"Raw events: {total_rows:,}; sampled candidates: {sample_counts}")
        for relation, keys in RELATIONS.items():
            print(f"Aggregating history: {relation}")
            create_relation_features(connection, relation, keys, AUTO_WINDOWS)
        frame = fetch_feature_frame(connection, AUTO_WINDOWS)
    finally:
        connection.close()

    print(f"Training on {len(frame):,} sampled rows and {frame.shape[1] - 3} feature columns")
    results = {
        "dataset": "Taobao UserBehavior",
        "rows": total_rows,
        "sample_counts": sample_counts,
        "sampling_cap_per_class_per_split": max_per_class,
        "seed": seed,
        "manual": evaluate_variant("manual", frame, MANUAL_WINDOWS),
        "automatic": evaluate_variant("automatic", frame, AUTO_WINDOWS),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(results, report_path, max_per_class)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/UserBehavior.csv"))
    parser.add_argument("--results", type=Path, default=Path("reports/results.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/experiment.md"))
    parser.add_argument("--max-per-class", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_per_class <= 0:
        parser.error("--max-per-class must be positive")
    results = run(args.data, args.results, args.report, args.max_per_class, args.seed)
    for variant in ("manual", "automatic"):
        print(f"\n{variant}: {json.dumps(results[variant]['test'], ensure_ascii=False)}")
    print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
