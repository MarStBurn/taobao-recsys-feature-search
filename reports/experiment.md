# Taobao UserBehavior: automated RecSys feature search

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
- Up to 100,000 examples per class and split are sampled uniformly;
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

| Split | Metric | Manual | Auto search |
|---|---|---:|---:|
| validation | weighted_log_loss | 0.07008 | 0.06953 |
| validation | weighted_auc | 0.86922 | 0.87429 |
| validation | ndcg_at_10 | 0.50916 | 0.50901 |
| validation | recall_at_10 | 0.99999 | 0.99998 |
| test | weighted_log_loss | 0.07702 | 0.07648 |
| test | weighted_auc | 0.86786 | 0.87355 |
| test | ndcg_at_10 | 0.51279 | 0.51268 |
| test | recall_at_10 | 0.99999 | 0.99999 |

Sampled candidate counts: {"train": 200000, "validation": 200000, "test": 200000}.

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
