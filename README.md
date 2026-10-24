# Taobao UserBehavior: automated RecSys feature search

The experiment compares LightGBM with a hand-picked set of time windows against
the same model trained on a larger, automatically enumerated feature set.
Features aggregate prior events for user, item, user-item, user-category, and
category relationships.

## Dataset

Obtain the Taobao UserBehavior dataset from its official Alibaba/Tianchi
distribution, unzip it, and put the headerless CSV at `data/UserBehavior.csv`.
Each row must have these columns in order:

```text
user_id,item_id,category_id,behavior_type,timestamp
```

The raw dataset is not included in this repository. The script reads the full
file through DuckDB and only materializes a capped sample and its features for
model fitting.

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py --data data/UserBehavior.csv
```

`--max-per-class` controls the maximum sample count per target class and time
split (default: 100,000); `--seed` controls deterministic sampling. The run
creates `reports/experiment.md` with measured metrics and
`reports/results.json` with the full result.

The target is whether a logged event is a purchase (`behavior_type=buy`) versus
another logged behavior (`pv`/`view`, `cart`, or `fav`). This is not conversion
prediction over ad impressions: the dataset does not identify exposed but
unobserved items. See the report for the temporal protocol and evaluation caveats.
