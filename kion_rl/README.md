# KION sequential recommendation with lightweight RL

This sampled experiment covers next-item prediction with a small GPT-style
causal Transformer, a `watched_pct` reward model, and a KL-constrained
REINFORCE update. It preserves the existing Taobao feature-search experiment
in the repository.

## Data

Download the KION competition data separately and place `train.csv` at
`data/kion/train.csv`. Required columns are `user_id`, `item_id`,
`last_watch_dt`, and `watched_pct`. The dataset is not redistributed here.

## Run

```bash
python3 -m venv .venv-kion
. .venv-kion/bin/activate
pip install -r kion_rl/requirements.txt
python kion_rl/run_experiment.py --data data/kion/train.csv
```

Defaults cap the experiment at 10,000 users and 40 events per user, use a
2-layer/64-wide Transformer, and train for four epochs. Sampling and training
settings are configurable from the CLI. For a quick smoke run, reduce
`--max-users`, `--epochs`, `--reward-epochs`, and `--rl-steps`.

The report and JSON metrics are written under `kion_rl/reports/`. The report
distinguishes observed-item offline ranking from the reward model's unsupported
scores on unobserved candidates.
