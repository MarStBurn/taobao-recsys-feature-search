# Lightweight KION sequential recommendation with RL

## Method

This is a sampled, CPU/GPU-friendly prototype. It caps the run at
10,000 users and 40 events per user, then
trains a 2-layer, 64-wide causal Transformer with item IDs
as tokens. Training predicts the next item from the user's prior history.
KION timestamps are date-granular, so interactions from the same user on the
same date share one prefix and are appended only after all targets on that date
are formed.

A separate reward model predicts `watched_pct` from user, candidate item, prior
items, historical watch percentage, and recency. The policy is then updated
with sampled REINFORCE actions and a KL penalty to its frozen supervised
checkpoint. It is a one-step contextual policy update; five-step greedy
rollouts are diagnostic only.

## Data and protocol

- Input: KION `interactions.csv`, with `user_id`, `item_id`, `last_watch_dt`,
  and `watched_pct`.
- Source: [public KION dataset repository](https://github.com/irsafilo/KION_DATASET).
- Rows in the downloaded interaction archive: 1,594,787; sampled
  interactions: 27,725
  (1.74%).
- The dataset repository describes a broader public release of 5,476,251
  interactions; this run uses only the smaller `interactions.csv` archive
  available from its root.
- Users are sampled deterministically; each user's retained interactions are
  sorted in time and capped. Chronological 80/10/10 partitions use distinct
  timestamp cutoffs. Item vocabulary is built from train only; unseen test
  targets are excluded from ranking metrics.
- Reward is `watched_pct` scaled to [0, 1]. The reward model is evaluated by
  MAE, RMSE, and Spearman correlation on validation/test.
- Ranking uses Recall@10 and NDCG@10 on observed next-item targets. Coverage,
  top-item concentration, and repeat rate are reported to expose collapse.

## Results

| Test metric | Supervised baseline | RL-updated |
|---|---:|---:|
| Recall@10 | 0.21498 | 0.17319 |
| NDCG@10 | 0.14402 | 0.11128 |
| Sampled train-catalog coverage@10 | 0.20589 | 0.12615 |
| Top-1 item share | 0.30836 | 0.58396 |
| Repeat rate@10 | 0.10444 | 0.09221 |
| Reward-model score of top-1 | 0.57418 | 0.59857 |

Reward-model held-out metrics: `{"validation": {"mae": 0.37412598729133606, "rmse": 0.4027985160063499, "spearman": 0.15219538493644577}, "test": {"mae": 0.3752113878726959, "rmse": 0.40428983525583423, "spearman": 0.11135200615807764}}`.

Five-step greedy rollout: baseline `{"predicted_reward_by_step": [0.5762999653816223, 0.5844436883926392, 0.5866204500198364, 0.58845454454422, 0.5900337100028992], "repeat_rate_by_step": [0.3125, 0.9921875, 1.0, 1.0, 1.0], "generated_unique_items": 867}`;
RL `{"predicted_reward_by_step": [0.5976830124855042, 0.5961645841598511, 0.5992922782897949, 0.6018116474151611, 0.6039342880249023], "repeat_rate_by_step": [0.21484375, 0.99609375, 1.0, 1.0, 1.0], "generated_unique_items": 866}`.

## Interpretation

RL raised the reward model's top-1 predicted watch percentage by
+0.0244, but changed Recall@10 by -0.0418 and NDCG@10 by
-0.0327. Top-1 concentration changed by +0.2756.
The reward model's test Spearman correlation is only 0.111,
so its ranking signal is weak. This run is a negative result for reward-only
policy optimization: it obtains higher *predicted* reward while relevance and
catalog coverage fall and recommendations become more concentrated. The
five-step rollouts also show rapid repetition, underscoring that generated
items are not observed user actions.

## Caveats

The logged data contains outcomes only for items users watched. Reward-model
scores for unobserved candidates are extrapolations, so an RL reward increase
is not evidence of improved user satisfaction. The policy uses the reward model
as a surrogate and may exploit its errors; KL regularization, ranking metrics,
coverage, and repeat-rate checks are safeguards, not causal validation.
Offline ranking evaluates only observed next items, not a randomized full-catalog
candidate set. Date-only timestamps also prevent reliable within-day ordering.
An online experiment is required to establish user benefit.

## Reproduction

Put the public KION `interactions.csv` at `data/kion/interactions.csv`, install
`kion_rl/requirements.txt`, then run:

```bash
python kion_rl/run_experiment.py --data data/kion/interactions.csv
```

The full metrics are written to `kion_rl/reports/results.json`.
