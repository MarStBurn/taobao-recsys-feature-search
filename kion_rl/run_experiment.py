#!/usr/bin/env python3
"""Small KION next-item / reward-model / policy-gradient experiment."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.distributions import Categorical, kl_divergence
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Examples:
    contexts: np.ndarray
    lengths: np.ndarray
    targets: np.ndarray
    users: np.ndarray
    history_mean: np.ndarray
    gap_days: np.ndarray
    rewards: np.ndarray

    def __len__(self) -> int:
        return len(self.targets)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_kion(path: Path, max_users: int, max_events_per_user: int, seed: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"KION CSV not found: {path}. Expected user_id,item_id,last_watch_dt,watched_pct."
        )
    frame = pd.read_csv(path, usecols=lambda name: name in {
        "user_id", "item_id", "last_watch_dt", "watched_pct"
    })
    required = {"user_id", "item_id", "last_watch_dt", "watched_pct"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"KION CSV is missing columns: {sorted(missing)}")

    frame = frame.dropna(subset=["user_id", "item_id", "last_watch_dt", "watched_pct"]).copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    frame["timestamp"] = pd.to_datetime(frame["last_watch_dt"], errors="coerce", utc=True)
    frame["reward"] = pd.to_numeric(frame["watched_pct"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "reward"])
    if frame.empty:
        raise ValueError("No rows with valid timestamps and watched_pct were found.")

    if frame["reward"].quantile(0.95) > 1.0:
        frame["reward"] = frame["reward"] / 100.0
    frame["reward"] = frame["reward"].clip(0.0, 1.0).astype("float32")
    frame["timestamp"] = (frame["timestamp"].astype("int64") // 1_000_000_000).astype("int64")

    rng = np.random.default_rng(seed)
    users = np.array(sorted(frame["user_id"].unique()))
    if max_users > 0 and len(users) > max_users:
        selected = set(rng.choice(users, size=max_users, replace=False))
        frame = frame[frame["user_id"].isin(selected)]

    frame = frame.sort_values(
        ["user_id", "timestamp", "item_id"], kind="stable"
    )
    if max_events_per_user > 0:
        frame = frame.groupby("user_id", sort=False, group_keys=False).tail(max_events_per_user)
    frame = frame.sort_values(["timestamp", "user_id", "item_id"], kind="stable").reset_index(drop=True)

    unique_times = np.sort(frame["timestamp"].unique())
    if len(unique_times) < 3:
        raise ValueError("At least three distinct timestamps are required for a temporal split.")
    train_cut = unique_times[max(1, int(len(unique_times) * 0.8))]
    valid_cut = unique_times[min(len(unique_times) - 1, max(int(len(unique_times) * 0.9), 2))]
    if valid_cut <= train_cut:
        raise ValueError("Could not create three non-empty chronological partitions.")
    frame["split"] = np.select(
        [frame["timestamp"] < train_cut, frame["timestamp"] < valid_cut],
        ["train", "validation"],
        default="test",
    )
    frame.attrs["train_cut"] = int(train_cut)
    frame.attrs["valid_cut"] = int(valid_cut)
    return frame


def pack_examples(
    examples: list[dict[str, Any]], max_seq_len: int
) -> Examples:
    contexts = np.zeros((len(examples), max_seq_len), dtype=np.int64)
    lengths = np.zeros(len(examples), dtype=np.int64)
    for row, example in enumerate(examples):
        history = example["history"][-max_seq_len:]
        lengths[row] = len(history)
        if history:
            contexts[row, : len(history)] = history
    return Examples(
        contexts=contexts,
        lengths=lengths,
        targets=np.asarray([row["target"] for row in examples], dtype=np.int64),
        users=np.asarray([row["user"] for row in examples], dtype=np.int64),
        history_mean=np.asarray([row["history_mean"] for row in examples], dtype=np.float32),
        gap_days=np.asarray([row["gap_days"] for row in examples], dtype=np.float32),
        rewards=np.asarray([row["reward"] for row in examples], dtype=np.float32),
    )


def build_examples(
    frame: pd.DataFrame, max_seq_len: int
) -> tuple[dict[str, Examples], dict[str, int], dict[str, int]]:
    train_items = sorted(frame.loc[frame["split"] == "train", "item_id"].unique())
    item_to_index = {item: index + 2 for index, item in enumerate(train_items)}
    user_ids = sorted(frame["user_id"].unique())
    user_to_index = {user: index + 1 for index, user in enumerate(user_ids)}
    collected: dict[str, list[dict[str, Any]]] = {
        "train": [], "validation": [], "test": []
    }

    for user_id, user_frame in frame.groupby("user_id", sort=False):
        history: list[int] = []
        history_rewards: list[float] = []
        history_times: list[int] = []
        for timestamp, day in user_frame.groupby("timestamp", sort=False):
            # Timestamps are date-granular in KION; same-day events share the
            # same prefix so arbitrary CSV order cannot leak into the target.
            for row in day.itertuples(index=False):
                target = item_to_index.get(row.item_id, 1)
                if history and target >= 2:
                    collected[row.split].append(
                        {
                            "history": history,
                            "target": target,
                            "user": user_to_index[user_id],
                            "history_mean": float(np.mean(history_rewards)),
                            "gap_days": max(0.0, (int(timestamp) - history_times[-1]) / 86_400),
                            "reward": float(row.reward),
                        }
                    )
            for row in day.itertuples(index=False):
                history.append(item_to_index.get(row.item_id, 1))
                history_rewards.append(float(row.reward))
                history_times.append(int(timestamp))

    packed = {name: pack_examples(rows, max_seq_len) for name, rows in collected.items()}
    if any(len(packed[split]) == 0 for split in ("train", "validation", "test")):
        counts = {name: len(data) for name, data in packed.items()}
        raise ValueError(f"Temporal sampling yielded an empty example partition: {counts}")
    return packed, item_to_index, user_to_index


class SmallGPT2(nn.Module):
    """Compact causal Transformer decoder with item IDs as vocabulary tokens."""

    def __init__(
        self, vocab_size: int, max_seq_len: int, width: int, layers: int, heads: int
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("model width must be divisible by the number of attention heads")
        self.token_embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, width)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            block, num_layers=layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.max_seq_len = max_seq_len
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal = torch.triu(
            torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        hidden = self.blocks(
            hidden,
            mask=causal,
            src_key_padding_mask=tokens.eq(0),
        )
        batch = torch.arange(tokens.shape[0], device=tokens.device)
        last = self.final_norm(hidden[batch, lengths - 1])
        return self.lm_head(last)


class RewardModel(nn.Module):
    def __init__(self, n_users: int, vocab_size: int, width: int = 32) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, width, padding_idx=0)
        self.item_embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        self.network = nn.Sequential(
            nn.Linear(4 * width + 3, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        users: torch.Tensor,
        candidates: torch.Tensor,
        contexts: torch.Tensor,
        history_mean: torch.Tensor,
        gap_days: torch.Tensor,
    ) -> torch.Tensor:
        embedded_history = self.item_embedding(contexts)
        mask = contexts.ne(0).unsqueeze(-1)
        count = mask.sum(dim=1).clamp(min=1)
        history_mean_embedding = (embedded_history * mask).sum(dim=1) / count
        lengths = contexts.ne(0).sum(dim=1)
        positions = (lengths - 1).clamp(min=0)
        batch = torch.arange(contexts.shape[0], device=contexts.device)
        last_item = self.item_embedding(contexts[batch, positions])
        last_item = last_item * lengths.gt(0).unsqueeze(-1)
        numeric = torch.stack(
            [
                torch.log1p(lengths.float()),
                history_mean,
                torch.log1p(gap_days.clamp(min=0, max=365)),
            ],
            dim=1,
        )
        features = torch.cat(
            [
                self.user_embedding(users),
                self.item_embedding(candidates),
                last_item,
                history_mean_embedding,
                numeric,
            ],
            dim=1,
        )
        return self.network(features).squeeze(1)


def to_dataset(data: Examples, include_reward: bool) -> TensorDataset:
    tensors = [
        torch.from_numpy(data.contexts),
        torch.from_numpy(data.lengths),
        torch.from_numpy(data.targets),
        torch.from_numpy(data.users),
        torch.from_numpy(data.history_mean),
        torch.from_numpy(data.gap_days),
    ]
    if include_reward:
        tensors.append(torch.from_numpy(data.rewards))
    return TensorDataset(*tensors)


def mask_special_tokens(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.clone()
    logits[:, :2] = torch.finfo(logits.dtype).min
    return logits


@torch.no_grad()
def next_item_loss(model: SmallGPT2, data: Examples, device: torch.device, batch_size: int) -> float:
    model.eval()
    loader = DataLoader(to_dataset(data, include_reward=False), batch_size=batch_size)
    total_loss, total_count = 0.0, 0
    for contexts, lengths, targets, *_ in loader:
        contexts, lengths, targets = contexts.to(device), lengths.to(device), targets.to(device)
        logits = mask_special_tokens(model(contexts, lengths))
        loss = nn.functional.cross_entropy(logits, targets, reduction="sum")
        total_loss += float(loss)
        total_count += len(targets)
    return total_loss / max(total_count, 1)


def train_policy(
    model: SmallGPT2,
    train: Examples,
    validation: Examples,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> float:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loader = DataLoader(
        to_dataset(train, include_reward=False), batch_size=batch_size, shuffle=True
    )
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        for contexts, lengths, targets, *_ in loader:
            contexts, lengths, targets = contexts.to(device), lengths.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = mask_special_tokens(model(contexts, lengths))
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_loss = next_item_loss(model, validation, device, batch_size)
        print(f"  next-item epoch {epoch + 1}/{epochs}: val_loss={validation_loss:.4f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best_loss


def reward_batches(data: Examples, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(to_dataset(data, include_reward=True), batch_size=batch_size, shuffle=shuffle)


def train_reward_model(
    model: RewardModel,
    train: Examples,
    validation: Examples,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        for contexts, _, targets, users, history_mean, gap_days, rewards in reward_batches(
            train, batch_size, shuffle=True
        ):
            tensors = [x.to(device) for x in (
                contexts, targets, users, history_mean, gap_days, rewards
            )]
            contexts, targets, users, history_mean, gap_days, rewards = tensors
            optimizer.zero_grad(set_to_none=True)
            prediction = model(users, targets, contexts, history_mean, gap_days)
            loss = nn.functional.mse_loss(prediction, rewards)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_metrics = evaluate_reward_model(model, validation, device, batch_size)
        print(
            f"  reward epoch {epoch + 1}/{epochs}: "
            f"val_rmse={validation_metrics['rmse']:.4f}"
        )
        if validation_metrics["rmse"] < best_loss:
            best_loss = validation_metrics["rmse"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)


@torch.no_grad()
def evaluate_reward_model(
    model: RewardModel, data: Examples, device: torch.device, batch_size: int
) -> dict[str, float]:
    model.eval()
    predictions, targets = [], []
    for contexts, _, candidates, users, history_mean, gap_days, rewards in reward_batches(
        data, batch_size
    ):
        prediction = model(
            users.to(device), candidates.to(device), contexts.to(device),
            history_mean.to(device), gap_days.to(device),
        )
        predictions.extend(prediction.cpu().numpy())
        targets.extend(rewards.numpy())
    predicted = np.asarray(predictions)
    truth = np.asarray(targets)
    correlation = spearmanr(truth, predicted).statistic if len(truth) > 1 else 0.0
    return {
        "mae": float(mean_absolute_error(truth, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(truth, predicted))),
        "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
    }


@torch.no_grad()
def evaluate_policy(
    policy: SmallGPT2,
    reward_model: RewardModel,
    data: Examples,
    n_items: int,
    device: torch.device,
    batch_size: int,
    top_k: int = 10,
) -> dict[str, float]:
    policy.eval()
    reward_model.eval()
    loader = DataLoader(to_dataset(data, include_reward=True), batch_size=batch_size)
    hits, ndcgs, recommended, repeated = [], [], [], []
    predicted_rewards = []
    for contexts, lengths, targets, users, history_mean, gap_days, _ in loader:
        contexts_device, lengths_device = contexts.to(device), lengths.to(device)
        logits = mask_special_tokens(policy(contexts_device, lengths_device))
        top = logits.topk(min(top_k, n_items), dim=1).indices.cpu().numpy()
        target_array = targets.numpy()
        for i, row in enumerate(top):
            positions = np.flatnonzero(row == target_array[i])
            rank = int(positions[0]) + 1 if len(positions) else 0
            hits.append(float(rank > 0))
            ndcgs.append(1.0 / math.log2(rank + 1) if rank > 0 else 0.0)
            history = set(contexts[i][contexts[i] > 1].tolist())
            recommended.extend(row.tolist())
            repeated.extend([float(item in history) for item in row])

        top1 = torch.from_numpy(top[:, 0]).to(device)
        predicted_rewards.extend(
            reward_model(
                users.to(device), top1, contexts.to(device),
                history_mean.to(device), gap_days.to(device),
            ).cpu().numpy()
        )

    top1 = recommended[:: min(top_k, n_items)]
    unique_count = len(set(recommended))
    item_counts = pd.Series(top1).value_counts()
    return {
        "recall_at_10": float(np.mean(hits)),
        "ndcg_at_10": float(np.mean(ndcgs)),
        "catalog_coverage_at_10": unique_count / max(n_items, 1),
        "top1_item_share": float(item_counts.iloc[0] / len(top1)) if len(top1) else 0.0,
        "repeat_rate_at_10": float(np.mean(repeated)) if repeated else 0.0,
        "predicted_watch_pct_top1": float(np.mean(predicted_rewards)),
    }


def policy_gradient(
    policy: SmallGPT2,
    reference: SmallGPT2,
    reward_model: RewardModel,
    train: Examples,
    device: torch.device,
    steps: int,
    batch_size: int,
    learning_rate: float,
    kl_beta: float,
    seed: int,
) -> list[float]:
    if steps <= 0:
        return []
    policy.to(device).train()
    reference.to(device).eval()
    reward_model.to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    for parameter in reward_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    dataset = to_dataset(train, include_reward=True)
    device_data = [tensor.to(device) for tensor in dataset.tensors]
    step_rewards = []

    for step in range(steps):
        indices = torch.as_tensor(
            rng.integers(0, len(train), size=min(batch_size, len(train))),
            dtype=torch.long,
            device=device,
        )
        contexts, lengths, targets, users, history_mean, gap_days, _ = [
            tensor[indices] for tensor in device_data
        ]
        logits = mask_special_tokens(policy(contexts, lengths))
        with torch.no_grad():
            reference_logits = mask_special_tokens(reference(contexts, lengths))
            reference_distribution = Categorical(logits=reference_logits)
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        reference_actions = reference_distribution.sample()
        action_reward = reward_model(users, actions, contexts, history_mean, gap_days)
        with torch.no_grad():
            baseline_reward = reward_model(
                users, reference_actions, contexts, history_mean, gap_days
            )
        advantage = action_reward.detach() - baseline_reward
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
        divergence = kl_divergence(distribution, reference_distribution)
        loss = -(advantage * distribution.log_prob(actions)).mean() + kl_beta * divergence.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        step_rewards.append(float(action_reward.mean().detach().cpu()))
        if (step + 1) % max(1, steps // 5) == 0:
            print(
                f"  RL step {step + 1}/{steps}: "
                f"sample_reward={step_rewards[-1]:.4f}, "
                f"kl={divergence.mean().detach().item():.4f}"
            )
    policy.eval()
    return step_rewards


@torch.no_grad()
def rollout_metrics(
    policy: SmallGPT2,
    reward_model: RewardModel,
    data: Examples,
    device: torch.device,
    seed: int,
    max_examples: int = 256,
    steps: int = 5,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(data), size=min(len(data), max_examples), replace=False)
    sequences = [
        data.contexts[index][data.contexts[index] > 0].tolist()
        for index in selected
    ]
    users = torch.from_numpy(data.users[selected]).to(device)
    history_means = torch.from_numpy(data.history_mean[selected]).to(device)
    gaps = torch.from_numpy(data.gap_days[selected]).to(device)
    rewards_by_step, repeats_by_step = [], []
    policy.eval()
    reward_model.eval()
    for _ in range(steps):
        tokens = np.zeros((len(sequences), policy.max_seq_len), dtype=np.int64)
        lengths = np.asarray([min(len(seq), policy.max_seq_len) for seq in sequences], dtype=np.int64)
        for row, sequence in enumerate(sequences):
            recent = sequence[-policy.max_seq_len:]
            tokens[row, : len(recent)] = recent
        context_tensor = torch.from_numpy(tokens).to(device)
        length_tensor = torch.from_numpy(lengths).to(device)
        logits = mask_special_tokens(policy(context_tensor, length_tensor))
        actions = logits.argmax(dim=1)
        reward = reward_model(
            users, actions, context_tensor, history_means, gaps
        )
        rewards_by_step.append(float(reward.mean().cpu()))
        repeated = []
        for row, action in enumerate(actions.cpu().tolist()):
            repeated.append(float(action in sequences[row]))
            sequences[row] = (sequences[row] + [action])[-policy.max_seq_len:]
        repeats_by_step.append(float(np.mean(repeated)))
    return {
        "predicted_reward_by_step": rewards_by_step,
        "repeat_rate_by_step": repeats_by_step,
        "generated_unique_items": len({item for sequence in sequences for item in sequence}),
    }


def write_report(results: dict[str, Any], path: Path, args: argparse.Namespace) -> None:
    base, tuned = results["baseline"]["test"], results["rl"]["test"]
    report = f"""# Lightweight KION sequential recommendation with RL

## Method

This is a sampled, CPU/GPU-friendly prototype. It caps the run at
{args.max_users:,} users and {args.max_events_per_user} events per user, then
trains a {args.layers}-layer, {args.width}-wide causal Transformer with item IDs
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

- Input: KION `train.csv`, with `user_id`, `item_id`, `last_watch_dt`,
  and `watched_pct`.
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
| Recall@10 | {base["recall_at_10"]:.5f} | {tuned["recall_at_10"]:.5f} |
| NDCG@10 | {base["ndcg_at_10"]:.5f} | {tuned["ndcg_at_10"]:.5f} |
| Catalog coverage@10 | {base["catalog_coverage_at_10"]:.5f} | {tuned["catalog_coverage_at_10"]:.5f} |
| Top-1 item share | {base["top1_item_share"]:.5f} | {tuned["top1_item_share"]:.5f} |
| Repeat rate@10 | {base["repeat_rate_at_10"]:.5f} | {tuned["repeat_rate_at_10"]:.5f} |
| Reward-model score of top-1 | {base["predicted_watch_pct_top1"]:.5f} | {tuned["predicted_watch_pct_top1"]:.5f} |

Reward-model held-out metrics: `{json.dumps(results["reward_model"], ensure_ascii=False)}`.

Five-step greedy rollout: baseline `{json.dumps(results["baseline"]["rollout"], ensure_ascii=False)}`;
RL `{json.dumps(results["rl"]["rollout"], ensure_ascii=False)}`.

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

Put KION `train.csv` at `data/kion/train.csv`, install
`kion_rl/requirements.txt`, then run:

```bash
python kion_rl/run_experiment.py --data data/kion/train.csv
```

The full metrics are written to `kion_rl/reports/results.json`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, args.threads))
    device = choose_device(args.device)
    print(f"Using device: {device}")

    frame = load_kion(args.data, args.max_users, args.max_events_per_user, seed)
    print(
        f"Sampled interactions: {len(frame):,}; users: {frame.user_id.nunique():,}; "
        f"train/validation/test rows: "
        f"{(frame.split == 'train').sum():,}/"
        f"{(frame.split == 'validation').sum():,}/"
        f"{(frame.split == 'test').sum():,}"
    )
    examples, item_to_index, user_to_index = build_examples(frame, args.max_seq_len)
    print(
        f"Next-item examples train/validation/test: "
        f"{len(examples['train']):,}/{len(examples['validation']):,}/"
        f"{len(examples['test']):,}; catalog={len(item_to_index):,}"
    )

    vocab_size = len(item_to_index) + 2
    n_items = vocab_size - 2
    policy = SmallGPT2(
        vocab_size, args.max_seq_len, args.width, args.layers, args.heads
    ).to(device)
    baseline_validation_loss = train_policy(
        policy, examples["train"], examples["validation"], device,
        args.epochs, args.batch_size, args.learning_rate,
    )
    baseline = copy.deepcopy(policy).eval()

    reward_model = RewardModel(len(user_to_index) + 1, vocab_size).to(device)
    train_reward_model(
        reward_model, examples["train"], examples["validation"], device,
        args.reward_epochs, args.batch_size, args.reward_learning_rate,
    )
    reward_metrics = {
        split: evaluate_reward_model(reward_model, examples[split], device, args.batch_size)
        for split in ("validation", "test")
    }

    baseline_test = evaluate_policy(
        baseline, reward_model, examples["test"], n_items, device, args.batch_size
    )
    baseline_rollout = rollout_metrics(
        baseline, reward_model, examples["test"], device, seed
    )

    reference = copy.deepcopy(baseline).eval()
    rl_policy = copy.deepcopy(baseline).to(device)
    rl_rewards = policy_gradient(
        rl_policy, reference, reward_model, examples["train"], device,
        args.rl_steps, args.batch_size, args.rl_learning_rate, args.kl_beta, seed,
    )
    rl_test = evaluate_policy(
        rl_policy, reward_model, examples["test"], n_items, device, args.batch_size
    )
    rl_rollout = rollout_metrics(
        rl_policy, reward_model, examples["test"], device, seed
    )

    results = {
        "dataset": "KION",
        "seed": seed,
        "device": str(device),
        "sampled_interactions": int(len(frame)),
        "sampled_users": int(frame.user_id.nunique()),
        "catalog_size": int(n_items),
        "examples": {split: len(data) for split, data in examples.items()},
        "baseline_validation_next_item_loss": baseline_validation_loss,
        "reward_model": reward_metrics,
        "baseline": {
            "test": baseline_test,
            "rollout": baseline_rollout,
        },
        "rl": {
            "test": rl_test,
            "rollout": rl_rollout,
            "last_20_sampled_reward_mean": (
                float(np.mean(rl_rewards[-20:])) if rl_rewards else None
            ),
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(results, args.report, args)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/kion/train.csv"))
    parser.add_argument("--results", type=Path, default=Path("kion_rl/reports/results.json"))
    parser.add_argument("--report", type=Path, default=Path("kion_rl/reports/experiment.md"))
    parser.add_argument("--max-users", type=int, default=10_000)
    parser.add_argument("--max-events-per-user", type=int, default=40)
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--reward-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--reward-learning-rate", type=float, default=1e-3)
    parser.add_argument("--rl-learning-rate", type=float, default=3e-5)
    parser.add_argument("--rl-steps", type=int, default=100)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.max_users < 0 or args.max_events_per_user < 0:
        parser.error("sampling limits must be non-negative (0 means unlimited)")
    if args.max_seq_len < 1 or args.width < 1 or args.layers < 1 or args.heads < 1:
        parser.error("model dimensions must be positive")
    if args.epochs < 1 or args.reward_epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if args.rl_steps < 0 or args.kl_beta < 0:
        parser.error("--rl-steps and --kl-beta must be non-negative")
    return args


if __name__ == "__main__":
    options = parse_args()
    output = run(options)
    print("Test metrics:")
    print(json.dumps({
        "baseline": output["baseline"]["test"],
        "rl": output["rl"]["test"],
        "reward_model": output["reward_model"]["test"],
    }, indent=2, ensure_ascii=False))
