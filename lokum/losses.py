from __future__ import annotations

import torch
import torch.nn.functional as F


def sft_loss(logits: torch.Tensor, tokens: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    logits = logits[:, :-1, :]
    targets = tokens[:, 1:]
    mask = loss_mask[:, 1:].to(logits.dtype)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="none")
    ce = ce.reshape(targets.shape) * mask
    return ce.sum() / mask.sum().clamp(min=1.0)


def sequence_logprobs(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum of token log-probs where mask is 1. logits/tokens/mask are (B, T)."""
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    chosen = tokens[:, 1:].unsqueeze(-1)
    token_logp = logp.gather(-1, chosen).squeeze(-1)
    return (token_logp * mask[:, 1:].to(token_logp.dtype)).sum(dim=-1)


def bradley_terry(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(chosen - rejected).mean()


def dpo_loss(
    pi_chosen: torch.Tensor,
    pi_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
    loss = -F.logsigmoid(beta * logits).mean()
    accuracy = (logits > 0).float().mean()
    return loss, accuracy


def gae(rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, gamma: float = 1.0, lam: float = 0.95) -> torch.Tensor:
    """rewards, values, mask are (T,). mask is 1 on generated tokens."""
    adv = torch.zeros_like(rewards)
    last = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for t in range(rewards.size(0) - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < values.size(0) else rewards.new_zeros(())
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
        if mask[t] == 0:
            last = rewards.new_zeros(())
    return adv


def ppo_policy_loss(new_logp: torch.Tensor, old_logp: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor, clip: float = 0.2) -> torch.Tensor:
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    chosen = torch.minimum(unclipped, clipped)
    mask = mask.to(chosen.dtype)
    return -(chosen * mask).sum() / mask.sum().clamp(min=1.0)


def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    grouped = rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    return (centered / grouped.std(dim=1, keepdim=True).clamp(min=1e-4)).reshape(-1)