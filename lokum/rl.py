from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from lokum.chat_template import ASSISTANT, USER, decode, get_tokenizer
from lokum.checkpoint import load_model, save_checkpoint
from lokum.generate import generate
from lokum.losses import gae, group_advantages, ppo_policy_loss
from lokum.verifier import score


class ValueHead(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.head = nn.Linear(n_embed, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden).squeeze(-1)


def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def prompt_ids(prompt: str, device: str) -> torch.Tensor:
    tok = get_tokenizer()
    ids = tok.encode_ordinary(USER + prompt) + [tok.eot_token] + tok.encode_ordinary(ASSISTANT)
    return torch.tensor([ids], dtype=torch.long, device=device)


def token_logprobs(model, idx: torch.Tensor) -> torch.Tensor:
    logits, _ = model(idx[:, :-1])
    logp = F.log_softmax(logits.float(), dim=-1)
    taken = idx[:, 1:].unsqueeze(-1)
    return logp.gather(-1, taken).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("ppo", "grpo"), required=True)
    parser.add_argument("--init", default="models/sft.pt")
    parser.add_argument("--train", default="data/lokum/rl_prompts_train.jsonl")
    parser.add_argument("--curriculum", default="data/lokum/arithmetic_prompts.jsonl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--curriculum-iters", type=int, default=50)
    parser.add_argument("--rollout-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl", type=float, default=0.04)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, cfg, _ = load_model(args.init, device)
    policy.train()
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    value = ValueHead(cfg.n_embed).to(device) if args.algo == "ppo" else None
    params = list(policy.parameters()) + (list(value.parameters()) if value else [])
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    tok = get_tokenizer()
    train_rows = load_rows(args.train)
    curriculum = load_rows(args.curriculum) if args.curriculum_iters else []
    group = 1 if args.algo == "ppo" else args.group_size
    if group < 2 and args.algo == "grpo":
        raise SystemExit("GRPO needs group-size >= 2")
    out = args.out or ("models/ppo.pt" if args.algo == "ppo" else "models/grpo.pt")
    for step in range(args.iterations):
        pool = curriculum if step < args.curriculum_iters and curriculum else train_rows
        # PPO: 4 prompts. GRPO: 2 prompts, each with group_size samples.
        n_prompts = 4 if args.algo == "ppo" else 2
        rewards_seen = []
        optimizer.zero_grad(set_to_none=True)
        pieces = n_prompts * group
        for p in range(n_prompts):
            row = pool[(step * n_prompts + p) % len(pool)]
            prompt = prompt_ids(row["prompt"], device)
            completions = []
            for _ in range(group):
                sample = generate(
                    policy,
                    prompt,
                    args.rollout_len,
                    args.temperature,
                    top_k=40,
                    eot=tok.eot_token,
                    top_p=0.9,
                )
                completions.append(sample[0])
            rewards = []
            for seq in completions:
                text = decode(seq.tolist())
                rewards.append(score(text, float(row["gold"])))
            rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)
            rewards_seen.extend(rewards)
            if args.algo == "grpo":
                adv_seq = group_advantages(rewards_t, group)
            for i, seq in enumerate(completions):
                seq = seq.unsqueeze(0)
                start = prompt.size(1) - 1
                # One update per sample, so the sampling policy and the new policy
                # are the same weights. The ratio starts at 1. A second pass over
                # stored log-probs is what makes the clip do work.
                new_lp = token_logprobs(policy, seq)[0]
                old_lp = new_lp.detach()
                with torch.no_grad():
                    ref_lp = token_logprobs(ref, seq)
                gen = torch.zeros(new_lp.size(0), device=device)
                gen[start:] = 1
                token_reward = torch.zeros_like(gen)
                token_reward[start:] = -args.kl * (old_lp[start:] - ref_lp[0, start:])
                last = min(seq.size(1) - 2, token_reward.size(0) - 1)
                token_reward[last] = token_reward[last] + rewards[i]
                if args.algo == "ppo":
                    hidden = policy.hidden(seq[:, :-1])
                    values = value(hidden)[0]
                    advantages = gae(token_reward, values.detach(), gen)
                    returns = advantages + values.detach()
                    value_loss = 0.5 * ((values - returns) ** 2 * gen).sum() / gen.sum().clamp(min=1)
                    piece = ppo_policy_loss(new_lp, old_lp, advantages.detach(), gen, args.clip) + value_loss
                else:
                    advantages = torch.full_like(gen, float(adv_seq[i]))
                    piece = ppo_policy_loss(new_lp, old_lp, advantages, gen, args.clip)
                (piece / pieces).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if step % 5 == 0:
            mean_r = sum(rewards_seen) / max(1, len(rewards_seen))
            print(f"{args.algo} step {step} reward {mean_r:.3f}")
    save_checkpoint(out, policy, optimizer, args.iterations, cfg)


if __name__ == "__main__":
    main()