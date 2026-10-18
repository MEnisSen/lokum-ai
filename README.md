# Lokum

Lokum is a small Turkish language model I trained from scratch on my own machine, an RTX 4090 with 24 GB, over a few days. I wanted to see how far one GPU and a short schedule could get: a tokenizer, a pretrain, supervised fine-tuning, a reward head, DPO, PPO, and GRPO, then a look at whether any of that produced a model that can answer a Turkish grade-school question.

The network is about **163.6 million parameters** (163,618,560 with tied embeddings). The code in this repository is the trainer. The token files and the checkpoints stay on disk; they are too large for git, and several of the source datasets carry their own licenses.

## Where the design came from

Two public repos set the outline. Neither one is a dependency, and neither one is copied into this tree.

- [ufakai/ufakzeka](https://github.com/ufakai/ufakzeka) is the data list and the shape of the network: 24 layers, width 768, 12 query heads and 4 key/value heads, SwiGLU hidden size 2048, RoPE. It does not ship the corpus, and it does not ship this trainer.
- [FareedKhan-dev/train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch) is the map of the stages: data, a 2017-style block, a modern block, pretrain, generation, supervised fine-tuning, a reward model, DPO, PPO, GRPO, and eval.

`lokum/classic.py` is the 2017-style decoder (ReLU MLP, causal mask, one head per query). It is there to read. The checkpoints were trained with `lokum/model.py`.

## What was trained

| | |
|---|---|
| Parameters | 163,618,560 |
| Layers, width | 24, 768 |
| Attention | grouped-query, 12 query heads, 4 key/value heads, head size 64, RoPE |
| MLP | SwiGLU, hidden size 2048 |
| Vocabulary | 16,384 byte-level BPE, trained on Turkish. Digits stay one character each. The only special id is `<\|endoftext\|>` (16383). |
| Context | 1,024 tokens |
| Pretrain tokens | 3,000,000,291 train, 14,330,767 validation, in `data/lokum/train.h5` and `val.h5` |
| Optimizer | AdamW, cosine decay, learning rate `3e-4` down to `3e-5`, 100 warmup steps, weight decay 0.1, grad clip 1.0 |
| Pretrain batch | microbatch 4, sequence 1024, grad accumulation 128 (524,288 tokens/step), 6,000 steps, bfloat16 on CUDA |

Chat turns are ordinary text, not extra vocabulary rows. The markers live in `lokum/chat_template.py`: `<|user|>`, `<|assistant|>`, `<think>`, `</think>`, `<answer>`, `</answer>`.

## Datasets

The pretrain mix is capped at 3 billion training tokens. The loop streams from the Hugging Face hub and stops at each cap, so it does not download FineWeb in full. About 1 document in 200 is held out until validation reaches its cap. A document is cut at 100,000 characters. On this run the train file landed 291 tokens past the 3 billion cap, which is one document overshooting the stop check.

| Share | Tokens | Dataset | Config |
|---|---:|---|---|
| Web | 1,800,000,000 | [epfml/FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) | `tur_Latn` |
| Wiki | 375,000,000 | [HuggingFaceFW/finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) | `tr` |
| PDFs | 225,000,000 | [HuggingFaceFW/finepdfs-edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu) | `tur_Latn` |
| Synthetic prose | 300,000,000 | [Berkesule/COSMOS-Sentetic-Turkish-Corpus-2GB](https://huggingface.co/datasets/Berkesule/COSMOS-Sentetic-Turkish-Corpus-2GB) | — |
| Stories | 150,000,000 | [BILGEM-AI/BILGE-Synthetic-Stories](https://huggingface.co/datasets/BILGEM-AI/BILGE-Synthetic-Stories) | — |
| Math | 75,000,000 | [BILGEM-AI/BILGE-Synthetic-Math](https://huggingface.co/datasets/BILGEM-AI/BILGE-Synthetic-Math) | — |
| Math | 75,000,000 | [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | `finemath-4plus` |

FineMath is English on purpose. ufakzeka kept a few percent of English math in an otherwise Turkish mix. FineWeb, FinePDFs, and FineMath are ODC-By. FineWiki is CC-BY-SA. Those notices belong with the data. The token files are not in this git repo.

The tokenizer was trained on the first 200 MB of a local Turkish text file, `data/raw/turkish_sample.txt`. A Wikipedia extract is enough. That file is local too.

Supervised fine-tuning packed conversations into 145,740 train rows and 2,970 dev rows of length 1,024 (`data/lokum/sft_train.h5`, `sft_dev.h5`). Loss is on assistant tokens only. Sources, in order:

- [AlicanKiraz0/Turkish-SFT-Dataset-v1.0](https://huggingface.co/datasets/AlicanKiraz0/Turkish-SFT-Dataset-v1.0)
- [AlicanKiraz0/Turkce-Atlas-Instruct](https://huggingface.co/datasets/AlicanKiraz0/Turkce-Atlas-Instruct)
- [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset), rows whose language is Turkish
- [SoAp9035/everyday-conversations-tur](https://huggingface.co/datasets/SoAp9035/everyday-conversations-tur)
- [alibayram/diyalog-dataset](https://huggingface.co/datasets/alibayram/diyalog-dataset)
- [Metin/WikiRAG-TR](https://huggingface.co/datasets/Metin/WikiRAG-TR)
- [selimc/InstructPapers-TR](https://huggingface.co/datasets/selimc/InstructPapers-TR)
- [ytu-ce-cosmos/gsm8k_tr](https://huggingface.co/datasets/ytu-ce-cosmos/gsm8k_tr), formatted with think and answer tags, split on `####` when that marker is present

Preference pairs came from [selimc/orpo-dpo-mix-TR-20k](https://huggingface.co/datasets/selimc/orpo-dpo-mix-TR-20k): 19,266 train rows and 390 dev rows (`preferences.jsonl`, `preferences_test.jsonl`). Every 50th row is held out.

Reinforcement learning used the same GSM8K Turkish set. Every 50th row is the test file: 8,593 train prompts and 175 test prompts (`rl_prompts_train.jsonl`, `rl_prompts_test.jsonl`). A separate arithmetic warm-up, `arithmetic_prompts.jsonl`, has 2,000 lines of the form `Kaç eder: a + b?` with `+`, `-`, and `*`, operands drawn from 0–20 and 1–12.

## Five questions, one fresh turn each

These five questions follow the two formats the later stages actually train on: the `Kaç eder` warm-up, then a one-step word problem and a two-step word problem in the GSM8K Turkish style. Each prompt was a new context. Stacking a second question in the same `lokum.chat` process leaves the first turn in the history, and the model then loses the task. Decoding was greedy, the same rule as the evaluator, with at most 256 new tokens.

| Prompt | Gold | base | sft | dpo | ppo | grpo |
|---|---:|---|---|---|---|---|
| Kaç eder: 12 + 7? | 19 | wiki-table markup (`\|Tür\|`, `\|Boyut\|`) | echoes the question | `12 + 7 = 21` | repeats `22 + 22 + …` | echoes the question |
| Kaç eder: 15 - 6? | 9 | a loop of `\|<\|` | wanders off and reaches 135 | adds 15 over and over | repeats `25 + 15 + 6` | echoes the question |
| Kaç eder: 8 * 4? | 32 | wiki-table markup | `8 * 4 = 48` | rewrites it as `8 + 4 = 14` | a long run of zeros | echoes the question |
| Bir sepette 24 elma vardır. Ali 9 elma yerse sepette kaç elma kalır? | 15 | a loop of `\|<\|` | "sepette 24 elma kalır" | the same sentence | reaches 200 and repeats it | the same sentence |
| Ayşe'nin 3 kutusu var. Her kutuda 5 kalem vardır. Ayşe 4 kalem daha alırsa toplam kaç kalemi olur? | 19 | a loop of `\|<\|` | computes 5×5 and lands on 30 | computes 5×5 and 5×15 and lands on 115 | a long run of zeros | "Ayşe'nin 3 kutusu var." |

**0 / 5** for every checkpoint.

What that shows:

- **base** (`models/base.pt`) only learned to continue text. A chat question is more web text to it, so it emits table markup or a repeated glyph. It never opens an `<answer>` tag.
- **sft** learned the format. Answers sit inside `<answer>`. The arithmetic is still wrong: it copies `12 + 7`, calls `8 * 4` equal to 48, leaves all 24 apples in the basket, and multiplies 5 by 5 for the pencils.
- **dpo** is the one that tries an equation. `12 + 7 = 21` is the shape of an answer and the wrong number. It also changes `8 * 4` into `8 + 4 = 14`. Subtraction collapses into a sum of fifteens. The word problems stay in the same wrong sentences as SFT.
- **ppo** is the least stable. It locks onto one number and adds it forever, or it prints a huge run of zeros. The KL penalty and the verifier did not hold the policy on these prompts.
- **grpo** stays short. The three arithmetic lines are copied into the answer tag. The basket line matches SFT. The pencil line stops after restating that there are 3 boxes.

## Held-out GSM8K Turkish

The real test is `data/lokum/rl_prompts_test.jsonl`: 175 questions, every 50th row of [ytu-ce-cosmos/gsm8k_tr](https://huggingface.co/datasets/ytu-ce-cosmos/gsm8k_tr). `python -m lokum.evaluate` generates greedily, up to 420 new tokens, and counts a row correct when the parsed number matches the gold value within `1e-4`. The parser prefers the last `<answer>` block, then a `####` marker, then the last number in the text. A format bonus exists in `lokum/verifier.py`, and the logged accuracy ignores it: a row counts only when the number is right. Results from `logs/stage_table.jsonl`:

| Checkpoint | Correct | Accuracy |
|---|---:|---:|
| `models/base.pt` | 6 / 175 | 3.4% |
| `models/sft.pt` | 4 / 175 | 2.3% |
| `models/dpo.pt` | 6 / 175 | 3.4% |
| `models/ppo.pt` | 1 / 175 | 0.6% |
| `models/grpo.pt` | 4 / 175 | 2.3% |

This matches the five-prompt spot check. Supervised fine-tuning taught the answer tags and slightly lowered exact match relative to the base model. DPO came back to the base rate. GRPO stayed with SFT. PPO is the worst of the five, which is what the repeated additions and the digit runs look like up close. At this width, with a few days of training, preference tuning and reinforcement learning did not produce a model that can do these problems. ufakzeka reported the same pattern at this size: preference tuning hurt chat. The table above is that result on this run.

Base can still pick up a stray digit inside markup. The scorer will treat that digit as an answer, so a few of its 6 hits can be incidental. The spot check never showed base writing a worked solution.

`models/sft.pt` is the checkpoint to talk to. `models/reward.pt` is only the linear reward head, not a generator.

## Reproduce

Commands assume the shell is at the repository root and the virtualenv is active. This run used Python 3.12 and `torch==2.11.0+cu128`. Leave well over 50 GB free: the train file is about 12 GB of int32 tokens, each full checkpoint is about 1.9 GB because it stores Adam moments, and the Hugging Face cache sits beside both while the stream runs.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy h5py regex datasets tqdm
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Tokenizer

Put a few hundred megabytes of Turkish into `data/raw/turkish_sample.txt`. Training uses the first 200 MB and writes `data/lokum/tokenizer.json` plus a small smoke HDF5 under `data/lokum_smoke/`. Train the tokenizer once. A second run learns different merges, and every HDF5 built from the old merges becomes unreadable.

```bash
python -m lokum.prepare_tokenizer --input data/raw/turkish_sample.txt
```

Vocabulary is 16384 and the end-of-text id is 16383. `Ankara'da 42 şeker` splits so that `42` is the characters `4` and `2`, and `ş` stays inside a letter chunk.

### Pretrain corpus and base model

```bash
python -m lokum.prepare_pretrain
python -m lokum.pretrain
```

`prepare_pretrain` writes `data/lokum/train.h5` and `data/lokum/val.h5`. If a hub id 404s, change that one string in `MIX` inside `lokum/prepare_pretrain.py`. If FineWeb2-HQ will not stream, the fallback used by the original data list is `moganai/mogan-turkish-web`. `--start-at` and `--append` resume a mix that stopped partway.

`pretrain` trains 6,000 steps and writes `models/base.pt`. It also saves every 1,000 steps. Resume with `python -m lokum.pretrain --resume models/base.pt`. A cheap wiring check, same vocabulary and a 2-layer net, is:

```bash
python -m lokum.pretrain --smoke
python -m lokum.generate --ckpt models/smoke.pt --prompt "Bir zamanlar" --max-new 40
```

`models/base.pt` should continue a prompt rather than answer it:

```bash
python -m lokum.generate --ckpt models/base.pt --prompt "Bir zamanlar" --max-new 80
```

### Supervised fine-tuning

```bash
python -m lokum.prepare_sft
python -m lokum.sft
```

`prepare_sft` prints each dataset's column names and skips a repo that will not download. If a printed row uses different field names, add that key in `messages_of`. Rows longer than the context are dropped. `sft` starts from `models/base.pt` and runs 3 epochs, batch 4, grad accumulation 8, learning rate `1e-5`, warmup 100. The output is `models/sft.pt`.

### Preferences, reward head, DPO

```bash
python -m lokum.prepare_prefs
python -m lokum.reward
python -m lokum.dpo
```

`prepare_prefs` writes `{prompt, chosen, rejected}` JSONL and raises if a row is empty. The reward script freezes the SFT network, trains a new linear head on the last token with a Bradley-Terry loss for 2,000 steps at `1e-5`, and saves `models/reward.pt`. Accuracy above 0.5 means the head learned a preference. It is not the chat model.

DPO starts from `models/sft.pt`, keeps a frozen copy as the reference, and runs 1,000 steps at `5e-7` with `beta` 0.1 and sequences capped at 768. The output is `models/dpo.pt`.

### PPO and GRPO

```bash
python -m lokum.prepare_rl
python -m lokum.rl --algo ppo
python -m lokum.rl --algo grpo
```

`prepare_rl` downloads GSM8K Turkish and writes the train, test, and arithmetic JSONL files. Both algorithms start from `models/sft.pt`, sample a completion, score it with `lokum/verifier.py`, and take one clipped policy step. A KL penalty against the frozen SFT model is applied inside the reward (`--kl 0.04`). The first 50 iterations use the arithmetic warm-up. The remaining iterations, out of 200, use the GSM8K train prompts. Learning rate is `1e-6`, rollout length 256, clip 0.2.

PPO also trains a value head and uses GAE. Each iteration draws 4 prompts. GRPO has no value head: each iteration draws 2 prompts and 8 samples each, and a sample's advantage is its reward minus the group mean, divided by the group standard deviation. The group size has to stay at least 2. If GRPO runs out of memory, lower the prompt count before the group size. Outputs are `models/ppo.pt` and `models/grpo.pt`.

On this implementation the new policy and the sampling policy are the same weights, so the probability ratio starts at 1 and the clip does little until a second pass over stored log-probs is added. The loop is one prompt group at a time so it can be read. It is slower than a packed rollout.

### Score and chat

```bash
python -m lokum.evaluate --limit 200
```

That rewrites `logs/stage_table.jsonl` with one accuracy per checkpoint that exists.

Chat sampling follows the ufakzeka card: temperature 0.3, top-p 0.9, top-k 40. Ask one question, then quit with an empty line, and start the process again for the next question.

```bash
python -m lokum.chat --ckpt models/sft.pt
```

Swap `--ckpt` for `models/base.pt`, `models/dpo.pt`, `models/ppo.pt`, or `models/grpo.pt`.

## Layout

```text
lokum/                 the package
data/raw/              local Turkish text for the tokenizer
data/lokum_smoke/      small HDF5 from the same tokenizer
data/lokum/            tokenizer.json, pretrain HDF5, SFT rows, JSONL
models/                checkpoints
logs/                  stage_table.jsonl and other run logs
```

Git keeps those directories and ignores the files inside them, along with `.venv/`. Cloning the repo gives the code. Rebuilding the token files and the checkpoints is the reproduction path above.
